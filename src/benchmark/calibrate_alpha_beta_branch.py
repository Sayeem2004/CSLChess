import argparse
import csv
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

# Ensure benchmark and utils are in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_standard_alpha_beta


MOVE_BUF_LEN = 8


def load_fens(phase, max_positions=None):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f) if row.get("UpdatedFEN")]
    return fens[:max_positions] if max_positions else fens


def run_hybrid_benchmark(ab_fn, fens, time_ms):
    """Executes search and returns the depths reached."""
    buf = ctypes.create_string_buffer(MOVE_BUF_LEN)
    depths = []
    for fen in fens:
        # C++ best_move_alpha_beta_time returns the depth reached as an int
        reached = ab_fn(fen.encode(), time_ms, buf, MOVE_BUF_LEN)
        depths.append(float(reached))
    return depths


def run_worker(time_budgets, max_positions):
    # Suppress build output on stdout
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Load the TIME-based function
        ab_fn = load_standard_alpha_beta()["time"]
    finally:
        sys.stdout = real_stdout

    results = {}
    for phase in PHASES:
        fens = load_fens(phase, max_positions)
        if not fens: continue

        phase_results = {}
        for t_limit in time_budgets:
            depths = run_hybrid_benchmark(ab_fn, fens, t_limit)
            phase_results[t_limit] = {
                "avg": statistics.mean(depths),
                "std": statistics.stdev(depths) if len(depths) > 1 else 0.0,
                "n":   len(depths),
            }
        results[phase] = phase_results

    sys.stdout = real_stdout
    sys.stdout.write(json.dumps(results) + "\n")


def spawn_worker(threads, budgets, max_positions):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [
        sys.executable, __file__,
        "--worker",
        "--budgets", ",".join(map(str, budgets))
    ]
    if max_positions:
        cmd += ["--max-positions", str(max_positions)]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"Worker failed: {proc.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)


def run_comparison(budgets, max_positions, single_t, multi_t):
    print(f"[calibrate] Single-thread (1T) vs Multi-thread ({multi_t}T Hybrid)")
    print(f"[calibrate] Budgets: {budgets} ms\n")

    single = spawn_worker(single_t, budgets, max_positions)
    multi  = spawn_worker(multi_t, budgets, max_positions)

    header = f"{'Phase':<8}  {'1T Avg Depth':>15}  {'Multi Avg Depth':>18}  {'Depth Gain':>12}"
    sep = "-" * len(header)

    for t in budgets:
        print(f"--- Budget: {t}ms ---")
        print(header)
        print(sep)
        for phase in PHASES:
            s = single.get(phase, {}).get(str(t))
            m = multi.get(phase, {}).get(str(t))
            if not s or not m: continue

            gain = m['avg'] - s['avg']
            print(f"{phase:<8}  {s['avg']:15.2f}  {m['avg']:18.2f}  {gain:>+11.2f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=str, default="10,50,100,500", help="Comma-separated ms limits")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]

    if args.worker:
        run_worker(budgets, args.max_positions)
    else:
        run_comparison(budgets, args.max_positions, 1, args.threads)
