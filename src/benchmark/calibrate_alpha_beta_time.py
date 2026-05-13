import argparse
import csv
import ctypes
import json
import os
import re
import signal
import statistics
import subprocess
import sys


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
    cmd = [sys.executable, __file__, "--worker", "--budgets", ",".join(map(str, budgets))]
    if max_positions:
        cmd += ["--max-positions", str(max_positions)]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        # Decode the failure
        status = f"Exit Code {proc.returncode}"
        if proc.returncode < 0:
            try:
                status = f"Signal {signal.Signals(-proc.returncode).name} (e.g. Segfault)"
            except ValueError:
                status = f"Signal {-proc.returncode}"

        print(f"\n[CRITICAL] Worker failed: {status}", file=sys.stderr)
        print("--- WORKER STDERR (Look for OpenMP errors here) ---", file=sys.stderr)
        print(proc.stderr.strip() or "[No stderr output recorded]", file=sys.stderr)
        print("---------------------------------------------------------", file=sys.stderr)
        sys.exit(1)

    # Parse hybrid schema from the C++ stderr line:
    # "[alpha-beta] threads: 2 outer x 32 inner = 64 total"
    schema = f"{threads}T"
    m = re.search(r"(\d+) outer x (\d+) inner", proc.stderr)
    if m:
        schema = f"{m.group(1)}x{m.group(2)}"

    try:
        return json.loads(proc.stdout), schema
    except json.JSONDecodeError:
        print(f"Error: Worker sent invalid JSON: {proc.stdout}", file=sys.stderr)
        sys.exit(1)


def run_comparison(budgets, max_positions, thread_counts):
    print(f"[calibrate] Thread counts: {thread_counts}")
    print(f"[calibrate] Budgets: {budgets} ms\n")

    results = {}
    schemas = {}
    for t in thread_counts:
        print(f"[calibrate] running OMP_NUM_THREADS={t} ...")
        results[t], schemas[t] = spawn_worker(t, budgets, max_positions)

    col_w = 14
    labels = [schemas[t] for t in thread_counts]

    for t_ms in budgets:
        print(f"\n--- Budget: {t_ms}ms ---")
        header = f"{'Phase':<8}" + "".join(f"  {lbl:>{col_w}}" for lbl in labels) + f"  {'Depth Gain':>12}"
        print(header)
        print("-" * len(header))
        for phase in PHASES:
            row_parts = []
            all_ok = True
            for t in thread_counts:
                entry = results[t].get(phase, {}).get(str(t_ms))
                if not entry:
                    all_ok = False
                    break
                row_parts.append(entry)
            if not all_ok:
                continue
            gain = row_parts[-1]["avg"] - row_parts[0]["avg"]
            line = f"{phase:<8}" + "".join(f"  {e['avg']:>{col_w}.2f}" for e in row_parts)
            line += f"  {gain:>+11.2f}"
            print(line)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets",       type=str, default="10,100,500",
                        help="Comma-separated ms limits")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap FENs per phase (default: all)")
    parser.add_argument("--threads",       type=str, default="1,32,64,128,256",
                        help="comma-separated thread counts (default: 1,32,64,128,256)")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]

    if args.worker:
        run_worker(budgets, args.max_positions)
    else:
        thread_counts = [int(x) for x in args.threads.split(",")]
        run_comparison(budgets, args.max_positions, thread_counts)
