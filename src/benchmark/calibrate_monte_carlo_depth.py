"""
Compares single-threaded vs max-thread Monte Carlo Tree Search (MCTS) performance.

Each thread configuration runs in its own subprocess so OMP_NUM_THREADS is
read before the OpenMP runtime initialises — changing the env var inside an
already-running process has no effect once the thread pool is created.

Output (stdout only, no CSV):
  For each depth 1..MAX_DEPTH, for each game phase, print:
    avg ± std for single-thread  and  avg ± std for multi-thread  + speedup.
  12 numbers per depth (3 phases × 2 configs × avg+std).
"""

import argparse
import csv
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_standard_monte_carlo, load_standard_monte_carlo_rp

MOVE_BUF_LEN = 8


def load_fens(phase, max_positions=None):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f) if row.get("UpdatedFEN")]
    return fens[:max_positions] if max_positions else fens


def time_monte_carlo_depth(mc_fn, fens, depth):
    buf = ctypes.create_string_buffer(MOVE_BUF_LEN)
    times = []
    for fen in fens:
        t0 = time.perf_counter()
        mc_fn(fen.encode(), depth, buf, MOVE_BUF_LEN)
        times.append(time.perf_counter() - t0)
    return times


# Invoked as a subprocess with OMP_NUM_THREADS already set in env.
# Prints a JSON dict to stdout.
def run_worker(max_depth, max_positions, rp):
    # Redirect stdout → stderr during library load so build messages don't corrupt the JSON.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if rp:
            mc_fn = load_standard_monte_carlo_rp()["depth"]
        else:
            mc_fn = load_standard_monte_carlo()["depth"]
    finally:
        sys.stdout = real_stdout

    results = {}
    for phase in PHASES:
        fens = load_fens(phase, max_positions)
        if not fens:
            continue
        phase_results = {}
        for depth in range(1, max_depth + 1):
            times = time_monte_carlo_depth(mc_fn, fens, depth)
            phase_results[depth] = {
                "avg": statistics.mean(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "n":   len(times),
            }
        results[phase] = phase_results
    sys.stdout.write(json.dumps(results) + "\n")
    sys.stdout.flush()


def spawn_worker(threads, max_depth, max_positions, rp):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [
        sys.executable, __file__,
        "--worker",
        "--max-depth", str(max_depth),
    ]
    if max_positions is not None:
        cmd += ["--max-positions", str(max_positions)]
    if rp:
        cmd += ["--rp"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"[calibrate_monte_carlo_depth] worker (threads={threads}) failed "
              f"(exit {proc.returncode})\n"
              f"  stderr: {proc.stderr!r}\n"
              f"  stdout: {proc.stdout!r}",
              file=sys.stderr)
        sys.exit(1)
    # Build messages go to stderr; only the JSON dict is on stdout
    print(proc.stderr, end="", file=sys.stderr)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[calibrate_monte_carlo_depth] worker output is not valid JSON: {exc}\n"
              f"  stdout: {proc.stdout!r}", file=sys.stderr)
        sys.exit(1)


def fmt(avg_s, std_s):
    avg_ms = avg_s * 1000
    std_ms = std_s * 1000
    return f"{avg_ms:8.2f}ms ± {std_ms:6.2f}ms"


def run_comparison(max_depth, max_positions, single_threads, multi_threads, rp):
    print(f"[calibrate_monte_carlo_depth] running single-thread  (OMP_NUM_THREADS={single_threads}) ...")
    single = spawn_worker(single_threads, max_depth, max_positions, rp)

    print(f"[calibrate_monte_carlo_depth] running multi-thread   (OMP_NUM_THREADS={multi_threads}) ...")
    multi  = spawn_worker(multi_threads,  max_depth, max_positions, rp)

    header = (f"{'Phase':<8}  {'Single (1T) avg ± std':>26}  "
              f"{'Multi (' + str(multi_threads) + 'T) avg ± std':>26}  {'Speedup':>8}")
    sep    = "-" * len(header)

    for depth in range(1, max_depth + 1):
        print(f"\n=== Depth {depth} ===")
        print(header)
        print(sep)
        for phase in PHASES:
            if phase not in single or str(depth) not in single[phase] and depth not in single[phase]:
                continue
            # json keys round-trip as strings; handle both int and str keys
            s = single[phase].get(depth) or single[phase].get(str(depth))
            m = multi[phase].get(depth)  or multi[phase].get(str(depth))
            if s is None or m is None:
                continue
            speedup = s["avg"] / m["avg"] if m["avg"] > 0 else float("nan")
            print(f"{phase:<8}  {fmt(s['avg'], s['std']):>26}  {fmt(m['avg'], m['std']):>26}  {speedup:>7.2f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth",     type=int, default=5,
                        help="maximum depth / iterations for MCTS (default: 5)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap FENs per phase (default: all)")
    parser.add_argument("--single-threads", type=int, default=1,
                        help="thread count for single-threaded baseline (default: 1)")
    parser.add_argument("--multi-threads",  type=int, default=os.cpu_count(),
                        help="thread count for parallel run (default: cpu count)")
    parser.add_argument("--worker", action="store_true",
                        help="internal: run as timing subprocess, print JSON")
    parser.add_argument("--rp", action="store_true",
                        help="use load_standard_monte_carlo_rp variant")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.max_depth, args.max_positions, args.rp)
    else:
        run_comparison(args.max_depth, args.max_positions,
                       args.single_threads, args.multi_threads, args.rp)
