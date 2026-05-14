"""
Benchmarks MPI MCTS (2 ranks x 32 OMP threads) vs single-threaded MCTS
using a depth limit per position.

Output: For each depth, for each game phase, print avg wall time per position
for 1T and 2Rx32T, plus speedup.

Note: the standard 1T engine runs 10,000 simulations per depth; the MPI engine
runs 1,000,000 total simulations split across ranks. The wall-time comparison
reflects this difference in simulation budget.
"""

import argparse
import csv
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_standard_monte_carlo, load_mpi_monte_carlo

MOVE_BUF_LEN   = 8
DEFAULT_DEPTHS = "1,2,3,4,5"


def load_fens(phase, max_positions=None):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f) if row.get("UpdatedFEN")]
    return fens[:max_positions] if max_positions else fens


def run_worker(depths, max_positions):
    """Single-threaded subprocess worker — prints JSON to stdout."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        mc_fn = load_standard_monte_carlo()["depth"]
    finally:
        sys.stdout = real_stdout

    buf = ctypes.create_string_buffer(MOVE_BUF_LEN)
    results = {}
    for phase in PHASES:
        fens = load_fens(phase, max_positions)
        if not fens:
            continue
        phase_results = {}
        for depth in depths:
            times = []
            for fen in fens:
                t0 = time.perf_counter()
                mc_fn(fen.encode(), depth, buf, MOVE_BUF_LEN)
                times.append(time.perf_counter() - t0)
            phase_results[depth] = {
                "avg": statistics.mean(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "n":   len(times),
            }
        results[phase] = phase_results
    sys.stdout.write(json.dumps(results) + "\n")
    sys.stdout.flush()


def spawn_1t_worker(depths, max_positions):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cmd = [sys.executable, __file__, "--worker", "--depths", ",".join(map(str, depths))]
    if max_positions is not None:
        cmd += ["--max-positions", str(max_positions)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"[benchmark_mpi_mc] 1T worker failed (exit {proc.returncode})\n"
              f"  stderr: {proc.stderr!r}\n  stdout: {proc.stdout!r}", file=sys.stderr)
        sys.exit(1)
    print(proc.stderr, end="", file=sys.stderr)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[benchmark_mpi_mc] worker JSON error: {exc}\n  stdout: {proc.stdout!r}", file=sys.stderr)
        sys.exit(1)


def run_mpi_timing(depths, max_positions):
    """Launch MPI engine with 2 ranks x 32 OMP threads and time each position."""
    os.environ["OMP_NUM_THREADS"] = "32"
    mpi = load_mpi_monte_carlo(nranks=2)
    results = {}
    try:
        for phase in PHASES:
            fens = load_fens(phase, max_positions)
            if not fens:
                continue
            phase_results = {}
            for depth in depths:
                times = []
                for fen in fens:
                    t0 = time.perf_counter()
                    mpi.depth(fen, depth)
                    times.append(time.perf_counter() - t0)
                phase_results[depth] = {
                    "avg": statistics.mean(times),
                    "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "n":   len(times),
                }
            results[phase] = phase_results
    finally:
        mpi.close()
    return results


def fmt(avg_s, std_s):
    avg_ms = avg_s * 1000
    std_ms = std_s * 1000
    return f"{avg_ms:8.2f}ms ± {std_ms:6.2f}ms"


def run_comparison(depths, max_positions):
    print("[benchmark_mpi_mc] running 1T baseline ...")
    data_1t = spawn_1t_worker(depths, max_positions)

    print("[benchmark_mpi_mc] running MPI 2 ranks x 32 OMP threads ...")
    data_mpi = run_mpi_timing(depths, max_positions)

    col_w  = 20
    labels = ["1T", "2Rx32T"]

    for depth in depths:
        print(f"\n=== Depth {depth} ===")
        header = f"{'Phase':<8}  {labels[0]:>{col_w}}  {labels[1]:>{col_w}}  {'Speedup':>8}"
        print(header)
        print("-" * len(header))
        for phase in PHASES:
            e1 = (data_1t.get(phase)  or {}).get(depth) or (data_1t.get(phase)  or {}).get(str(depth))
            e2 = (data_mpi.get(phase) or {}).get(depth) or (data_mpi.get(phase) or {}).get(str(depth))
            if e1 is None or e2 is None:
                continue
            speedup = e1["avg"] / e2["avg"] if e2["avg"] > 0 else float("nan")
            print(f"{phase:<8}  {fmt(e1['avg'], e1['std']):>{col_w}}  {fmt(e2['avg'], e2['std']):>{col_w}}  {speedup:>7.2f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths",        type=str, default=DEFAULT_DEPTHS,
                        help=f"comma-separated rollout depths (default: {DEFAULT_DEPTHS})")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap FENs per phase (default: all)")
    parser.add_argument("--worker", action="store_true",
                        help="internal: run as 1T timing subprocess, print JSON")
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",")]

    if args.worker:
        run_worker(depths, args.max_positions)
    else:
        run_comparison(depths, args.max_positions)
