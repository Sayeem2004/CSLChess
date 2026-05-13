"""
Compares MCTS performance across multiple thread counts using a time budget.

Each thread configuration runs in its own subprocess so OMP_NUM_THREADS is
read before the OpenMP runtime initialises.

Output: For each time budget, for each game phase, print avg wall time per
position for every thread count plus speedup relative to the first (baseline).
"""

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
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_standard_monte_carlo, load_standard_monte_carlo_rp

MOVE_BUF_LEN    = 8
DEFAULT_THREADS = "1,32,64,128,256"


def load_fens(phase, max_positions=None):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f) if row.get("UpdatedFEN")]
    return fens[:max_positions] if max_positions else fens


def time_monte_carlo_time(mc_fn, fens, time_ms):
    buf = ctypes.create_string_buffer(MOVE_BUF_LEN)
    times = []
    for fen in fens:
        t0 = time.perf_counter()
        mc_fn(fen.encode(), time_ms, buf, MOVE_BUF_LEN)
        times.append(time.perf_counter() - t0)
    return times


def run_worker(time_budgets, max_positions, rp):
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if rp:
            mc_fn = load_standard_monte_carlo_rp()["time"]
        else:
            mc_fn = load_standard_monte_carlo()["time"]
    finally:
        sys.stdout = real_stdout

    results = {}
    for phase in PHASES:
        fens = load_fens(phase, max_positions)
        if not fens:
            continue
        phase_results = {}
        for t_limit in time_budgets:
            times = time_monte_carlo_time(mc_fn, fens, t_limit)
            phase_results[t_limit] = {
                "avg": statistics.mean(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "n":   len(times),
            }
        results[phase] = phase_results

    sys.stdout = real_stdout
    sys.stdout.write(json.dumps(results) + "\n")
    sys.stdout.flush()


def spawn_worker(threads, budgets, max_positions, rp):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [sys.executable, __file__, "--worker", "--budgets", ",".join(map(str, budgets))]
    if max_positions is not None:
        cmd += ["--max-positions", str(max_positions)]
    if rp:
        cmd += ["--rp"]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        status = f"Exit Code {proc.returncode}"
        if proc.returncode < 0:
            try:
                status = f"Signal {signal.Signals(-proc.returncode).name} (e.g. Segfault)"
            except ValueError:
                status = f"Signal {-proc.returncode}"
        print(f"\n[CRITICAL] Worker failed: {status}", file=sys.stderr)
        print("--- WORKER STDERR ---", file=sys.stderr)
        print(proc.stderr.strip() or "[No stderr output recorded]", file=sys.stderr)
        print("---------------------", file=sys.stderr)
        sys.exit(1)

    print(proc.stderr, end="", file=sys.stderr)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[calibrate_monte_carlo_time] worker output is not valid JSON: {exc}\n"
              f"  stdout: {proc.stdout!r}", file=sys.stderr)
        sys.exit(1)


def fmt(avg_s, std_s):
    avg_ms = avg_s * 1000
    std_ms = std_s * 1000
    return f"{avg_ms:7.2f}ms±{std_ms:5.2f}ms"


def run_comparison(budgets, max_positions, thread_counts, rp):
    print(f"[calibrate_monte_carlo_time] Thread counts: {thread_counts}")
    print(f"[calibrate_monte_carlo_time] Budgets: {budgets} ms\n")

    data = {}
    for t in thread_counts:
        print(f"[calibrate_monte_carlo_time] running OMP_NUM_THREADS={t} ...")
        data[t] = spawn_worker(t, budgets, max_positions, rp)

    col_w = 18
    t_labels = [f"{t}T" for t in thread_counts]

    for t_ms in budgets:
        print(f"\n=== Budget: {t_ms}ms ===")
        header = f"{'Phase':<8}" + "".join(f"  {lbl:>{col_w}}" for lbl in t_labels) + f"  {'Speedup':>8}"
        print(header)
        print("-" * len(header))
        for phase in PHASES:
            row_parts = []
            base_avg = None
            all_ok = True
            for t in thread_counts:
                d = data[t].get(phase, {})
                entry = d.get(t_ms) or d.get(str(t_ms))
                if entry is None:
                    all_ok = False
                    break
                row_parts.append(entry)
                if base_avg is None:
                    base_avg = entry["avg"]
            if not all_ok or base_avg is None:
                continue
            last_avg = row_parts[-1]["avg"]
            speedup  = base_avg / last_avg if last_avg > 0 else float("nan")
            line = f"{phase:<8}"
            for e in row_parts:
                line += f"  {fmt(e['avg'], e['std']):>{col_w}}"
            line += f"  {speedup:>7.2f}x"
            print(line)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets",       type=str, default="10,100,500",
                        help="comma-separated time limits in ms (default: 10,100,500)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap FENs per phase (default: all)")
    parser.add_argument("--threads",       type=str, default=DEFAULT_THREADS,
                        help=f"comma-separated thread counts (default: {DEFAULT_THREADS})")
    parser.add_argument("--rp", action="store_true",
                        help="use load_standard_monte_carlo_rp variant")
    parser.add_argument("--worker", action="store_true",
                        help="internal: run as timing subprocess, print JSON")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]

    if args.worker:
        run_worker(budgets, args.max_positions, args.rp)
    else:
        thread_counts = [int(x) for x in args.threads.split(",")]
        run_comparison(budgets, args.max_positions, thread_counts, args.rp)