import argparse
import csv
import ctypes
import os
import platform
import re
import statistics
import subprocess
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_perft, load_standard_alpha_beta, STOCKFISH_LINUX_BIN, STOCKFISH_MAC_BIN


MOVE_BUF_LEN = 8


def load_phase_fens(phase, max_positions=None):
    path = os.path.join(DATA_DIR, phase, "puzzles.csv")
    if not os.path.exists(path):
        print(f"[compare_depths] {phase}: puzzles.csv not found, skipping")
        return []
    with open(path, newline="") as f:
        fens = [row["UpdatedFEN"] for row in csv.DictReader(f) if row.get("UpdatedFEN")]
    if max_positions:
        fens = fens[:max_positions]
    return fens


def time_perft(fn, fens, depth):
    times = []
    for fen in fens:
        t0 = time.perf_counter()
        fn(fen.encode(), depth)
        times.append(time.perf_counter() - t0)
    return times


def run_alpha_beta(time_fn, count_fn, fens, depth):
    """Returns (times, node_counts) for alpha-beta across all fens."""
    buf   = ctypes.create_string_buffer(MOVE_BUF_LEN)
    times  = []
    counts = []
    for fen in fens:
        enc = fen.encode()
        t0 = time.perf_counter()
        time_fn(enc, depth, buf, MOVE_BUF_LEN)
        times.append(time.perf_counter() - t0)
        counts.append(int(count_fn(enc, depth)))
    return times, counts


def run_stockfish(sf_path, fens, depth, threads=1):
    """Returns (times, node_counts) for Stockfish across all fens.
    Node count is taken from the last 'info' line before bestmove."""
    proc = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    proc.stdin.write(f"uci\nsetoption name Threads value {threads}\nisready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.strip() == "readyok":
            break

    node_re = re.compile(r"\bnodes\s+(\d+)")
    times  = []
    counts = []
    for fen in fens:
        t0 = time.perf_counter()
        proc.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        proc.stdin.flush()
        last_nodes = 0
        for line in proc.stdout:
            m = node_re.search(line)
            if m:
                last_nodes = int(m.group(1))
            if line.startswith("bestmove"):
                break
        times.append(time.perf_counter() - t0)
        counts.append(last_nodes)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.communicate()
    return times, counts


def summarize(vals):
    avg = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return avg, std


def compare_phase(phase, max_depth, max_positions, sf_path, sys_subfolder, sf_threads=1):
    fens = load_phase_fens(phase, max_positions)
    if not fens:
        return
    print(f"[compare_depths] {phase}: {len(fens)} positions, depths 1-{max_depth}, stockfish threads={sf_threads}")

    ab_time_fn  = load_standard_alpha_beta()["depth"]
    ab_count_fn = load_perft()["alpha_beta"]
    perft_fn    = load_perft()["perft"]

    fieldnames = [
        "depth",
        "perft_avg_s",          "perft_std_s",
        "alpha_beta_avg_s",     "alpha_beta_std_s",
        "alpha_beta_avg_nodes", "alpha_beta_std_nodes",
        "stockfish_avg_s",      "stockfish_std_s",
        "stockfish_avg_nodes",  "stockfish_std_nodes",
    ]
    rows = []


    for depth in range(1, max_depth + 1):
        print(f"  depth {depth}/{max_depth} ...")

        perft_times               = time_perft(perft_fn, fens, depth)
        ab_times, ab_nodes        = run_alpha_beta(ab_time_fn, ab_count_fn, fens, depth)
        sf_times, sf_nodes        = run_stockfish(sf_path, fens, depth, sf_threads)

        perft_avg, perft_std      = summarize(perft_times)
        ab_avg,    ab_std         = summarize(ab_times)
        ab_n_avg,  ab_n_std       = summarize(ab_nodes)
        sf_avg,    sf_std         = summarize(sf_times)
        sf_n_avg,  sf_n_std       = summarize(sf_nodes)

        sf_scaled_ms = (sf_avg / sf_n_avg * ab_n_avg * 1000) if sf_n_avg > 0 else float("nan")
        print(f"    perft:      avg={perft_avg*1000:8.2f}ms  std={perft_std*1000:7.2f}ms")
        print(f"    alpha-beta: avg={ab_avg*1000:8.2f}ms  std={ab_std*1000:7.2f}ms  "
              f"nodes avg={ab_n_avg:>10.0f}  std={ab_n_std:>10.0f}")
        print(f"    stockfish:  avg={sf_avg*1000:8.2f}ms  std={sf_std*1000:7.2f}ms  "
              f"nodes avg={sf_n_avg:>10.0f}  std={sf_n_std:>10.0f}  "
              f"scaled to our nodes={sf_scaled_ms:8.2f}ms")

        rows.append({
            "depth":                depth,
            "perft_avg_s":          f"{perft_avg:.6f}",
            "perft_std_s":          f"{perft_std:.6f}",
            "alpha_beta_avg_s":     f"{ab_avg:.6f}",
            "alpha_beta_std_s":     f"{ab_std:.6f}",
            "alpha_beta_avg_nodes": f"{ab_n_avg:.1f}",
            "alpha_beta_std_nodes": f"{ab_n_std:.1f}",
            "stockfish_avg_s":      f"{sf_avg:.6f}",
            "stockfish_std_s":      f"{sf_std:.6f}",
            "stockfish_avg_nodes":  f"{sf_n_avg:.1f}",
            "stockfish_std_nodes":  f"{sf_n_std:.1f}",
        })

    out_dir  = os.path.join(DATA_DIR, phase, sys_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_depths.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare_depths] {phase}: wrote {out_path}\n")


def compare_depths(max_depth, max_positions, sf_path, sys_subfolder, sf_threads=1):
    for phase in PHASES:
        compare_phase(phase, max_depth, max_positions, sf_path, sys_subfolder, sf_threads)


if __name__ == "__main__":
    sf_default = STOCKFISH_MAC_BIN if platform.system() == "Darwin" else STOCKFISH_LINUX_BIN

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth",     type=int, default=4,
                        help="max depth to test (default: 4)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap number of FENs across all phases (default: all)")
    parser.add_argument("--stockfish",        default=sf_default,
                        help="path to Stockfish binary")
    parser.add_argument("--stockfish-threads", type=int,
                        default=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())),
                        help="threads to give Stockfish (default: OMP_NUM_THREADS or cpu_count)")
    args = parser.parse_args()

    sys_subfolder = "darwin" if platform.system() == "Darwin" else "linux"
    compare_depths(args.max_depth, args.max_positions, args.stockfish, sys_subfolder, args.stockfish_threads)