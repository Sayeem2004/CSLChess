import argparse
import csv
import ctypes
import os
import platform
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


def time_alpha_beta(fn, fens, depth):
    buf   = ctypes.create_string_buffer(MOVE_BUF_LEN)
    times = []
    for fen in fens:
        t0 = time.perf_counter()
        fn(fen.encode(), depth, buf, MOVE_BUF_LEN)
        times.append(time.perf_counter() - t0)
    return times


def time_stockfish(sf_path, fens, depth):
    """Run all positions in one persistent Stockfish process to amortize startup overhead."""
    proc = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    proc.stdin.write("uci\nisready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.strip() == "readyok":
            break

    times = []
    for fen in fens:
        proc.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        proc.stdin.flush()
        t0 = time.perf_counter()
        for line in proc.stdout:
            if line.startswith("bestmove"):
                break
        times.append(time.perf_counter() - t0)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.communicate()
    return times


def summarize(times):
    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return avg, std


def compare_phase(phase, max_depth, max_positions, sf_path, sys_subfolder):
    fens = load_phase_fens(phase, max_positions)
    if not fens:
        return
    print(f"[compare_depths] {phase}: {len(fens)} positions, depths 1-{max_depth}")

    perft_fn = load_perft()["perft"]
    ab_fn    = load_standard_alpha_beta()["depth"]

    fieldnames = [
        "depth",
        "perft_avg_s",      "perft_std_s",
        "alpha_beta_avg_s", "alpha_beta_std_s",
        "stockfish_avg_s",  "stockfish_std_s",
    ]
    rows = []

    for depth in range(1, max_depth + 1):
        print(f"  depth {depth}/{max_depth} ...")

        perft_times = time_perft(perft_fn, fens, depth)
        ab_times    = time_alpha_beta(ab_fn, fens, depth)
        sf_times    = time_stockfish(sf_path, fens, depth)

        perft_avg, perft_std = summarize(perft_times)
        ab_avg,    ab_std    = summarize(ab_times)
        sf_avg,    sf_std    = summarize(sf_times)

        print(f"    perft:      avg={perft_avg*1000:8.2f}ms  std={perft_std*1000:7.2f}ms")
        print(f"    alpha-beta: avg={ab_avg*1000:8.2f}ms  std={ab_std*1000:7.2f}ms")
        print(f"    stockfish:  avg={sf_avg*1000:8.2f}ms  std={sf_std*1000:7.2f}ms")

        rows.append({
            "depth":              depth,
            "perft_avg_s":        f"{perft_avg:.6f}",
            "perft_std_s":        f"{perft_std:.6f}",
            "alpha_beta_avg_s":   f"{ab_avg:.6f}",
            "alpha_beta_std_s":   f"{ab_std:.6f}",
            "stockfish_avg_s":    f"{sf_avg:.6f}",
            "stockfish_std_s":    f"{sf_std:.6f}",
        })

    out_dir  = os.path.join(DATA_DIR, phase, sys_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_depths.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare_depths] {phase}: wrote {out_path}\n")


def compare_depths(max_depth, max_positions, sf_path, sys_subfolder):
    for phase in PHASES:
        compare_phase(phase, max_depth, max_positions, sf_path, sys_subfolder)


if __name__ == "__main__":
    sf_default = STOCKFISH_MAC_BIN if platform.system() == "Darwin" else STOCKFISH_LINUX_BIN

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth",     type=int, default=5,
                        help="max depth to test (default: 5)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="cap number of FENs across all phases (default: all)")
    parser.add_argument("--stockfish",     default=sf_default,
                        help="path to Stockfish binary")
    args = parser.parse_args()

    sys_subfolder = "darwin" if platform.system() == "Darwin" else "linux"
    compare_depths(args.max_depth, args.max_positions, args.stockfish, sys_subfolder)
