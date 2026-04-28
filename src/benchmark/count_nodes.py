import argparse
import csv
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.load import load_perft


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PHASES   = ["early", "mid", "late"]


def run_count_nodes(max_depth: int, num_threads: int):
    count_nodes = load_perft()
    depth_cols  = [f"depth_{d}" for d in range(max_depth + 1)]

    def count_position(fen: str) -> dict:
        return {f"depth_{d}": count_nodes(fen.encode(), d) for d in range(max_depth + 1)}

    for phase in PHASES:
        in_path  = os.path.join(DATA_DIR, phase, "updatedFEN.csv")
        out_path = os.path.join(DATA_DIR, phase, "node_counts.csv")
        if not os.path.exists(in_path):
            print(f"[count_nodes] {phase}: updatedFEN.csv not found, run update_fen.py first")
            continue

        with open(in_path, newline="") as f:
            fens = [row["UpdatedFEN"] for row in csv.DictReader(f)]

        print(f"[count_nodes] {phase}: counting nodes up to depth {max_depth} across {num_threads} threads ...")
        results = [None] * len(fens)
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(count_position, fen): i for i, fen in enumerate(fens)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        totals = {col: sum(r[col] for r in results) for col in depth_cols}
        n_pos  = len(fens)

        with open(out_path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["UpdatedFEN"] + depth_cols)
            writer.writeheader()
            for fen, counts in zip(fens, results):
                writer.writerow({"UpdatedFEN": fen, **counts})

        summary_path = os.path.join(DATA_DIR, phase, "node_counts_summary.csv")
        with open(summary_path, "w", newline="") as f_sum:
            writer = csv.DictWriter(f_sum, fieldnames=["depth", "sum", "average"])
            writer.writeheader()
            for col in depth_cols:
                writer.writerow({"depth": col, "sum": totals[col], "average": totals[col] / n_pos})

        print(f"  {'depth':<10} {'sum':>12} {'average':>12}")
        for col in depth_cols:
            print(f"  {col:<10} {totals[col]:>12} {totals[col] / n_pos:>12.1f}")
        print(f"[count_nodes] {phase}: wrote {out_path} \nand {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth",  type=int, default=5)
    parser.add_argument("--threads",    type=int, default=os.cpu_count())
    args = parser.parse_args()
    run_count_nodes(args.max_depth, args.threads)
