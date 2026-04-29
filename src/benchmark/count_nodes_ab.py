import argparse
import csv
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_perft


def run_count_nodes_alpha_beta(max_depth: int, num_threads: int):
    count_fn  = load_perft()["alpha_beta"]
    depth_cols = [f"depth_{d}" for d in range(1, max_depth + 1)]  # depth 0 = 1 node always

    def count_position(fen: str) -> dict:
        return {f"depth_{d}": count_fn(fen.encode(), d) for d in range(1, max_depth + 1)}

    for phase in PHASES:
        in_path  = os.path.join(DATA_DIR, phase, "puzzles.csv")
        out_path = os.path.join(DATA_DIR, phase, "node_counts_ab.csv")
        if not os.path.exists(in_path):
            print(f"[count_nodes_ab] {phase}: puzzles.csv not found")
            continue

        with open(in_path, newline="") as f:
            fens = [row["UpdatedFEN"] for row in csv.DictReader(f)]

        print(f"[count_nodes_ab] {phase}: counting alpha-beta nodes up to depth {max_depth} across {num_threads} threads ...")
        results = [None] * len(fens)
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(count_position, fen): i for i, fen in enumerate(fens)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        totals = {col: sum(r[col] for r in results) for col in depth_cols}
        n_pos  = len(fens)
        avgs   = [totals[col] / n_pos for col in depth_cols]

        with open(out_path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["UpdatedFEN"] + depth_cols)
            writer.writeheader()
            for fen, counts in zip(fens, results):
                writer.writerow({"UpdatedFEN": fen, **counts})

        summary_path = os.path.join(DATA_DIR, phase, "node_counts_ab_summary.csv")
        with open(summary_path, "w", newline="") as f_sum:
            writer = csv.DictWriter(f_sum, fieldnames=["depth", "sum", "average", "effective_branching_factor"])
            writer.writeheader()
            for i, col in enumerate(depth_cols):
                prev_avg = avgs[i - 1] if i > 0 else 1.0
                bf = f"{avgs[i] / prev_avg:.2f}" if prev_avg > 0 else ""
                writer.writerow({
                    "depth":                    col,
                    "sum":                      totals[col],
                    "average":                  f"{avgs[i]:.1f}",
                    "effective_branching_factor": bf,
                })

        print(f"  {'depth':<10} {'sum':>12} {'average':>12} {'eff_bf':>8}")
        for i, col in enumerate(depth_cols):
            prev_avg = avgs[i - 1] if i > 0 else 1.0
            bf = f"{avgs[i] / prev_avg:.2f}" if prev_avg > 0 else ""
            print(f"  {col:<10} {totals[col]:>12} {avgs[i]:>12.1f} {bf:>8}")
        print(f"[count_nodes_ab] {phase}: wrote {out_path}\nand {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--threads",   type=int, default=os.cpu_count())
    args = parser.parse_args()
    run_count_nodes_alpha_beta(args.max_depth, args.threads)
