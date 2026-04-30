import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import DATA_DIR


def load_avg_branching_factor(phase):
    """Return the mean effective branching factor across all depths for a phase."""
    path = os.path.join(DATA_DIR, phase, "node_counts_ab_summary.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    bfs = [float(r["effective_branching_factor"]) for r in rows if r["effective_branching_factor"]]
    return sum(bfs) / len(bfs)


def parse_branch_file(path):
    """
    Parse calibrate_alpha_beta_branch output.
    Returns dict: budget_ms -> phase -> {single_depth, multi_depth, depth_gain, schema_single, schema_multi}
    """
    results = {}
    schema_single = schema_multi = "?"
    current_budget = None

    schema_re = re.compile(r"Schemas\s*—\s*single:\s*(\S+)\s+multi:\s*(\S+)")
    budget_re  = re.compile(r"---\s*Budget:\s*(\d+)ms\s*---")
    data_re    = re.compile(
        r"^(early|mid|late)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)"
    )

    with open(path) as f:
        for line in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)

            m = schema_re.search(line)
            if m:
                schema_single, schema_multi = m.group(1), m.group(2)
                continue

            m = budget_re.search(line)
            if m:
                current_budget = int(m.group(1))
                results[current_budget] = {}
                continue

            if current_budget is None:
                continue

            m = data_re.match(line.strip())
            if m:
                phase, s_depth, m_depth, gain = m.groups()
                results[current_budget][phase] = {
                    "single_depth": float(s_depth),
                    "multi_depth":  float(m_depth),
                    "depth_gain":   float(gain),
                    "schema_single": schema_single,
                    "schema_multi":  schema_multi,
                }

    return results


def plot_branch_speedup(results, budget_ms, input_path, output_dir):
    data = results.get(budget_ms)
    if data is None:
        available = sorted(results.keys())
        print(f"Error: budget {budget_ms}ms not found. Available: {available}")
        sys.exit(1)

    phases = ["early", "mid", "late"]

    # Load avg branching factor per phase
    bf = {}
    for phase in phases:
        try:
            bf[phase] = load_avg_branching_factor(phase)
        except FileNotFoundError:
            print(f"Warning: node_counts_ab_summary.csv not found for {phase}, using bf=1")
            bf[phase] = 1.0

    # Compute estimated speedup = bf^depth_gain for each phase
    speedups = {}
    for phase in phases:
        gain = data.get(phase, {}).get("depth_gain", 0.0)
        speedups[phase] = bf[phase] ** gain

    # Get schema labels from first phase entry
    first = next(iter(data.values()))
    schema_single = first["schema_single"]
    schema_multi  = first["schema_multi"]

    labels = [
        f"Early\n({schema_single})",
        f"Early\n({schema_multi})",
        f"Mid\n({schema_single})",
        f"Mid\n({schema_multi})",
        f"Late\n({schema_single})",
        f"Late\n({schema_multi})",
    ]
    colors = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]

    avgs = []
    for phase in phases:
        d = data.get(phase, {})
        avgs += [d.get("single_depth", 0), d.get("multi_depth", 0)]

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, avgs, width, color=colors)
    ax.set_ylim(0, max(avgs) + 2 if avgs else 1)

    # Speedup labels on the 3 multi bars (indices 1, 3, 5)
    y_max = max(avgs) if avgs else 1
    for bar_idx, phase in zip([1, 3, 5], phases):
        sp = speedups[phase]
        gain = data.get(phase, {}).get("depth_gain", 0.0)
        bar = bars[bar_idx]
        top = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + y_max * 0.03,
            f"{sp:.1f}x speedup\n(+{gain:.2f} depth gain)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333"
        )

    ax.set_ylabel("Average Depth Reached", fontsize=12)
    ax.set_title(
        f"Alpha-Beta Hybrid Parallelism Speedup - {budget_ms}ms Budget",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Separator lines between phase groups
    for sep in [1.5, 3.5]:
        ax.axvline(sep, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # BF annotation per phase group
    y_top = ax.get_ylim()[1]
    for gx, phase, gname in [(0.5, "early", "Early"), (2.5, "mid", "Mid"), (4.5, "late", "Late")]:
        ax.text(gx, y_top * 0.98, f"{gname}\n(BF≈{bf[phase]:.1f})",
                ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{out_name}-budget{budget_ms}ms.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs", "calibrate", "alpha-beta-branch", "alpha-beta-branch-32.txt"
    )
    default_output = os.path.join(DATA_DIR, "graphs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=default_input,
                        help="Path to calibrate_alpha_beta_branch output file")
    parser.add_argument("--budget", type=int, required=True,
                        help="Which budget block to plot in ms (e.g. 500)")
    parser.add_argument("--output", default=default_output,
                        help="Directory to save the graph (default: data/graphs/)")
    args = parser.parse_args()

    results = parse_branch_file(args.input)
    plot_branch_speedup(results, args.budget, args.input, args.output)
