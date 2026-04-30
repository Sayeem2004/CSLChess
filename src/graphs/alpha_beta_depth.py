import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import DATA_DIR


def parse_output_file(path):
    """Parse calibrate_alpha_beta_depth output, returning dict depth -> phase -> {s_avg, s_std, m_avg, m_std, speedup}."""
    results = {}
    current_depth = None

    line_re = re.compile(
        r"^(early|mid|late)\s+"
        r"([\d.]+)ms\s*±\s*([\d.]+)ms\s+"
        r"([\d.]+)ms\s*±\s*([\d.]+)ms\s+"
        r"([\d.]+)x"
    )

    with open(path) as f:
        for line in f:
            # Strip ANSI escape codes
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = re.match(r"=== Depth (\d+) ===", line.strip())
            if m:
                current_depth = int(m.group(1))
                results[current_depth] = {}
                continue
            if current_depth is None:
                continue
            m = line_re.match(line.strip())
            if m:
                phase, s_avg, s_std, m_avg, m_std, speedup = m.groups()
                results[current_depth][phase] = {
                    "s_avg":   float(s_avg),
                    "s_std":   float(s_std),
                    "m_avg":   float(m_avg),
                    "m_std":   float(m_std),
                    "speedup": float(speedup),
                }

    return results


def plot_alpha_beta_depth(results, depth, input_path, output_dir):
    data = results.get(depth)
    if data is None:
        print(f"Error: depth {depth} not found in {input_path}")
        sys.exit(1)

    phases  = ["early", "mid", "late"]
    labels  = ["Early\n(1 Thread)", "Early\n(256 Threads)", "Mid\n(1 Thread)", "Mid\n(256 Threads)", "Late\n(1 Thread)", "Late\n(256 Threads)"]
    colors  = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e74c3c", "#c0392b"]

    avgs = []
    stds = []
    for phase in phases:
        d = data.get(phase, {})
        avgs += [d.get("s_avg", 0), d.get("m_avg", 0)]
        stds += [d.get("s_std", 0), d.get("m_std", 0)]

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, avgs, width, yerr=stds, capsize=5, color=colors,
                  error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2})

    # Speedup labels on the 3 multi bars (indices 1, 3, 5)
    for bar_idx, phase in zip([1, 3, 5], phases):
        speedup = data.get(phase, {}).get("speedup", None)
        if speedup is None:
            continue
        bar = bars[bar_idx]
        top = bar.get_height() + stds[bar_idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + max(avgs) * 0.05,
            f"{speedup:.2f}x Speedup",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333"
        )

    # Extract thread counts from filename for axis label
    fname = os.path.basename(input_path)
    m = re.search(r"(\d+)T.*?(\d+)T", fname)

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Alpha-Beta Root Parallelism Speedup - Depth {depth}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Separator lines between phase groups
    for sep in [1.5, 3.5]:
        ax.axvline(sep, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Phase group labels
    y_top = ax.get_ylim()[1]
    for gx, gname in [(0.5, "Early"), (2.5, "Mid"), (4.5, "Late")]:
        ax.text(gx, y_top * 0.98, gname, ha="center", va="top", fontsize=11,
                fontweight="bold", color="#222222")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{out_name}-depth{depth}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs", "calibrate", "alpha-beta-depth", "alpha-beta-depth-20cache-5.txt"
    )
    default_output = os.path.join(DATA_DIR, "graphs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=default_input,
                        help="Path to calibrate_alpha_beta_depth output file")
    parser.add_argument("--depth",  type=int, required=True,
                        help="Which depth block to plot (e.g. 5)")
    parser.add_argument("--output", default=default_output,
                        help="Directory to save the graph (default: data/graphs/)")
    args = parser.parse_args()

    results = parse_output_file(args.input)
    plot_alpha_beta_depth(results, args.depth, args.input, args.output)
