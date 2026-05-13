import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import DATA_DIR


def schema_to_nthreads(schema):
    """'32T' -> 32, '2x32' -> 64, '1x1' -> 1."""
    m = re.match(r"(\d+)x(\d+)", schema)
    if m:
        return int(m.group(1)) * int(m.group(2))
    m = re.match(r"(\d+)T$", schema)
    if m:
        return int(m.group(1))
    return 1


def parse_depth_file(path):
    """
    Parse calibrate_alpha_beta_depth output.
    Returns:
        schemas: list of schema labels (e.g. ["1T","32T","64T","128T","256T"])
        data: dict depth -> phase -> list of avg_ms per schema (same order)
    """
    schemas       = []
    data          = {}
    current_depth = None
    header_parsed = False

    depth_re = re.compile(r"=== Depth (\d+) ===")
    phase_re = re.compile(r"^(early|mid|late)\s+(.+)$")

    with open(path) as f:
        for raw in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()

            m = depth_re.search(line)
            if m:
                current_depth = int(m.group(1))
                data[current_depth] = {}
                header_parsed = False
                continue

            if current_depth is None:
                continue

            if not header_parsed and "Phase" in line:
                parts = line.split()
                raw_schemas = []
                i = 0
                while i < len(parts):
                    if parts[i] == "Phase":
                        i += 1
                        continue
                    if parts[i] == "Speedup":
                        break
                    raw_schemas.append(parts[i])
                    i += 1
                if not schemas:
                    schemas = raw_schemas
                header_parsed = True
                continue

            if not header_parsed or line.strip().startswith("-"):
                continue

            m = phase_re.match(line.strip())
            if m:
                phase = m.group(1)
                avgs = re.findall(r"([\d.]+)ms±", m.group(2))
                data[current_depth][phase] = [float(x) for x in avgs[:len(schemas)]]

    return schemas, data


def plot_pe(schemas, data, depth, input_path, output_dir):
    phases = ["early", "mid", "late"]

    d = data.get(depth)
    if d is None:
        print(f"Error: depth {depth} not found. Available: {sorted(data)}")
        sys.exit(1)

    n_threads = [schema_to_nthreads(s) for s in schemas]
    n_cols    = len(schemas)

    avg_speedups = []
    avg_pes      = []

    for col_idx in range(n_cols):
        N = n_threads[col_idx]
        speedups_phases = []
        for phase in phases:
            avgs = d.get(phase)
            if avgs is None or len(avgs) <= col_idx:
                continue
            base_avg = d[phase][0]
            this_avg = avgs[col_idx]
            if this_avg > 0:
                speedups_phases.append(base_avg / this_avg)

        if not speedups_phases:
            avg_speedups.append(0.0)
            avg_pes.append(0.0)
            continue

        avg_sp = sum(speedups_phases) / len(speedups_phases)
        avg_pe = avg_sp / N if N > 0 else 0.0
        avg_speedups.append(avg_sp)
        avg_pes.append(avg_pe)

    x      = np.arange(n_cols)
    width  = 0.55
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_cols))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_pes, width, color=colors, edgecolor="white", linewidth=0.8)

    y_max = max(avg_pes) if avg_pes else 1.0
    for bar, sp in zip(bars, avg_speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.02,
            f"{sp:.1f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222222"
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Perfect efficiency")
    ax.set_ylim(0, max(1.1, y_max * 1.18))
    ax.set_ylabel("Parallel Efficiency  (speedup / N)", fontsize=12)
    ax.set_title(
        f"Alpha-Beta Parallel Efficiency — Depth {depth}\n"
        f"(speedup = wall-time ratio, avg across early/mid/late)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(schemas, fontsize=10)
    ax.set_xlabel("Thread Configuration", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{out_name}-pe-depth{depth}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs", "calibrate", "alpha-beta-depth", "alpha-beta-depth.txt"
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

    schemas, data = parse_depth_file(args.input)
    plot_pe(schemas, data, args.depth, args.input, args.output)