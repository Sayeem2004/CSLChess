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
    path = os.path.join(DATA_DIR, phase, "node_counts_ab_summary.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    bfs = [float(r["effective_branching_factor"]) for r in rows if r["effective_branching_factor"]]
    return sum(bfs) / len(bfs)


def schema_to_nthreads(schema):
    """'2x32' -> 64, '1x1' -> 1, '64T' -> 64."""
    m = re.match(r"(\d+)x(\d+)", schema)
    if m:
        return int(m.group(1)) * int(m.group(2))
    m = re.match(r"(\d+)T", schema)
    if m:
        return int(m.group(1))
    return 1


def parse_time_file(path):
    """
    Parse calibrate_alpha_beta_time output.
    Returns:
        schemas: list of schema labels in column order
        data: dict budget_ms -> phase -> list of avg_depth per schema (same order)
    """
    schemas     = []
    data        = {}
    current_budget = None
    header_parsed  = False

    budget_re = re.compile(r"---\s*Budget:\s*(\d+)ms\s*---")
    phase_re  = re.compile(r"^(early|mid|late)\s+(.+)$")

    with open(path) as f:
        for raw in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()

            m = budget_re.search(line)
            if m:
                current_budget = int(m.group(1))
                data[current_budget] = {}
                header_parsed = False
                continue

            if current_budget is None:
                continue

            # The header row contains schema labels (first non-separator line after budget marker)
            if not header_parsed and line.startswith("Phase"):
                parts = line.split()
                # parts[0] = "Phase", rest are schemas until "Depth" "Gain"
                raw_schemas = []
                i = 1
                while i < len(parts):
                    if parts[i] in ("Depth", "Gain"):
                        break
                    raw_schemas.append(parts[i])
                    i += 1
                if not schemas:
                    schemas = raw_schemas
                header_parsed = True
                continue

            if not header_parsed or line.startswith("-"):
                continue

            m = phase_re.match(line.strip())
            if m:
                phase = m.group(1)
                nums  = re.findall(r"[-+]?\d+\.\d+", m.group(2))
                # Last number is "Depth Gain"; preceding ones are avg depths per schema
                depths = [float(x) for x in nums[:len(schemas)]]
                data[current_budget][phase] = depths

    return schemas, data


def plot_pe(schemas, data, budget_ms, bf, input_path, output_dir):
    phases = ["early", "mid", "late"]

    d = data.get(budget_ms)
    if d is None:
        print(f"Error: budget {budget_ms}ms not found. Available: {sorted(data)}")
        sys.exit(1)

    n_threads = [schema_to_nthreads(s) for s in schemas]
    n_cols    = len(schemas)

    # Per-schema: avg speedup and avg parallel efficiency across phases
    avg_speedups = []
    avg_pes      = []

    for col_idx in range(n_cols):
        N      = n_threads[col_idx]
        speedups_phases = []
        for phase in phases:
            depths = d.get(phase)
            if depths is None or len(depths) <= col_idx:
                continue
            depth_baseline = d[phase][0]     # first schema = 1T baseline
            depth_N        = depths[col_idx]
            gain           = depth_N - depth_baseline
            speedup        = bf[phase] ** gain
            speedups_phases.append(speedup)

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
    for bar, sp, pe in zip(bars, avg_speedups, avg_pes):
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
        f"Alpha-Beta Parallel Efficiency — {budget_ms}ms Budget\n"
        f"(speedup = BF^depth_gain, avg across early/mid/late)",
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
    out_path = os.path.join(output_dir, f"{out_name}-pe-budget{budget_ms}ms.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs", "calibrate", "alpha-beta-time", "alpha-beta-time.txt"
    )
    default_output = os.path.join(DATA_DIR, "graphs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=default_input,
                        help="Path to calibrate_alpha_beta_time output file")
    parser.add_argument("--budget", type=int, required=True,
                        help="Which budget block to plot in ms (e.g. 100)")
    parser.add_argument("--output", default=default_output,
                        help="Directory to save the graph (default: data/graphs/)")
    args = parser.parse_args()

    schemas, data = parse_time_file(args.input)

    bf = {}
    for phase in ["early", "mid", "late"]:
        try:
            bf[phase] = load_avg_branching_factor(phase)
        except FileNotFoundError:
            print(f"Warning: node_counts_ab_summary.csv not found for {phase}, using bf=1")
            bf[phase] = 1.0

    plot_pe(schemas, data, args.budget, bf, args.input, args.output)