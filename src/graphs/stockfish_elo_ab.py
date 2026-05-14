import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import DATA_DIR

ELOS            = [1350, 1650, 1950, 2250, 2550, 2850]
COMPETITIVE_DIR = os.path.join(DATA_DIR, "outputs", "competitive")

# (label, filename, linestyle, marker)
CONFIGS = [
    ("Depth 3",     "cpp-alpha-beta-depth3-elo.txt",    "-",  "o"),
    ("Depth 4",     "cpp-alpha-beta-depth4-elo.txt",    "-",  "s"),
    ("Depth 5",     "cpp-alpha-beta-depth5-elo.txt",    "-",  "^"),
    ("Time 1ms",    "cpp-alpha-beta-time1-elo.txt",     "--", "o"),
    ("Time 10ms",   "cpp-alpha-beta-time10-elo.txt",    "--", "s"),
    ("Time 100ms",  "cpp-alpha-beta-time100-elo.txt",   "--", "^"),
    ("Cycles 1M",   "cpp-alpha-beta-cycles1-elo.txt",   ":",  "o"),
    ("Cycles 10M",  "cpp-alpha-beta-cycles10-elo.txt",  ":",  "s"),
    ("Cycles 100M", "cpp-alpha-beta-cycles100-elo.txt", ":",  "^"),
]


def parse_elo_file(path):
    """Parse a competitive output file. Returns dict: elo -> points_out_of_100."""
    results = {}
    current = {}

    with open(path) as f:
        for raw in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()

            m = re.search(r"Opponent\s*:\s*Stockfish\s+(\d+)\s+ELO", line)
            if m:
                current["elo"] = int(m.group(1))
                continue
            m = re.search(r"Games\s*:\s*(\d+)", line)
            if m:
                current["n"] = int(m.group(1))
                continue
            m = re.search(r"Wins\s*:\s*(\d+)", line)
            if m:
                current["wins"] = int(m.group(1))
                continue
            m = re.search(r"Losses\s*:\s*(\d+)", line)
            if m:
                current["losses"] = int(m.group(1))
                continue
            m = re.search(r"Draws\s*:\s*(\d+)", line)
            if m:
                current["draws"] = int(m.group(1))
                continue
            if "Win %" in line and all(k in current for k in ("elo", "n", "wins", "losses", "draws")):
                pts = (current["wins"] + 0.5 * current["draws"]) / current["n"] * 100
                results[current["elo"]] = pts
                current = {}

    return results


def plot_elo(output_dir):
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(CONFIGS)))

    fig, ax = plt.subplots(figsize=(11, 6))

    for (label, fname, ls, marker), color in zip(CONFIGS, colors):
        path = os.path.join(COMPETITIVE_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {fname} not found, skipping")
            continue
        data = parse_elo_file(path)
        pts = [data.get(elo, np.nan) for elo in ELOS]
        ax.plot(ELOS, pts, linestyle=ls, marker=marker, color=color,
                label=label, linewidth=1.8, markersize=6)

    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
               label="50 pts (Break-Even)")

    ax.set_xlabel("Stockfish 11 ELO", fontsize=12)
    ax.set_ylabel("Points (Out Of 100 games)", fontsize=12)
    ax.set_title("Alpha-Beta Vs Stockfish — Points By ELO\n"
                 "(Win=1, Draw=0.5, Loss=0, Scaled To 100 Games)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(ELOS)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "stockfish-elo-ab.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "graphs"),
                        help="Directory to save the graph (default: data/graphs/)")
    args = parser.parse_args()
    plot_elo(args.output)
