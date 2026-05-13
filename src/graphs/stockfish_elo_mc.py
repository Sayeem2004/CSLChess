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
    ("Time 1ms",    "cpp-monte-carlo-time1-elo.txt",    "--", "o"),
    ("Time 10ms",   "cpp-monte-carlo-time10-elo.txt",   "--", "s"),
    ("Time 100ms",  "cpp-monte-carlo-time100-elo.txt",  "--", "^"),
    ("Cycles 1M",   "cpp-monte-carlo-cycles1-elo.txt",  ":",  "o"),
    ("Cycles 10M",  "cpp-monte-carlo-cycles10-elo.txt", ":",  "s"),
    ("Cycles 100M", "cpp-monte-carlo-cycles100-elo.txt",":",  "^"),
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
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(CONFIGS)))

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
               label="50 pts (break-even)")

    ax.set_xlabel("Stockfish ELO", fontsize=12)
    ax.set_ylabel("Points (out of 100 games)", fontsize=12)
    ax.set_title("MCTS vs Stockfish — Points by ELO\n"
                 "(win=1, draw=0.5, loss=0, scaled to 100 games)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(ELOS)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "stockfish-elo-mc.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Graph saved as '{out_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "graphs"),
                        help="Directory to save the graph (default: data/graphs/)")
    args = parser.parse_args()
    plot_elo(args.output)