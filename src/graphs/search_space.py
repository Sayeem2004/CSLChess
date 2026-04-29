import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmark import PHASES, DATA_DIR
from utils.load import load_perft


def plot_chess_log2_bars(phase_data_dict):
    """
    Creates a grouped bar chart with log2 scale and a branching factor table.
    phase_data_dict: Dict of {'Phase Name': 'path/to/csv'}
    """
    processed_dfs = {}
    for phase, path in phase_data_dict.items():
        df = pd.read_csv(path)
        df['depth_num'] = df['depth'].str.extract('(\d+)').astype(int)
        df['sum'] = pd.to_numeric(df['sum'])
        df['branching_factor'] = pd.to_numeric(df['branching_factor'], errors='coerce')
        df["max_branching_factor"] = df["branching_factor"].sum() / (len(df) - 1)
        processed_dfs[phase] = df


    # Plot setup
    depths = processed_dfs['Middle']['depth_num'].values
    x = np.arange(len(depths))
    width = 0.25
    colors = {'Early': '#2ecc71', 'Middle': '#3498db', 'Late': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bars
    ax.bar(x - width, processed_dfs['Early']['sum'], width,
           label=f'Early ({processed_dfs["Early"]["max_branching_factor"].iloc[0]:.2f}x BF)', color=colors['Early'])
    ax.bar(x, processed_dfs['Middle']['sum'], width,
           label=f'Middle ({processed_dfs["Middle"]["max_branching_factor"].iloc[0]:.2f}x BF)', color=colors['Middle'])
    ax.bar(x + width, processed_dfs['Late']['sum'], width,
           label=f'Late ({processed_dfs["Late"]["max_branching_factor"].iloc[0]:.2f}x BF)', color=colors['Late'])
    ax.set_yscale('log', base=10)

    # Axes formatting
    ax.set_ylabel('Total Board Positions', fontsize=12)
    ax.set_xlabel('Search Depth', fontsize=12)
    ax.set_title('Exponential Growth of Chess Search Space', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in depths])
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    outpath = os.path.join(DATA_DIR, "graphs", "search_space.png")
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    print("Graph saved as 'search_space.png'")


if __name__ == "__main__":
    paths = {
        'Early': os.path.join(DATA_DIR, "early", "node_counts_summary.csv"),
        'Middle': os.path.join(DATA_DIR, "mid", "node_counts_summary.csv"),
        'Late': os.path.join(DATA_DIR, "late", "node_counts_summary.csv")
    }
    plot_chess_log2_bars(paths)
