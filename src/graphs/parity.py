import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the image
algorithms = ['RP-100K', 'RP-1M', 'TP-100K']
wins = np.array([1, 10, 4])
ties = np.array([5, 5, 6])
losses = np.array([14, 5, 10])

# Calculate percentages for the Y-axis
total = wins + ties + losses
wins_pct = (wins / total) * 100
ties_pct = (ties / total) * 100
losses_pct = (losses / total) * 100

# Plotting - Setting a wider figsize for landscape orientation
fig, ax = plt.subplots(figsize=(10, 4)) 

width = 0.6  # Bar width

# Create stacked bars with specific colors
# Colors: Green (#4CAF50), Yellow (#FFEB3B), Red (#F44336)
p1 = ax.bar(algorithms, wins_pct, width, color='#4CAF50', label='Wins', edgecolor='black', linewidth=0.8)
p2 = ax.bar(algorithms, ties_pct, width, bottom=wins_pct, color='#FFEB3B', label='Ties', edgecolor='black', linewidth=0.8)
p3 = ax.bar(algorithms, losses_pct, width, bottom=wins_pct + ties_pct, color='#F44336', label='Losses', edgecolor='black', linewidth=0.8)

# Function to add the raw count labels inside the bar segments
def add_labels(rects, counts, offsets):
    for rect, count, offset in zip(rects, counts, offsets):
        height = rect.get_height()
        y_pos = offset + height / 2
        ax.text(rect.get_x() + rect.get_width() / 2., y_pos,
                f'{count}', ha='center', va='center', fontsize=12, color='black')

add_labels(p1, wins, np.zeros(3))
add_labels(p2, ties, wins_pct)
add_labels(p3, losses, wins_pct + ties_pct)

# Formatting axes and labels
ax.set_ylabel('Percentage of Games (%)', fontsize=12)
ax.set_xlabel('Algorithm', fontsize=12)
ax.set_title('MCTS Parity Figure (Depth 6 Search)', fontsize=14, pad=20)
ax.set_ylim(0, 100)

# Legend - Reversed to match the visual stack order (Losses on top)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), title='Outcome', loc='upper left', bbox_to_anchor=(1.01, 1))

plt.tight_layout()
plt.savefig('landscape_mcts_plot.png', dpi=300)
plt.show()