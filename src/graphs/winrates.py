import matplotlib.pyplot as plt

columns = ['Time (1ms)', 'Time (10ms)', '1 MCycle', '10 MCycle', '100 MCycle']
rows = ['Leaf Parallel', 'Root Parallel', 'Alpha-Beta']
data = [
    [ (1,3,96), (1,5,94), (0,0,100), (0,3,97), (6,29,65) ],
    [ (12,7,81), (24,11,65), (0,0,100), (0,0,100), (0,0,100) ],
    [ ('-','-','-'), ('-','-','-'), ('-','-','-'), ('-','-','-'), ('-','-','-') ]
]

fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 6)
ax.set_ylim(0, 4)
ax.axis('off')

# Draw grid lines
for x in range(7):
    ax.plot([x, x], [0, 4], color='black', lw=1)
for y in range(5):
    ax.plot([0, 6], [y, y], color='black', lw=1)

# Add column headers
ax.text(0.5, 3.5, 'Algorithm', ha='center', va='center', weight='bold', fontsize=12)
for i, col in enumerate(columns):
    ax.text(i + 1.5, 3.5, col, ha='center', va='center', weight='bold', fontsize=12)

# Add row headers
for i, row in enumerate(rows):
    ax.text(0.5, 2.5 - i, row, ha='center', va='center', weight='bold', fontsize=12)

# Add data cells
colors = ['green', '#D8A000', 'red']
for i, row_data in enumerate(data):
    for j, cell in enumerate(row_data):
        x_center = j + 1.5
        y_center = 2.5 - i
        
        w, t, l = cell
        
        # Position the W | T | L numbers manually within the cell
        ax.text(x_center - 0.25, y_center, str(w), color=colors[0], ha='center', va='center', weight='bold', fontsize=12)
        ax.text(x_center - 0.12, y_center, '|', color='black', ha='center', va='center', fontsize=12)
        ax.text(x_center, y_center, str(t), color=colors[1], ha='center', va='center', weight='bold', fontsize=12)
        ax.text(x_center + 0.12, y_center, '|', color='black', ha='center', va='center', fontsize=12)
        ax.text(x_center + 0.25, y_center, str(l), color=colors[2], ha='center', va='center', weight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('winrates_table.png')
plt.show()