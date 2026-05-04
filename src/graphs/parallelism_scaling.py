# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data = """
Here is my tree parallel algorithm:

=== Depth 1 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            278.77ms ± 173.45ms         235.74ms ± 107.70ms     1.18x
mid              283.93ms ± 152.05ms         228.28ms ±  42.59ms     1.24x
late             243.17ms ± 117.65ms         209.00ms ±  34.95ms     1.16x

=== Depth 2 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            294.52ms ± 186.05ms         221.73ms ±  43.15ms     1.33x
mid              299.07ms ± 162.88ms         225.73ms ±  41.34ms     1.32x
late             253.50ms ± 125.14ms         209.56ms ±  35.00ms     1.21x

=== Depth 3 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            309.46ms ± 197.09ms         221.78ms ±  43.41ms     1.40x
mid              313.06ms ± 171.48ms         226.58ms ±  41.39ms     1.38x
late             264.67ms ± 131.55ms         209.78ms ±  35.27ms     1.26x

=== Depth 4 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            326.24ms ± 209.37ms         220.73ms ±  43.73ms     1.48x
mid              328.01ms ± 181.80ms         226.89ms ±  41.54ms     1.45x
late             274.21ms ± 137.16ms         210.02ms ±  35.38ms     1.31x

=== Depth 5 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            341.48ms ± 221.06ms         220.40ms ±  43.60ms     1.55x
mid              342.74ms ± 190.60ms         226.75ms ±  41.95ms     1.51x
late             284.86ms ± 143.65ms         208.94ms ±  34.82ms     1.36x

Here is my root parallel algorithm:

=== Depth 1 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            265.88ms ± 166.05ms          46.16ms ± 108.51ms     5.76x
mid              270.10ms ± 145.40ms          32.18ms ±   3.33ms     8.39x
late             228.86ms ± 111.84ms          35.01ms ±   6.78ms     6.54x

=== Depth 2 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            282.30ms ± 179.42ms          33.02ms ±   3.50ms     8.55x
mid              286.08ms ± 156.29ms          35.79ms ±   3.81ms     7.99x
late             239.74ms ± 119.43ms          40.69ms ±  13.09ms     5.89x

=== Depth 3 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            297.36ms ± 190.93ms          32.67ms ±   3.24ms     9.10x
mid              300.82ms ± 166.18ms          35.56ms ±   3.18ms     8.46x
late             250.71ms ± 126.14ms          38.64ms ±  10.99ms     6.49x

=== Depth 4 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            312.93ms ± 202.54ms          32.73ms ±   2.87ms     9.56x
mid              315.84ms ± 175.82ms          35.88ms ±   3.49ms     8.80x
late             261.06ms ± 132.57ms          37.31ms ±   6.72ms     7.00x

=== Depth 5 ===
Phase          Single (1T) avg ± std      Multi (256T) avg ± std   Speedup
--------------------------------------------------------------------------
early            329.12ms ± 214.30ms          33.12ms ±   2.85ms     9.94x
mid              330.79ms ± 185.39ms          36.33ms ±   3.46ms     9.10x
late             272.06ms ± 139.42ms          40.65ms ±   6.41ms     6.69x
"""

# 2. Parse the text into a structured list of dictionaries
records = []
current_algo = None
current_depth = None

for line in data.split('\n'):
    line = line.strip()
    # Identify the algorithm type
    if 'tree parallel' in line.lower():
        current_algo = 'Tree Parallel'
    elif 'root parallel' in line.lower():
        current_algo = 'Root Parallel'
    # Extract the depth using regex
    elif '=== Depth' in line:
        current_depth = int(re.search(r'Depth (\d+)', line).group(1))
    # Extract the timing data for early, mid, late phases
    elif line.startswith(('early', 'mid', 'late')):
        parts = line.split()
        phase = parts[0]
        # Clean the strings to extract just the float values
        single_time = float(parts[1].replace('ms', ''))
        multi_time = float(parts[4].replace('ms', ''))
        
        # Add Single-thread record
        records.append({
            'Algorithm': current_algo,
            'Depth': current_depth,
            'Phase': phase,
            'Thread Count': '1T',
            'Time (ms)': single_time
        })
        
        # Add Multi-thread record
        records.append({
            'Algorithm': current_algo,
            'Depth': current_depth,
            'Phase': phase,
            'Thread Count': '256T',
            'Time (ms)': multi_time
        })

# 3. Create the DataFrame
df = pd.DataFrame(records)

# 4. Plotting setup
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

df_tree = df[df['Algorithm'] == 'Tree Parallel']
df_root = df[df['Algorithm'] == 'Root Parallel']

# Tree Parallel Subplot
sns.lineplot(
    data=df_tree, x='Depth', y='Time (ms)', 
    hue='Thread Count', style='Phase', 
    markers=True, dashes=True, ax=axes[0]
)
axes[0].set_title('Tree Parallel: Execution Time vs Depth')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_ylabel('Execution Time (ms)')

# Root Parallel Subplot
sns.lineplot(
    data=df_root, x='Depth', y='Time (ms)', 
    hue='Thread Count', style='Phase', 
    markers=True, dashes=True, ax=axes[1]
)
axes[1].set_title('Root Parallel: Execution Time vs Depth')
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].set_ylabel('Execution Time (ms)')

# Final rendering adjustments
plt.tight_layout()
plt.savefig('parallelism_scaling_MCTS.png')
plt.show()