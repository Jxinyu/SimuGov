import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Color and texture
COLOR_BLUE = '#4682B4'  # Standard Baseline
COLOR_ORANGE = '#ff7f0e'  # Historical Reproduction (Ref.)
COLOR_BAND = '#E0E0E0'  # Reference interval gray
BAR_WIDTH = 0.6

# ================= 2. Data Preparation =================
scales_u = ['30', '40', '50', '100']
x_pos = np.arange(len(scales_u))

# Define the color and shadow of each bar
# The index of U=40 is 1
colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE]
hatches = ['', '////', '', '']  # Add diagonal line shadow for U=40

metrics_data = {
    'Power-law Index (α)': {
        'data': [1.32, 1.37, 1.39, 1.42],
        'ref': [1.20, 2.50],
        'ylim': [0, 3.2]
    },
    'Goodness-of-fit (R²)': {
        'data': [0.79, 0.71, 0.78, 0.86],
        'ref': [0.70, 1.00],
        'ylim': [0, 1.2]
    },
    'Clustering (C)': {
        'data': [0.54, 0.47, 0.51, 0.39],
        'ref': [0.30, 0.70],
        'ylim': [0, 0.9]
    },
    'Homophily (r)': {
        'data': [0.71, 0.75, 0.80, 0.60],
        'ref': [0.40, 0.90],
        'ylim': [0, 1.1]
    },
    'Gini Index (G)': {
        'data': [0.50, 0.40, 0.58, 0.55],
        'ref': [0.30, 0.70],
        'ylim': [0, 0.9]
    }
}

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']


# ================= 3. Plotting Core Logic =================
def plot_robustness_comparison(output_dir):
    # Create 1x5 facet layout
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8), dpi=300)

    for i, (key, info) in enumerate(metrics_data.items()):
        ax = axes[i]

        # A. Draw background reference band
        ax.axhspan(info['ref'][0], info['ref'][1], color=COLOR_BAND,
                   alpha=0.6, label='Ref. Range', zorder=0)

        # B. Draw bar chart (set color and shadow one by one)
        for j in range(len(scales_u)):
            ax.bar(x_pos[j], info['data'][j], width=BAR_WIDTH,
                   color=colors[j], hatch=hatches[j],
                   edgecolor='black', linewidth=0.6, zorder=3)

        # C. Style settings
        ax.set_title(f"{subplot_labels[i]} {key}", fontweight='bold', pad=12, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scales_u)
        ax.set_xlabel('Scale (U)')

        if i == 0:
            ax.set_ylabel('Metric Value', fontweight='bold')

        ax.set_ylim(info['ylim'])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Numerical annotation (simplified font size)
        for j, val in enumerate(info['data']):
            ax.text(x_pos[j], val + 0.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # ================= 4. Custom Legend (as required) =================
    legend_elements = [
        Patch(facecolor=COLOR_BLUE, edgecolor='black', label='Standard Baseline Policy'),
        Patch(facecolor=COLOR_ORANGE, edgecolor='black', hatch='////', label='Historical Reproduction Policy (Ref.)'),
        Patch(facecolor=COLOR_BAND, edgecolor='none', label='Empirical Reference Range')
    ]

    # Place the legend at the center of the top of the canvas
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, fontsize=9.5)

    # Adjust layout, leave space for the legend at the top
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ================= 5. Save =================
    timestamp = datetime.now().strftime("%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"KDD_Robustness_Comparison_{timestamp}.pdf")
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.savefig(file_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=600)

    print(f"✅ Success: Robustness plot generated at {file_path}")


if __name__ == "__main__":
    # Replace with your actual output path
    OUT_DIR = "./output"
    plot_robustness_comparison(OUT_DIR)