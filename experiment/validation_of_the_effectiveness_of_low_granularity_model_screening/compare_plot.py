import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator


data_ranks = []

# Group 1 to 5 Rankings (HF list vs LF list)
groups_data = [
    {
        "hf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20"],
        "lf": ["edu_高_ai_0.90_f_0.50", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.90_f_0.10", "edu_中_ai_0.70_f_0.80", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_中_ai_0.50_f_0.50", "edu_低_ai_0.30_f_0.20", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90"]
    },
    {
        "hf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20"],
        "lf": ["edu_高_ai_0.90_f_1.00", "edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.10_f_0.90", "edu_中_ai_0.50_f_0.50", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.30_f_0.20"]
    },
    {
        "hf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20"],
        "lf": ["edu_中_ai_0.70_f_0.80", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_0.50", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.50_f_0.50", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20", "edu_低_ai_0.10_f_0.90", "edu_中_ai_0.20_f_0.40"]
    },
    {
        "hf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20"],
        "lf": ["edu_高_ai_0.90_f_1.00", "edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_中_ai_0.70_f_0.80", "edu_低_ai_0.60_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_中_ai_0.50_f_0.50", "edu_低_ai_0.30_f_0.20", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.10_f_0.90"]
    },
    {
        "hf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.10_f_0.90", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_低_ai_0.30_f_0.20"],
        "lf": ["edu_高_ai_0.90_f_0.50", "edu_中_ai_0.90_f_0.10", "edu_高_ai_0.90_f_1.00", "edu_中_ai_0.70_f_0.80", "edu_低_ai_0.10_f_0.00", "edu_低_ai_0.60_f_0.50", "edu_中_ai_0.50_f_0.50", "edu_中_ai_0.20_f_0.40", "edu_低_ai_0.30_f_0.20", "edu_低_ai_0.10_f_0.90"]
    }
]

scatter_points = []
for group in groups_data:
    hf_list = group['hf']
    lf_list = group['lf']
    for rank_idx, policy in enumerate(hf_list):
        hf_rank = rank_idx + 1
        lf_rank = lf_list.index(policy) + 1
        scatter_points.append({'HF Rank': hf_rank, 'LF Rank': lf_rank})

df_scatter = pd.DataFrame(scatter_points)

recall_data = {
    'Threshold': ['Top 20%', 'Top 30%', 'Top 40%'],
    'Recall Mean': [0.60, 0.933, 1.0],
    'Recall Std': [0.20, 0.133, 0.0]  # Standard Deviation
}
df_recall = pd.DataFrame(recall_data)

sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

color_scatter = "#2c7bb6" # 蓝色
color_line = "#d7191c"    # 红色
color_band = "#fdae61"    # 橙色 (用于误差带)

ax1 = axes[0]

jitter_x = np.random.normal(0, 0.12, size=len(df_scatter))
jitter_y = np.random.normal(0, 0.12, size=len(df_scatter))

ax1.scatter(df_scatter['HF Rank'] + jitter_x,
            df_scatter['LF Rank'] + jitter_y,
            color=color_scatter, alpha=0.6, s=80, edgecolors='w', linewidth=0.5, zorder=3)

ax1.plot([0, 11], [0, 11], ls="--", c="gray", alpha=0.6, lw=1.5, zorder=2, label="Perfect Match")

ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(0.5, 10.5)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel("High-Fidelity Rank (Ground Truth)", fontweight='bold')
ax1.set_ylabel("RSC Low-Fidelity Rank (Proxy)", fontweight='bold')
ax1.set_title("(a) Rank Consistency Analysis", loc='left', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle=':', alpha=0.4)

spearman_val = 0.7648  # From JSON
ax1.text(0.05, 0.95, f"Avg. Spearman $\\rho = {spearman_val:.4f}$",
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))

ax1.add_patch(plt.Rectangle((0.5, 0.5), 3.5, 3.5, color='green', alpha=0.1, zorder=1))
ax1.text(2.25, 3.5, "Elite Zone", color='green', fontsize=10, ha='center', va='bottom', fontweight='bold')


ax2 = axes[1]
x_vals = np.arange(3) # 0, 1, 2 for categorical plotting

ax2.fill_between(x_vals,
                 df_recall['Recall Mean'] - df_recall['Recall Std'],
                 df_recall['Recall Mean'] + df_recall['Recall Std'],
                 color=color_line, alpha=0.15, label='Std. Dev (5 Runs)')

ax2.plot(x_vals, df_recall['Recall Mean'], marker='o', markersize=10,
         color=color_line, lw=2.5, label='Mean Elite Recall')

key_idx = 1
ax2.annotate('Sweet Spot:\n93.3% Recall',
             xy=(key_idx, df_recall['Recall Mean'][key_idx]),
             xytext=(key_idx, df_recall['Recall Mean'][key_idx] - 0.2),
             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
             ha='center', fontsize=11, fontweight='bold', color='#333')

ax2.set_xticks(x_vals)
ax2.set_xticklabels(df_recall['Threshold'])
ax2.set_ylim(0.0, 1.1)
ax2.set_xlabel("Screening Threshold (Top N%)", fontweight='bold')
ax2.set_ylabel("Elite Strategy Recall Rate", fontweight='bold')
ax2.set_title("(b) Screening Efficiency & Recall", loc='left', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

ax2.text(2, 1.02, "100%", ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_line)

plt.tight_layout(pad=3.0)
plt.show()