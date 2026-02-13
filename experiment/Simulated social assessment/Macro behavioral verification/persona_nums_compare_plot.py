import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.patches import Patch

# ================= 1. KDD 绘图标准设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 配色与纹理
COLOR_BLUE = '#4682B4'  # Standard Baseline
COLOR_ORANGE = '#ff7f0e'  # Historical Reproduction (Ref.)
COLOR_BAND = '#E0E0E0'  # 参考区间灰色
BAR_WIDTH = 0.6

# ================= 2. 数据准备 =================
scales_u = ['30', '40', '50', '100']
x_pos = np.arange(len(scales_u))

# 定义每根柱子的颜色和阴影
# U=40 的索引是 1
colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE]
hatches = ['', '////', '', '']  # 为 U=40 增加斜线阴影

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


# ================= 3. 绘图核心逻辑 =================
def plot_robustness_comparison(output_dir):
    # 创建 1x5 分面布局
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8), dpi=300)

    for i, (key, info) in enumerate(metrics_data.items()):
        ax = axes[i]

        # A. 绘制背景参考带
        ax.axhspan(info['ref'][0], info['ref'][1], color=COLOR_BAND,
                   alpha=0.6, label='Ref. Range', zorder=0)

        # B. 绘制柱状图 (逐个设置颜色和阴影)
        for j in range(len(scales_u)):
            ax.bar(x_pos[j], info['data'][j], width=BAR_WIDTH,
                   color=colors[j], hatch=hatches[j],
                   edgecolor='black', linewidth=0.6, zorder=3)

        # C. 样式设置
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

        # 数值标注 (精简字号)
        for j, val in enumerate(info['data']):
            ax.text(x_pos[j], val + 0.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # ================= 4. 自定义图例 (按要求) =================
    legend_elements = [
        Patch(facecolor=COLOR_BLUE, edgecolor='black', label='Standard Baseline Policy'),
        Patch(facecolor=COLOR_ORANGE, edgecolor='black', hatch='////', label='Historical Reproduction Policy (Ref.)'),
        Patch(facecolor=COLOR_BAND, edgecolor='none', label='Empirical Reference Range')
    ]

    # 将图例放置在画布顶部的中央
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, fontsize=9.5)

    # 调整布局，为顶部的图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ================= 5. 保存 =================
    timestamp = datetime.now().strftime("%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"KDD_Robustness_Comparison_{timestamp}.pdf")
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.savefig(file_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=600)

    print(f"✅ Success: Robustness plot generated at {file_path}")


if __name__ == "__main__":
    # 替换为您的实际输出路径
    OUT_DIR = "./output"
    plot_robustness_comparison(OUT_DIR)