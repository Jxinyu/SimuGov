import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ================= KDD 绘图标准设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['hatch.linewidth'] = 0.5

# 配色方案 (Blue vs Orange)
COLOR_OURS = '#4682B4'  # SimuGov (Blue)
COLOR_BASE = '#ff7f0e'  # Baseline (Orange)
COLOR_ERR = 'black'  # 误差线颜色

# ================= 数据准备 =================
data_map = {
    'pearson': {
        'title': 'Pearson Correlation (r)',
        'data': [0.9220, 0.02, 0.7843, 0.07, '*** (p<0.01)'],
        'ylabel': 'Correlation Coefficient',
        # 【修改点】统一上限为 1.15，既对齐了 Spearman，又给显著性标注留出空间
        'ylim': (0.0, 1.15)
    },
    'spearman': {
        'title': 'Spearman Correlation (ρ)',
        'data': [0.8737, 0.03, 0.7950, 0.09, None],
        'ylabel': 'Correlation Coefficient',
        # 【修改点】统一上限为 1.15，实现视觉上的极端统一
        'ylim': (0.0, 1.15)
    },
    'lag': {
        'title': 'Peak Lag (Days)',
        'data': [-1.60, 2.41, -3.00, 2.53, None],
        'ylabel': 'Time Lag (Days)',
        'ylim': (-6.5, 1.0)
    },
    'mechanism': {
        'title': 'Mechanism Correlation (r)',
        'data': [-0.5598, 0.14, -0.5029, 0.08, None],
        'ylabel': 'Correlation Coefficient',
        'ylim': (-0.8, 0.1)
    }
}

subplots_config = [
    ('pearson', '(a)'),
    ('spearman', '(b)'),
    ('lag', '(c)'),
    ('mechanism', '(d)')
]


# ================= 绘图函数 =================

def draw_significance_bar(ax, x1, x2, y, h, text):
    """绘制显著性横线与星号"""
    line_x = [x1, x1, x2, x2]
    line_y = [y, y + h, y + h, y]
    ax.plot(line_x, line_y, lw=1.2, c='k')
    # 稍微调整 text 的垂直位置，防止顶到上边框
    ax.text((x1 + x2) / 2, y + h + 0.01, text, ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_2x2_metrics(output_dir):
    # KDD 标准双栏宽度 (7.5 inch)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), dpi=300)
    axes = axes.flatten()

    bar_width = 0.5
    x_pos = [0, 0.8]
    labels = ['SimuGov\n(Ours)', 'Baseline']

    for idx, (key, label_prefix) in enumerate(subplots_config):
        ax = axes[idx]
        info = data_map[key]

        mean_sim, err_sim, mean_base, err_base, p_val = info['data']
        means = [mean_sim, mean_base]
        errs = [err_sim, err_base]

        # 1. 绘制柱状图
        ax.bar(x_pos[0], means[0], yerr=errs[0], width=bar_width,
               color=COLOR_OURS, label='SimuGov (Ours)',
               capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': COLOR_ERR},
               edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3)

        ax.bar(x_pos[1], means[1], yerr=errs[1], width=bar_width,
               color=COLOR_BASE, label='Baseline',
               capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': COLOR_ERR},
               edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3)

        # 2. 0轴参考线
        ax.axhline(0, color='black', linewidth=0.8, zorder=4)

        # 3. 显著性标注 (自动计算位置)
        if p_val:
            y_max = max(abs(means[0]) + errs[0], abs(means[1]) + errs[1])
            if means[0] > 0:
                # 动态计算横线高度：在柱子顶部上方留出 5% 的空间
                h_line = y_max * 1.05
                h_tick = y_max * 0.02
                draw_significance_bar(ax, x_pos[0], x_pos[1], h_line, h_tick, p_val)

        # 4. 样式设置
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel(info['ylabel'])

        # 应用统一的 Y 轴范围
        if 'ylim' in info:
            ax.set_ylim(info['ylim'])

        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 5. 底部标题 (Bold, KDD Style)
        ax.text(0.5, -0.25, f"{label_prefix} {info['title']}",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold', color='black')

    # 全局图例
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=10)

    # 布局微调
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.45, wspace=0.25)

    # 输出文件
    timestamp = datetime.now().strftime("%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_path = os.path.join(output_dir, f"KDD_Metrics_UnifiedY_{timestamp}.pdf")
    png_path = os.path.join(output_dir, f"KDD_Metrics_UnifiedY_{timestamp}.png")

    plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')

    print(f"✅ 生成完成 (Y轴已统一对齐):\nPDF: {pdf_path}")


if __name__ == "__main__":
    OUTPUT_DIR = r"experiment\仿真社会评估\案例验证\output"
    plot_2x2_metrics(OUTPUT_DIR)
