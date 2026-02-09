import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================= KDD 绘图标准设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 9  # 刻度统一设为 9pt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 配色方案
COLOR_OURS = '#4682B4'  # SimuGov (Blue)
COLOR_BASE_1 = '#E67E22'  # Orange (Laissez-faire)
COLOR_BASE_2 = '#2ca02c'  # Green (Linear)
COLOR_BASE_3 = '#d62728'  # Red (Strict)
COLOR_JITTER = '#E67E22'  # Green (专门用于抖动缩减)

# 数据准备
radar_labels = ['Safety', 'Creativity', 'Satisfaction']
radar_data = {
    'SimuGov': [0.92, 0.88, 0.94],
    'Base1 (Laissez-faire)': [0.60, 0.82, 0.85],
    'Base2 (Linear-compromise)': [0.72, 0.45, 0.70],
    'Base3 (Extremely-strict)': [0.90, 0.25, 0.30]
}
metrics_labels = ['HV Expansion', 'Dominance\n(C-Metric)', 'Global\nStability', 'Jitter\nReduction']
metrics_values = [30.2, 66.7, 26.5, 52.4]
metrics_colors = [COLOR_OURS, COLOR_OURS, COLOR_OURS, COLOR_JITTER]


def plot_kdd_pixel_perfect(output_dir):
    # 创建画布
    fig = plt.figure(figsize=(10, 4.5), dpi=300)

    # ---------------------------------------------------------
    # 子图 (a): 雷达图单元 (包含图例)
    # ---------------------------------------------------------
    # [left, bottom, width, height]
    ax1 = fig.add_axes([0.08, 0.22, 0.4, 0.52], projection='polar')
    ax1.set_theta_zero_location('N')

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    # 绘制圆形网格
    ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], color='gray', alpha=0.3)
    # 强制设置雷达图内数字大小，与右图对齐
    ax1.yaxis.set_tick_params(labelsize=9, labelcolor='#666666')
    ax1.set_rmax(1.1)

    # 绘制曲线
    ax1.plot(angles, radar_data['Base1 (Laissez-faire)'] + [radar_data['Base1 (Laissez-faire)'][0]],
             color=COLOR_BASE_1, ls='--', lw=1.8, label='Base1 (Laissez-faire)')
    ax1.plot(angles, radar_data['Base2 (Linear-compromise)'] + [radar_data['Base2 (Linear-compromise)'][0]],
             color=COLOR_BASE_2, ls='-.', lw=1.8, label='Base2 (Linear-compromise)')
    ax1.plot(angles, radar_data['Base3 (Extremely-strict)'] + [radar_data['Base3 (Extremely-strict)'][0]],
             color=COLOR_BASE_3, ls=':', lw=1.8, label='Base3 (Extremely-strict)')
    ax1.plot(angles, radar_data['SimuGov'] + [radar_data['SimuGov'][0]],
             color=COLOR_OURS, ls='-', lw=2.5, marker='o', markersize=5, label='SimuGov')
    ax1.fill(angles, radar_data['SimuGov'] + [radar_data['SimuGov'][0]], facecolor=COLOR_OURS, alpha=0.15)

    # 【手工绘制顶点标签】确保完美对称和距离控制
    ax1.set_xticklabels([])  # 隐藏默认标签

    # Safety: 下移一些，距离圆心 1.18 处 (之前可能在 1.25)
    ax1.text(0, 1.0, 'Safety', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Creativity: 240度 (左下)，设置 ha='right'
    ax1.text(angles[1], 1.0, 'Creativity ', ha='right', va='top', fontweight='bold', fontsize=11)

    # Satisfaction: 120度 (右下)，设置 ha='left'
    ax1.text(angles[2], 1.0, ' Satisfaction', ha='left', va='top', fontweight='bold', fontsize=11)

    # 图例：2x2 布局
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, fontsize=9.5, columnspacing=1.2, handlelength=2.5)

    # 底部对齐标题 (a)
    ax1.text(0.5, -0.28, '(a) Multi-dimensional Strategy Radar', transform=ax1.transAxes,
             ha='center', va='top', fontweight='bold', fontsize=12)

    # ---------------------------------------------------------
    # 子图 (b): 柱状图单元
    # ---------------------------------------------------------
    ax2 = fig.add_axes([0.58, 0.22, 0.38, 0.72])

    x_pos = np.arange(len(metrics_labels))
    bars = ax2.bar(x_pos, metrics_values, color=metrics_colors, width=0.55, edgecolor='black', linewidth=0.8, zorder=3)

    # 柱顶标注
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 1.2, f'+{h:.1f}%',
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_labels, fontsize=10)
    ax2.set_ylabel('Improvement Rate (%)', fontsize=11)
    ax2.set_ylim(0, 80)
    # 确保 Y 轴刻度字号为 9pt，与左图一致
    ax2.tick_params(axis='y', labelsize=9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 底部对齐标题 (b)
    ax2.text(0.5, -0.20, '(b) Optimization Performance Gains', transform=ax2.transAxes,
             ha='center', va='top', fontweight='bold', fontsize=12)

    # ================= 保存 =================
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%H%M")
    save_path = os.path.join(output_dir, f"KDD_Pixel_Perfect_{timestamp}.pdf")

    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), format='png', dpi=600, bbox_inches='tight')
    print(f"✅ 像素级对齐版已生成: {save_path}")


if __name__ == "__main__":
    out = r"experiment\多粒度方法评估\闭环有效性实验\output"
    plot_kdd_pixel_perfect(out)