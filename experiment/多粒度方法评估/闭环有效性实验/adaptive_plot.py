import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, ArrowStyle

# --------------------------
# 1. 数据录入
# --------------------------

# Group A: 顺从环境 - 最优解 (Low Reactance)
data_a = {
    "safety": [0.74, 0.83, 0.85, 0.78, 0.8, 0.76, 0.76, 0.78, 0.75, 0.73, 0.7, 0.77, 0.82],
    "theta": [0.9, 0.9, 0.9, 0.9, 0.897, 0.895, 0.885, 0.881, 0.877, 0.868, 0.868, 0.863, 0.856, 0.852, 0.848],
    "jitter": 0.0037
}

# Group B: 激进环境 - 最优解 (High Reactance - Adapted)
data_b = {
    "safety": [0.64, 0.6, 0.54, 0.54, 0.53, 0.62, 0.7, 0.8, 0.81, 0.82, 0.84, 0.87, 0.9],
    "theta": [0.69, 0.69, 0.69, 0.725, 0.763, 0.746, 0.711, 0.676, 0.651, 0.637, 0.638, 0.643, 0.65, 0.656, 0.655],
    "jitter": 0.0156
}

# Group C: 激进环境 - 策略错配 (High Reactance - Mismatch)
data_c = {
    "safety": [0.6, 0.53, 0.35, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "satisfaction": [0.15, 0.09, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "theta": [0.9, 0.9, 0.9, 0.927, 0.933, 0.922, 0.913, 0.908, 0.904, 0.895, 0.885, 0.883, 0.875, 0.87, 0.859],
    "jitter": 0.0076
}

# 统一时间轴 (取前13个点用于KPI对比，Theta用前15个点)
time_kpi = list(range(13))
time_theta = list(range(15))

# --------------------------
# 2. 绘图设置
# --------------------------
sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# 颜色定义
color_a = "#2c7bb6" # 蓝色 - 顺从/稳定
color_b = "#d7191c" # 红色 - 激进/适应
color_c = "#fdae61" # 橙色/灰色 - 错配/崩塌
color_gray = "#7f7f7f"

# --------------------------
# 左图 (a): 系统崩塌 (System Collapse)
# 对比 Group B (适应) vs Group C (错配) 的 Safety 指标
# --------------------------
ax1 = axes[0]

# 绘制 Group B Safety (成功适应)
ax1.plot(time_kpi, data_b['safety'], color=color_b, lw=3, label='Group B: Adapted Strategy', marker='o', markersize=6)

# 绘制 Group C Safety (崩塌)
ax1.plot(time_kpi, data_c['safety'], color=color_gray, lw=3, ls='--', label='Group C: Mismatched Strategy', marker='x', markersize=7)

# 绘制 Group C Satisfaction (辅助说明崩塌)
ax1.plot(time_kpi, data_c['satisfaction'], color=color_c, lw=2, ls=':', alpha=0.7, label='Group C: Satisfaction')

# 添加崩塌区域阴影
ax1.fill_between(time_kpi, 0, data_c['safety'], where=[x < 5 for x in time_kpi], color=color_gray, alpha=0.1)
ax1.axvspan(3, 13, color="#ffebee", alpha=0.3) # 红色背景表示危险区
ax1.text(7, 0.3, "System Collapse\n(Metrics $\\to$ 0)", color=color_b, fontsize=12, fontweight='bold', ha='center', va='center')

# 标注
ax1.annotate('Cliff Effect', xy=(3, 0.1), xytext=(5, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold')

ax1.set_title("(a) Impact of Strategy Mismatch (Collapse)", loc='left', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Simulation Days (t)", fontweight='bold')
ax1.set_ylabel("Normalized KPI Value", fontweight='bold')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)


# --------------------------
# 右图 (b): 治理代价不对称性 (Theta Dynamics)
# 对比 Group A (顺从) vs Group B (激进) 的 Theta 波动
# --------------------------
ax2 = axes[1]

# 绘制 Group A Theta (平滑)
ax2.plot(time_theta, data_a['theta'], color=color_a, lw=2.5, label='Group A: Low Reactance Env.', marker='s', markersize=5)

# 绘制 Group B Theta (波动)
ax2.plot(time_theta, data_b['theta'], color=color_b, lw=2.5, label='Group B: High Reactance Env.', marker='^', markersize=6)

# 计算并标注 Jitter 差异
# 使用误差棒或者标注来展示 Jitter (Sigma)
ax2.text(8, 0.92, f"Group A Stability:\nLow Jitter ($\\sigma={data_a['jitter']:.4f}$)",
         color=color_a, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=color_a, alpha=0.8))

ax2.text(8, 0.77, f"Group B Volatility:\nHigh Jitter ($\\sigma={data_b['jitter']:.4f}$)",
         color=color_b, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=color_b, alpha=0.8))

# 添加双箭头表示波动幅度
ax2.annotate('', xy=(5, 0.763), xytext=(5, 0.637), arrowprops=dict(arrowstyle='<->', color=color_b, lw=1.5))
ax2.text(5.2, 0.7, "Dynamic\nAdjustment", color=color_b, fontsize=9, va='center')

ax2.set_title("(b) Asymmetric Governance Cost ($\\theta$ Dynamics)", loc='left', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel("Simulation Days (t)", fontweight='bold')
ax2.set_ylabel("Audit Threshold ($\\theta_t$)", fontweight='bold')
ax2.set_ylim(0.5, 1.0)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)

plt.tight_layout(pad=3.0)
plt.show()