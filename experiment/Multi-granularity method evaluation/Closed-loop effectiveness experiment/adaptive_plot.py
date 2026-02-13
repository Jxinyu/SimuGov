import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, ArrowStyle

# --------------------------
# 1. Data Entry
# --------------------------

# Group A: Compliant Environment - Optimal Solution (Low Reactance)
data_a = {
    "safety": [0.74, 0.83, 0.85, 0.78, 0.8, 0.76, 0.76, 0.78, 0.75, 0.73, 0.7, 0.77, 0.82],
    "theta": [0.9, 0.9, 0.9, 0.9, 0.897, 0.895, 0.885, 0.881, 0.877, 0.868, 0.868, 0.863, 0.856, 0.852, 0.848],
    "jitter": 0.0037
}

# Group B: Radical Environment - Optimal Solution (High Reactance - Adapted)
data_b = {
    "safety": [0.64, 0.6, 0.54, 0.54, 0.53, 0.62, 0.7, 0.8, 0.81, 0.82, 0.84, 0.87, 0.9],
    "theta": [0.69, 0.69, 0.69, 0.725, 0.763, 0.746, 0.711, 0.676, 0.651, 0.637, 0.638, 0.643, 0.65, 0.656, 0.655],
    "jitter": 0.0156
}

# Group C: Radical Environment - Strategy Mismatch (High Reactance - Mismatch)
data_c = {
    "safety": [0.6, 0.53, 0.35, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "satisfaction": [0.15, 0.09, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "theta": [0.9, 0.9, 0.9, 0.927, 0.933, 0.922, 0.913, 0.908, 0.904, 0.895, 0.885, 0.883, 0.875, 0.87, 0.859],
    "jitter": 0.0076
}

# Unified timeline (take the first 13 points for KPI comparison, use the first 15 points for Theta)
time_kpi = list(range(13))
time_theta = list(range(15))

# --------------------------
# 2. Drawing Settings
# --------------------------
sns.set_theme(style="white", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Color definitions
color_a = "#2c7bb6"  # Blue - Compliant/Stable
color_b = "#d7191c"  # Red - Radical/Adapted
color_c = "#fdae61"  # Orange/Gray - Mismatch/Collapse
color_gray = "#7f7f7f"

# --------------------------
# Left figure (a): System Collapse (System Collapse)
# Compare the Safety indicators of Group B (Adapted) vs Group C (Mismatch)
# --------------------------
ax1 = axes[0]

# Plot Group B Safety (Successful Adaptation)
ax1.plot(time_kpi, data_b['safety'], color=color_b, lw=3, label='Group B: Adapted Strategy', marker='o', markersize=6)

# Plot Group C Safety (Collapse)
ax1.plot(time_kpi, data_c['safety'], color=color_gray, lw=3, ls='--', label='Group C: Mismatched Strategy', marker='x',
         markersize=7)

# Plot Group C Satisfaction (Supplementary explanation of collapse)
ax1.plot(time_kpi, data_c['satisfaction'], color=color_c, lw=2, ls=':', alpha=0.7, label='Group C: Satisfaction')

# Add shadow to the collapse area
ax1.fill_between(time_kpi, 0, data_c['safety'], where=[x < 5 for x in time_kpi], color=color_gray, alpha=0.1)
ax1.axvspan(3, 13, color="#ffebee", alpha=0.3)  # Red background represents the danger zone
ax1.text(7, 0.3, "System Collapse\n(Metrics $\\to$ 0)", color=color_b, fontsize=12, fontweight='bold', ha='center',
         va='center')

# Annotation
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
# Right figure (b): Governance Cost Asymmetry (Theta Dynamics)
# Compare the Theta fluctuations of Group A (Compliant) vs Group B (Radical)
# --------------------------
ax2 = axes[1]

# Plot Group A Theta (Smooth)
ax2.plot(time_theta, data_a['theta'], color=color_a, lw=2.5, label='Group A: Low Reactance Env.', marker='s',
         markersize=5)

# Plot Group B Theta (Fluctuation)
ax2.plot(time_theta, data_b['theta'], color=color_b, lw=2.5, label='Group B: High Reactance Env.', marker='^',
         markersize=6)

# Calculate and annotate Jitter difference
# Use error bars or annotations to display Jitter (Sigma)
ax2.text(8, 0.92, f"Group A Stability:\nLow Jitter ($\\sigma={data_a['jitter']:.4f}$)",
         color=color_a, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=color_a, alpha=0.8))

ax2.text(8, 0.77, f"Group B Volatility:\nHigh Jitter ($\\sigma={data_b['jitter']:.4f}$)",
         color=color_b, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=color_b, alpha=0.8))

# Add double arrows to represent the fluctuation amplitude
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