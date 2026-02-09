import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 新增：用于数据保存
from scipy import stats
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. KDD Paper Style Configuration
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.8

# ==========================================
# 2. 配色方案
# ==========================================
COLOR_GT = '#ff7f0e'
COLOR_SIM = '#4682B4'
COLOR_CI = '#4682B4'
COLOR_BG_OUTBREAK = '#f39c12'
COLOR_BG_COMPROMISE = '#7f8c8d'


def load_all_runs(root_dir, folder_names):
    """
    加载数据逻辑保持不变
    """
    all_sim_data = []
    ground_truth = None

    print(f"📂 Loading data from: {root_dir}")

    for folder in folder_names:
        folder_path = Path(root_dir) / folder
        if not folder_path.exists():
            continue

        json_files = list(folder_path.glob("validation_report_*.json"))
        if not json_files:
            continue

        latest_json = max(json_files, key=os.path.getmtime)

        with open(latest_json, 'r', encoding='utf-8') as f:
            try:
                report = json.load(f)
                sim_list = report['data']['simulation']
                if not sim_list or len(sim_list) == 0:
                    continue

                all_sim_data.append(sim_list)
                if ground_truth is None:
                    ground_truth = report['data']['ground_truth']
            except Exception:
                pass

    if not all_sim_data:
        return None, None

    min_len = min(len(x) for x in all_sim_data)
    aligned_sim_data = [x[:min_len] for x in all_sim_data]
    gt_vec = np.array(ground_truth)[:min_len]

    return np.array(aligned_sim_data), gt_vec


def plot_subplot(ax, gt_data, sim_matrix, title, subplot_id):
    """
    绘制单个子图，逻辑保持不变
    """
    if sim_matrix is None or gt_data is None:
        ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
        return

    n_runs, n_days = sim_matrix.shape
    days = np.arange(1, n_days + 1)

    mean_sim = np.mean(sim_matrix, axis=0)
    sem_sim = stats.sem(sim_matrix, axis=0) if n_runs > 1 else np.zeros(n_days)

    ax.axvspan(8, 12, color=COLOR_BG_OUTBREAK, alpha=0.08, label='Outbreak' if subplot_id == '(a)' else None)
    ax.axvspan(19, n_days, color=COLOR_BG_COMPROMISE, alpha=0.1, label='Compromise' if subplot_id == '(a)' else None)

    ax.plot(days, gt_data, label='Ground Truth', color=COLOR_GT, linestyle='--', linewidth=1.8, alpha=0.9, zorder=3)
    ax.plot(days, mean_sim, label='Simulation Mean', color=COLOR_SIM, linewidth=2.0, marker='o', markersize=4.5,
            markevery=2, zorder=4)
    ax.fill_between(days, mean_sim - sem_sim, mean_sim + sem_sim, color=COLOR_CI, alpha=0.2, label='Standard Error',
                    zorder=2)

    ax.set_xlabel("Time (Day)")
    if subplot_id == '(a)':
        ax.set_ylabel("Protest Proportion")

    ax.set_title(f"{subplot_id} {title}", y=-0.28, fontsize=11, fontweight='bold', color='black')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, n_days + 0.5)
    ax.grid(True, linestyle=':', alpha=0.5, color='gray', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if subplot_id == '(a)':
        ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='white', framealpha=0.95, fontsize=8,
                  ncol=1)


if __name__ == "__main__":
    BASE_ROOT = r"experiment\仿真社会评估\案例验证\验证通过"
    FOLDERS = ["1", "2", "3", "4", "5"]

    experiments = [
        {"folder": "关闭心理参数", "display_title": "Baseline", "subplot_id": "(a)"},
        {"folder": "开启心理参数", "display_title": "SimuGov (Ours)", "subplot_id": "(b)"}
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=300, sharey=True)

    # 用于收集待保存的数据
    all_metrics = []

    try:
        for idx, exp_cfg in enumerate(experiments):
            full_path = os.path.join(BASE_ROOT, exp_cfg["folder"])
            sim_mat, gt_vec = load_all_runs(full_path, FOLDERS)

            if sim_mat is not None:
                # 1. 计算统计量供保存
                mean_vals = np.mean(sim_mat, axis=0)
                sem_vals = stats.sem(sim_mat, axis=0)
                days = np.arange(1, len(gt_vec) + 1)

                # 2. 存入列表
                for i in range(len(days)):
                    all_metrics.append({
                        "Group": exp_cfg["display_title"],
                        "Day": days[i],
                        "Ground_Truth": gt_vec[i],
                        "Sim_Mean": mean_vals[i],
                        "Sim_SEM": sem_vals[i]
                    })

                # 3. 绘图
                plot_subplot(axes[idx], gt_vec, sim_mat, exp_cfg["display_title"], exp_cfg["subplot_id"])

        # ================= 1. 保存绘图底层数据 =================
        timestamp = datetime.now().strftime("%H%M")
        if all_metrics:
            df_save = pd.DataFrame(all_metrics)
            csv_path = os.path.join(BASE_ROOT, f"KDD_Ablation_Data_Export_{timestamp}.csv")
            df_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"📊 Data Saved Successfully: {csv_path}")

        # ================= 2. 保存图形 =================
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.12)

        pdf_path = os.path.join(BASE_ROOT, f"KDD_Ablation_OrangeBlue_{timestamp}.pdf")
        png_path = os.path.join(BASE_ROOT, f"KDD_Ablation_OrangeBlue_{timestamp}.png")

        plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')

        print(f"✅ Plots Generated Successfully!")
        print(f"   Saved PDF: {pdf_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()