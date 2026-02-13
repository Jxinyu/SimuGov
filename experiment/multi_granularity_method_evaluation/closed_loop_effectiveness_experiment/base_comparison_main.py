import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pygmo as pg
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Publication-level plotting configuration ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_theta_jitter(theta_history: list) -> float:
    if not theta_history or len(theta_history) < 2: return 0.0
    clean_theta = [t for t in theta_history if t is not None]
    if len(clean_theta) < 2: return 0.0
    diffs = [abs(clean_theta[i] - clean_theta[i - 1]) for i in range(1, len(clean_theta))]
    return float(np.mean(diffs))


def calculate_robust_score(kpi_series: list, theta_hist: list, penalty_weight: float = 1.0,
                           jitter_weight: float = 2.0) -> float:
    if not kpi_series: return 0.0
    data = np.array(kpi_series)
    score = np.mean(data) - (penalty_weight * np.std(data)) - (calculate_theta_jitter(theta_hist) * jitter_weight)
    return float(max(0.0, score))


def get_robust_vector(snapshot):
    th = snapshot['theta_list']
    return np.array([
        calculate_robust_score(snapshot['safety_list'], th),
        calculate_robust_score(snapshot['creativity_list'], th),
        calculate_robust_score(snapshot['satisfaction_list'], th)
    ])


def extract_snapshot(folder: Path):
    day_dirs = sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith('day_time_')],
                      key=lambda x: int(x.name.split('_')[-1]))
    if not day_dirs: return None
    last_day = day_dirs[-1]
    try:
        with open(last_day / "output_system_kpi.json", 'r', encoding='utf-8') as f:
            kpi_j = json.load(f)
        with open(last_day / "output_policy.json", 'r', encoding='utf-8') as f:
            policy = json.load(f)
        return {
            "id": folder.name, "policy": policy,
            "safety_list": kpi_j.get('safety', []),
            "creativity_list": kpi_j.get('creativity', []),
            "satisfaction_list": kpi_j.get('satisfaction', []),
            "theta_list": kpi_j.get('theta', [])
        }
    except:
        return None


def load_all_experiment_data(base_path: Path):
    data = {"elites": [], "baselines": [], "run_id": "None"}
    b_dir = base_path / "baseline"
    if b_dir.exists():
        for f in b_dir.iterdir():
            if f.is_dir():
                res = extract_snapshot(f)
                if res: data["baselines"].append(res)
    all_runs = [d for d in base_path.iterdir() if d.is_dir() and d.name not in ["baseline", "output"]]
    if not all_runs: return data
    latest = sorted(all_runs)[-1]
    data["run_id"] = latest.name
    for f in latest.iterdir():
        if f.is_dir() and f.name != "简化":
            res = extract_snapshot(f)
            if res: data["elites"].append(res)
    return data


def run_elite_optimal_benchmarking(data, output_root: Path):
    if not data['elites'] or not data['baselines']:
        print("❌ 数据不足，无法对比。")
        return

    # A. Space transformation
    elite_vectors = np.array([get_robust_vector(e) for e in data['elites']])
    base_vectors = np.array([get_robust_vector(b) for b in data['baselines']])

    # B. Finding the optimal solution in the elite set (based on comprehensive robust total score)
    elite_total_scores = np.sum(elite_vectors, axis=1)
    best_idx = np.argmax(elite_total_scores)
    best_v = elite_vectors[best_idx]
    best_p = data['elites'][best_idx]

    # C. Item-by-item benchmarking calculation (absolute value + percentage)
    detailed_results = []
    dim_names = ["安全性", "创造力", "满意度"]

    for i, b_v in enumerate(base_vectors):
        diff = best_v - b_v
        # Avoid division by zero, set a minimal value
        ratios = diff / (b_v + 1e-6)

        entry = {
            "baseline_id": data['baselines'][i]['id'],
            "comparison": {}
        }
        for idx, name in enumerate(dim_names):
            entry["comparison"][name] = {
                "base_val": float(b_v[idx]),
                "elite_val": float(best_v[idx]),
                "abs_diff": float(diff[idx]),
                "percent_change": f"{ratios[idx] * 100:+.2f}%"
            }
        detailed_results.append(entry)

    # D. Overall gain
    def calc_hv(matrix):
        min_m = -1.0 * matrix
        try:
            return float(pg.hypervolume(min_m).compute([0.01, 0.01, 0.01]))
        except:
            return 0.0

    hv_elite = calc_hv(elite_vectors)
    hv_base = calc_hv(base_vectors)

    # E. Drawing and archiving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = output_root / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Drawing ---
    fig = plt.figure(figsize=(12, 9), dpi=250)
    ax = fig.add_subplot(111, projection='3d')
    # Background elites
    ax.scatter(elite_vectors[:, 0], elite_vectors[:, 1], elite_vectors[:, 2], c='lightcoral', s=40, alpha=0.3,
               label='精英解集 (Elite Set)')
    # Baselines
    ax.scatter(base_vectors[:, 0], base_vectors[:, 1], base_vectors[:, 2], c='blue', s=160, marker='*',
               edgecolors='black', label='人工基准 (Baselines)')
    # Optimal elite highlight
    ax.scatter(best_v[0], best_v[1], best_v[2], c='gold', s=550, marker='P', edgecolors='black', linewidth=2,
               label='精英解集最优解 (Optimal Elite)', zorder=30)

    ax.set_xlabel('稳健安全性');
    ax.set_ylabel('稳健创造力');
    ax.set_zlabel('稳健满意度')
    ax.set_title(f"治理策略稳健性空间对比 (最优解 vs 人工基准)\nRun: {data['run_id']}")
    ax.view_init(elev=20, azim=135)
    ax.legend(loc='upper left', fontsize=10)
    plt.savefig(save_dir / "optimal_elite_comparison.png")

    # F. Construct console text output and capture
    output_lines = []
    output_lines.append("\n" + "★" * 25 + " 闭环寻优有效性深度报告 " + "★" * 25)
    output_lines.append(f"📡 实验数据源: {data['run_id']}")
    output_lines.append(f"🥇 精英解集最优解: {best_p['policy']}")
    output_lines.append(f"📊 稳健性坐标: 安全:{best_v[0]:.4f} | 创造:{best_v[1]:.4f} | 满意:{best_v[2]:.4f}")
    output_lines.append("-" * 75)
    output_lines.append(
        f"📈 [边界增益] 治理超体积(HV)扩张率: {round((hv_elite / hv_base - 1) * 100, 2) if hv_base > 0 else 0}%")
    output_lines.append("-" * 75)
    output_lines.append("🔍 维度级逐项对标分析 (最优精英 vs 人工基准):")

    for entry in detailed_results:
        output_lines.append(f"  ➤ 基准策略: {entry['baseline_id']}")
        for dim in dim_names:
            comp = entry['comparison'][dim]
            status = "提升" if comp['abs_diff'] >= 0 else "代价/下降"
            output_lines.append(
                f"     • {dim}: 基准({comp['base_val']:.3f}) -> 最优({comp['elite_val']:.3f}) | {status}: {comp['percent_change']} (绝对值:{comp['abs_diff']:+.4f})")
        output_lines.append("")

    output_lines.append("=" * 75)
    full_output_text = "\n".join(output_lines)

    # Print to console
    print(full_output_text)

    # G. Save files
    # 1. Save console text
    with open(save_dir / "execution_summary.txt", "w", encoding="utf-8") as f:
        f.write(full_output_text)

    # 2. Save JSON results
    report_json = {
        "metadata": {"run_id": data['run_id'], "best_policy_id": best_p['id']},
        "metrics": {
            "hv_elite": hv_elite, "hv_base": hv_base,
            "best_elite_robust_vector": best_v.tolist()
        },
        "benchmarking": detailed_results
    }
    with open(save_dir / "benchmarking_data.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4, ensure_ascii=False)

    print(f"✅ 3D可视化图、JSON数据及本控制台文本已保存至: {save_dir}")


if __name__ == "__main__":
    BASE_PATH = Path(r"experiment\Multi-granularity method evaluation\Closed-loop effectiveness experiment\data")
    OUT_PATH = Path(r"experiment\Multi-granularity method evaluation\Closed-loop effectiveness experiment\output")

    exp_data = load_all_experiment_data(BASE_PATH)
    run_elite_optimal_benchmarking(exp_data, OUT_PATH)