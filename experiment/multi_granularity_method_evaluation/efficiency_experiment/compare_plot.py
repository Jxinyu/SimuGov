import json
import os
import re
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def set_kdd_style():
    # 强制嵌入 TrueType 字体 (KDD/ACM 必需)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # 字体切换为无衬线字体 (Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    # 配色方案：深蓝 (Coupled) vs 亮橙 (Baseline)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#E67E22', '#1F77B4'])


# 模型单价 (单位：元 / 1M Tokens)
PRICING = {
    "qwen-flash": 0.1,
    "qwen-plus": 0.8,
    "qwen-max": 20.0,
    "default": 20.0
}

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# ==========================================
# 2. 核心逻辑：解析与计算 (逻辑保持不变)
# ==========================================
def parse_tokens_from_logs(log_dir: Path) -> Dict[str, Dict[str, int]]:
    stats = {}
    token_pattern = re.compile(r"['\"]total_tokens['\"]:\s*(\d+)")
    model_pattern = re.compile(r"['\"]model_name['\"]:\s*['\"]([^'\"]+)['\"]")

    if not log_dir.exists():
        return stats

    for log_file in log_dir.glob("*.log"):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "usage_metadata" not in line:
                    continue
                token_match = token_pattern.search(line)
                if not token_match:
                    continue
                total_t = int(token_match.group(1))
                if total_t == 0: continue
                model_match = model_pattern.search(line)
                m_name = model_match.group(1) if model_match else "default"
                if m_name not in stats:
                    stats[m_name] = {"total": 0, "count": 0}
                stats[m_name]["total"] += total_t
                stats[m_name]["count"] += 1
    return stats


def calculate_financial_cost(token_stats: Dict[str, Dict[str, int]]) -> float:
    total_cost = 0.0
    for model, usage in token_stats.items():
        price = PRICING["default"]
        for k, v in PRICING.items():
            if k in model:
                price = v
                break
        cost = (usage["total"] / 1_000_000.0) * price
        total_cost += cost
    return total_cost


def analyze_scalability_with_costs(data_root: str) -> List[Dict[str, Any]]:
    root = Path(data_root)
    all_results = []
    timestamp_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])

    for ts_dir in timestamp_dirs:
        summary_file = ts_dir / "experiment_results.json"
        if not summary_file.exists():
            continue
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        params = summary.get("parameters", {})
        pop = params.get("population", params.get("population_size", 0))
        gen = params.get("generations", 0)
        scale = pop * gen

        group_a = summary.get("groups", {}).get("Group_A_Baseline", {})
        group_b = summary.get("groups", {}).get("Group_B_Proposed", {})
        time_a = group_a.get("time_minutes", 0)
        time_b = group_b.get("time_minutes", 0)

        log_dir_a = ts_dir / "Group_A_全高粒度" / "logs"
        log_dir_b = ts_dir / "Group_B_高低粒度" / "logs"
        tokens_a = parse_tokens_from_logs(log_dir_a)
        tokens_b = parse_tokens_from_logs(log_dir_b)
        cost_a = calculate_financial_cost(tokens_a)
        cost_b = calculate_financial_cost(tokens_b)

        all_results.append({
            "timestamp": ts_dir.name,
            "scale": scale,
            "time_a": time_a,
            "time_b": time_b,
            "cost_a": cost_a,
            "cost_b": cost_b,
            "saving": (1 - cost_b / cost_a) * 100 if cost_a > 0 else 0
        })
        log.info(f"✅ Processed: {ts_dir.name} | Scale:{scale} | Cost A:{cost_a:.4f} vs B:{cost_b:.4f}")

    all_results.sort(key=lambda x: x["scale"])
    return all_results


# ==========================================
# 3. 增强的绘图逻辑（含数据保存）
# ==========================================
def plot_comparison_charts(results: List[Dict], output_dir: str):
    if not results: return
    set_kdd_style()
    df = pd.DataFrame(results)

    # --- 新增：保存绘图底层数据 (CSV & JSON) ---
    csv_path = os.path.join(output_dir, "scalability_metrics.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"💾 Plotting data saved to: {csv_path}")

    # 创建符合 KDD 双栏宽度的画布 (约 7 英寸)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

    # X轴：Scale
    x = df["scale"]

    # --- 图1：时间效率 (a) ---
    ax1.plot(x, df["time_a"], 's--', color='#E67E22', label='Baseline (Full High-Fid.)', markersize=6)
    ax1.plot(x, df["time_b"], 'o-', color='#1F77B4', label='Proposed (Coupled)', markersize=6, linewidth=2)

    ax1.set_xlabel('Experiment Scale (Pop × Gen)', fontweight='bold')
    ax1.set_ylabel('Execution Time (Min)', fontweight='bold')
    ax1.set_title('(a) Scalability of Time Efficiency', y=-0.3)
    ax1.grid(True, linestyle=':')

    # 动态标注加速比
    last_idx = df.index[-1]
    speedup = df.iloc[last_idx]["time_a"] / df.iloc[last_idx]["time_b"]
    ax1.annotate(f"{speedup:.1f}× Faster",
                 xy=(df.iloc[last_idx]["scale"], df.iloc[last_idx]["time_b"]),
                 xytext=(-10, 15), textcoords='offset points',
                 ha='right', fontweight='bold', color='#1F77B4', fontsize=9)

    # --- 图2：经济成本 (b) ---
    ax2.plot(x, df["cost_a"], 's--', color='#E67E22', label='Baseline (Full High-Fid.)', markersize=6)
    ax2.plot(x, df["cost_b"], 'o-', color='#1F77B4', label='Proposed (Coupled)', markersize=6, linewidth=2)

    ax2.set_xlabel('Experiment Scale (Pop × Gen)', fontweight='bold')
    ax2.set_ylabel('Financial Cost (RMB)', fontweight='bold')
    ax2.set_title('(b) Scalability of Financial Cost', y=-0.3)
    ax2.grid(True, linestyle=':')

    # 动态标注节约率
    saving = df.iloc[last_idx]["saving"]
    ax2.annotate(f"-{saving:.1f}% Cost",
                 xy=(df.iloc[last_idx]["scale"], df.iloc[last_idx]["cost_b"]),
                 xytext=(-10, 15), textcoords='offset points',
                 ha='right', fontweight='bold', color='#1F77B4', fontsize=9)

    # 全局图例 (顶部水平排列)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, frameon=False, fontsize=10)

    plt.tight_layout()

    # 同时保存 PDF (矢量) 和 PNG (高清)
    pdf_path = os.path.join(output_dir, "scalability_comparison_final.pdf")
    png_path = os.path.join(output_dir, "scalability_comparison_final.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')

    print(f"\n📊 Figure and Data saved to: {output_dir}")


# ==========================================
# 4. 主入口
# ==========================================
if __name__ == "__main__":
    DATA_PATH = r"experiment\Multi-granularity method evaluation\Efficiency Experiment\data"

    # 1. 分析数据
    final_data = analyze_scalability_with_costs(DATA_PATH)

    # 2. 打印控制台表格
    print("\n" + "=" * 85)
    print(f"{'Scale':<6} | {'Time A/B (min)':<18} | {'Cost A/B (RMB)':<18} | {'Saving %':<10}")
    print("-" * 85)
    for r in final_data:
        t_str = f"{r['time_a']:.1f}/{r['time_b']:.1f}"
        c_str = f"{r['cost_a']:.4f}/{r['cost_b']:.4f}"
        print(f"{r['scale']:<6} | {t_str:<18} | {c_str:<18} | {r['saving']:.1f}%")
    print("=" * 85)

    # 3. 绘制图表并自动保存数据
    if final_data:
        plot_comparison_charts(final_data, DATA_PATH)
    else:
        print("❌ No valid data for plotting")