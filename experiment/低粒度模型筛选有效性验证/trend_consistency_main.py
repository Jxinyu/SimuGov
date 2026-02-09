import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import scipy.stats as stats
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


# =========================================================================
# 1. 核心计算逻辑 (复刻 NSGA-II 适应度函数)
# =========================================================================

def calculate_stable_score(data_list: List[float], penalty_weight: float = 1.0) -> float:
    if not data_list: return 0.0
    arr = np.array(data_list)
    return float(np.mean(arr) - (penalty_weight * np.std(arr)))


def calculate_theta_jitter(theta_list: List[float]) -> float:
    if not theta_list or len(theta_list) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(np.array(theta_list)))))


def compute_final_metrics(kpi_data: Dict[str, Any]) -> Dict[str, float]:
    def get_s(k):
        v = kpi_data.get(k, [])
        return v if isinstance(v, list) else ([float(v)] if v is not None else [])

    # 计算带惩罚的得分
    s = calculate_stable_score(get_s('safety'), 1.0)
    c = calculate_stable_score(get_s('creativity'), 1.0)
    sat = calculate_stable_score(get_s('satisfaction'), 0.8)
    jitter = calculate_theta_jitter(get_s('theta')) * 2.0

    return {'safety': s - jitter, 'creativity': c - jitter, 'satisfaction': sat - jitter, 'jitter': jitter}


# =========================================================================
# 2. 数据读取
# =========================================================================

def _extract_metrics_from_root(root_path: str) -> Dict[str, Dict[str, float]]:
    results = {}
    root = Path(root_path)
    if not root.exists(): return results

    for strategy_dir in root.iterdir():
        if not strategy_dir.is_dir(): continue
        # 寻找最后一天
        day_dirs = [d for d in strategy_dir.iterdir() if d.is_dir() and d.name.startswith("day_time_")]
        if not day_dirs: continue
        last_day = max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))

        try:
            with open(last_day / "output_system_kpi.json", 'r', encoding='utf-8') as f:
                kpi = json.load(f)
            with open(last_day / "output_policy.json", 'r', encoding='utf-8') as f:
                pol = json.load(f)

            sig = f"edu_{pol.get('e_edu')}_ai_{float(pol.get('ai_threshold', 0)):.2f}_f_{float(pol.get('f_penalty', 0)):.2f}"
            results[sig] = compute_final_metrics(kpi)
        except:
            continue
    return results


# =========================================================================
# 3. 帕累托与精英验证 (核心升级)
# =========================================================================

def _dominates(p1: Dict[str, float], p2: Dict[str, float]) -> bool:
    keys = ['safety', 'creativity', 'satisfaction']
    return all(p1[k] >= p2[k] for k in keys) and any(p1[k] > p2[k] for k in keys)


def _calculate_pareto_rank(metrics_dict: Dict[str, Dict[str, float]]) -> List[str]:
    policies = list(metrics_dict.keys())
    # 简化版帕累托排序实现
    domination_counts = {p: 0 for p in policies}
    for p in policies:
        for q in policies:
            if p == q: continue
            if _dominates(metrics_dict[q], metrics_dict[p]):  # q支配p
                domination_counts[p] += 1

    # 排序键: (被支配次数 [越少越好], -平均分 [越大越好])
    # 注意：这里直接用被支配次数作为粗略的 Rank，Domination Count 越小越接近前沿
    def sort_key(p):
        d = metrics_dict[p]
        avg = (d['safety'] + d['creativity'] + d['satisfaction']) / 3
        return (domination_counts[p], -avg)

    return sorted(policies, key=sort_key)


def verify_elite_group_retention(
        simple_data_path: str,
        complete_data_path: str,
        elite_ratio: float = 0.4  # 定义前 40% 为“精英区”
) -> Dict[str, Any]:
    logger.info(f">>> 开始精英群体保留验证 (Top {int(elite_ratio * 100)}%) <<<")

    # 1. 提取与对齐
    s_metrics = _extract_metrics_from_root(simple_data_path)
    c_metrics = _extract_metrics_from_root(complete_data_path)
    common = set(s_metrics.keys()) & set(c_metrics.keys())
    count = len(common)

    if count < 3: return {"status": "error", "msg": "样本太少"}

    # 2. 计算两边的全量排名
    s_data = {k: v for k, v in s_metrics.items() if k in common}
    c_data = {k: v for k, v in c_metrics.items() if k in common}

    lf_rank_list = _calculate_pareto_rank(s_data)
    hf_rank_list = _calculate_pareto_rank(c_data)

    # 3. 定义“真·精英”集合 (Complete 中的前 K 名)
    k_elites = max(1, int(count * elite_ratio))
    true_elites = hf_rank_list[:k_elites]  # 这是一个列表，包含策略签名

    # 4. 分析这些精英在 LF 中的下落
    retention_stats = []
    recalled_count = 0

    for elite_policy in true_elites:
        # 在 HF 中的排名 (从1开始)
        hf_rank = hf_rank_list.index(elite_policy) + 1
        # 在 LF 中的排名
        lf_rank = lf_rank_list.index(elite_policy) + 1

        # 是否被召回？(LF排名是否也在前 K 名之内)
        is_recalled = lf_rank <= k_elites
        if is_recalled: recalled_count += 1

        retention_stats.append({
            "policy": elite_policy,
            "hf_rank": hf_rank,
            "lf_rank": lf_rank,
            "is_recalled": is_recalled
        })

    # 计算召回率
    recall_rate = recalled_count / k_elites

    # 计算整体 Spearman
    s_indices = [lf_rank_list.index(p) for p in common]
    c_indices = [hf_rank_list.index(p) for p in common]
    corr, _ = stats.spearmanr(s_indices, c_indices)

    # 5. 准备打印数据
    def get_details(rank_list, source):
        return [{"rank": i + 1, "policy": p,
                 "scores": f"{source[p]['safety']:.2f}/{source[p]['creativity']:.2f}/{source[p]['satisfaction']:.2f}"}
                for i, p in enumerate(rank_list)]

    return {
        "status": "success",
        "meta": {"sample_count": count, "elite_count": k_elites, "ratio": elite_ratio},
        "metrics": {
            "spearman": round(corr if not np.isnan(corr) else 0, 4),
            "elite_recall": round(recall_rate, 4)
        },
        "elite_analysis": retention_stats,
        "full_rankings": {
            "LF": get_details(lf_rank_list, s_data),
            "HF": get_details(hf_rank_list, c_data)
        }
    }


def print_elite_report(res: Dict):
    meta = res['meta']
    elites = res['elite_analysis']

    print("\n" + "=" * 100)
    print(f"{f'【精英群体保留能力分析】 (Top {int(meta["ratio"] * 100)}% = {meta["elite_count"]} 策略)':^90}")
    print("=" * 100)

    print(f"{'True Rank (HF)':<15} | {'Predicted Rank (LF)':<20} | {'Status':<10} | {'Policy Signature'}")
    print("-" * 100)

    for item in elites:
        status = "✅ 召回" if item['is_recalled'] else "⚠️ 丢失"
        # 即使丢失，如果还在前 50%，也可以标个黄
        if not item['is_recalled'] and item['lf_rank'] <= meta['sample_count'] * 0.6:
            status = "🆗 尚可"

        p_str = item['policy'].replace("edu_", "E").replace("_ai_", " A").replace("_f_", " F")
        print(f"{item['hf_rank']:<15} | {item['lf_rank']:<20} | {status:<10} | {p_str}")

    print("-" * 100)
    print(f"统计摘要: 样本总数 {meta['sample_count']}, 精英数 {meta['elite_count']}")
    print(f"指标结果: 精英召回率 (Recall) = {res['metrics']['elite_recall'] * 100:.1f}%")
    print(f"          整体排序相关性 (Spearman) = {res['metrics']['spearman']:.3f}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    path_s = r"experiment\低粒度模型筛选有效性验证\验证通过\result\4\simple"
    path_c = r"experiment\低粒度模型筛选有效性验证\验证通过\result\4\complete"

    # 验证前 40% 的精英策略是否被 LF 留住
    # 如果你有 10 个策略，这就看前 4 名；如果有 5 个策略，看前 2 名
    result = verify_elite_group_retention(path_s, path_c, elite_ratio=0.4)

    print_elite_report(result)
