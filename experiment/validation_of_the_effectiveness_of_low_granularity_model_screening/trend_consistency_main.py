import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import scipy.stats as stats
import numpy as np

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

    # Calculate score with penalty
    s = calculate_stable_score(get_s('safety'), 1.0)
    c = calculate_stable_score(get_s('creativity'), 1.0)
    sat = calculate_stable_score(get_s('satisfaction'), 0.8)
    jitter = calculate_theta_jitter(get_s('theta')) * 2.0

    return {'safety': s - jitter, 'creativity': c - jitter, 'satisfaction': sat - jitter, 'jitter': jitter}


def _extract_metrics_from_root(root_path: str) -> Dict[str, Dict[str, float]]:
    results = {}
    root = Path(root_path)
    if not root.exists(): return results

    for strategy_dir in root.iterdir():
        if not strategy_dir.is_dir(): continue
        # Search for the last day
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


def _dominates(p1: Dict[str, float], p2: Dict[str, float]) -> bool:
    keys = ['safety', 'creativity', 'satisfaction']
    return all(p1[k] >= p2[k] for k in keys) and any(p1[k] > p2[k] for k in keys)


def _calculate_pareto_rank(metrics_dict: Dict[str, Dict[str, float]]) -> List[str]:
    policies = list(metrics_dict.keys())
    domination_counts = {p: 0 for p in policies}
    for p in policies:
        for q in policies:
            if p == q: continue
            if _dominates(metrics_dict[q], metrics_dict[p]):  # q dominates p
                domination_counts[p] += 1

    def sort_key(p):
        d = metrics_dict[p]
        avg = (d['safety'] + d['creativity'] + d['satisfaction']) / 3
        return (domination_counts[p], -avg)

    return sorted(policies, key=sort_key)


def verify_elite_group_retention(
        simple_data_path: str,
        complete_data_path: str,
        elite_ratio: float = 0.4  # Define the top 40% as the "elite zone"
) -> Dict[str, Any]:
    logger.info(f">>> Starting elite group retention verification (Top {int(elite_ratio * 100)}%) <<<")

    s_metrics = _extract_metrics_from_root(simple_data_path)
    c_metrics = _extract_metrics_from_root(complete_data_path)
    common = set(s_metrics.keys()) & set(c_metrics.keys())
    count = len(common)

    if count < 3: return {"status": "error", "msg": "Sample too few"}

    s_data = {k: v for k, v in s_metrics.items() if k in common}
    c_data = {k: v for k, v in c_metrics.items() if k in common}

    lf_rank_list = _calculate_pareto_rank(s_data)
    hf_rank_list = _calculate_pareto_rank(c_data)

    k_elites = max(1, int(count * elite_ratio))
    true_elites = hf_rank_list[:k_elites]

    retention_stats = []
    recalled_count = 0

    for elite_policy in true_elites:

        hf_rank = hf_rank_list.index(elite_policy) + 1

        lf_rank = lf_rank_list.index(elite_policy) + 1


        is_recalled = lf_rank <= k_elites
        if is_recalled: recalled_count += 1

        retention_stats.append({
            "policy": elite_policy,
            "hf_rank": hf_rank,
            "lf_rank": lf_rank,
            "is_recalled": is_recalled
        })

    recall_rate = recalled_count / k_elites

    s_indices = [lf_rank_list.index(p) for p in common]
    c_indices = [hf_rank_list.index(p) for p in common]
    corr, _ = stats.spearmanr(s_indices, c_indices)

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
    print(f"{f'[Elite Group Retention Ability Analysis] (Top {int(meta["ratio"] * 100)}% = {meta["elite_count"]} policies)':^90}")
    print("=" * 100)

    print(f"{'True Rank (HF)':<15} | {'Predicted Rank (LF)':<20} | {'Status':<10} | {'Policy Signature'}")
    print("-" * 100)

    for item in elites:
        status = "✅ Recalled" if item['is_recalled'] else "⚠️ Lost"
        # Even if lost, if still in the top 50%, can be marked in yellow
        if not item['is_recalled'] and item['lf_rank'] <= meta['sample_count'] * 0.6:
            status = "🆗 Passable"

        p_str = item['policy'].replace("edu_", "E").replace("_ai_", " A").replace("_f_", " F")
        print(f"{item['hf_rank']:<15} | {item['lf_rank']:<20} | {status:<10} | {p_str}")

    print("-" * 100)
    print(f"Statistical Summary: total sample count {meta['sample_count']}, elite count {meta['elite_count']}")
    print(f"Indicator results: elite recall rate (Recall) = {res['metrics']['elite_recall'] * 100:.1f}%")
    print(f"          Overall ranking correlation (Spearman) = {res['metrics']['spearman']:.3f}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    path_s = r"experiment\Validation of the effectiveness of low-granularity model screening\Verification passed\result\4\simple"
    path_c = r"experiment\Validation of the effectiveness of low-granularity model screening\Verification passed\result\4\complete"

    result = verify_elite_group_retention(path_s, path_c, elite_ratio=0.4)

    print_elite_report(result)