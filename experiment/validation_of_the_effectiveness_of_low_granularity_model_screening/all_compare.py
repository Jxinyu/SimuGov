import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import scipy.stats as stats
import numpy as np

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        day_dirs = [d for d in strategy_dir.iterdir() if d.is_dir() and d.name.startswith("day_time_")]
        if not day_dirs: continue
        last_day = max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))

        try:
            with open(last_day / "output_system_kpi.json", 'r', encoding='utf-8') as f:
                kpi = json.load(f)
            with open(last_day / "output_policy.json", 'r', encoding='utf-8') as f:
                pol = json.load(f)

            # Format Key for alignment
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
            if _dominates(metrics_dict[q], metrics_dict[p]):
                domination_counts[p] += 1

    def sort_key(p):
        d = metrics_dict[p]
        avg = (d['safety'] + d['creativity'] + d['satisfaction']) / 3
        return (domination_counts[p], -avg)

    return sorted(policies, key=sort_key)


def verify_elite_group_retention_multi_ratios(
        simple_data_path: str,
        complete_data_path: str,
        elite_ratios: List[float] = [0.2, 0.3, 0.4]
) -> Dict[str, Any]:
    """
    Calculate results for multiple elite ratios simultaneously
    """
    # 1. Extract and align
    s_metrics = _extract_metrics_from_root(simple_data_path)
    c_metrics = _extract_metrics_from_root(complete_data_path)
    common = set(s_metrics.keys()) & set(c_metrics.keys())
    count = len(common)

    if count < 3:
        return {"status": "error", "msg": f"Sample size too small (n={count})"}

    # 2. Calculate rankings
    s_data = {k: v for k, v in s_metrics.items() if k in common}
    c_data = {k: v for k, v in c_metrics.items() if k in common}

    lf_rank_list = _calculate_pareto_rank(s_data)
    hf_rank_list = _calculate_pareto_rank(c_data)

    # Calculate Spearman correlation coefficient (only need to calculate once, independent of elite ratio)
    s_indices = [lf_rank_list.index(p) for p in common]
    c_indices = [hf_rank_list.index(p) for p in common]
    corr, _ = stats.spearmanr(s_indices, c_indices)
    spearman_value = round(corr if not np.isnan(corr) else 0, 4)
    top40_spearman = calculate_top_percentage_spearman(lf_rank_list, hf_rank_list, common, 0.4)
    print(f"Spearman: {spearman_value}, Top40 Spearman: {top40_spearman}")

    # 3. Analyze multiple elite ratios
    all_ratios_results = {}
    elite_analysis_dict = {}

    for ratio in elite_ratios:
        # Elite definition
        k_elites = max(1, int(count * ratio))
        true_elites = hf_rank_list[:k_elites]

        # Analysis
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

        # Save results of this ratio
        ratio_key = f"ratio_{int(ratio * 100)}"
        all_ratios_results[ratio_key] = {
            "recall_rate": round(recall_rate, 4),
            "elite_count": k_elites,
            "sample_count": count
        }

        # Save detailed analysis
        elite_analysis_dict[ratio_key] = retention_stats

    return {
        "status": "success",
        "meta": {
            "sample_count": count,
            "elite_ratios": elite_ratios
        },
        "metrics": {
            "spearman": spearman_value,
            **all_ratios_results
        },
        "elite_analysis": elite_analysis_dict,
        "rankings": {
            "hf_rank_list": hf_rank_list,
            "lf_rank_list": lf_rank_list
        }
    }


def print_batch_summary_table_multi_ratios(all_results: List[Dict], elite_ratios: List[float]):
    """
    Print summary table for all groups, supports multiple elite ratios
    """
    print("\n" + "=" * 100)

    # Generate title
    ratio_strs = [f"Top {int(ratio * 100)}%" for ratio in elite_ratios]
    title = f"[Batch Experiment Summary Report - Multi-Elite Ratio Analysis]"
    print(f"{title:^90}")
    print("=" * 100)

    # Table header
    header = f"{'Group ID':<10} | {'Sample Count':<12} | {'Spearman':<10} "
    for ratio_str in ratio_strs:
        header += f"| {ratio_str + ' Recall':<12} "
    print(header)
    print("-" * 100)

    # Initialize statistics list
    spearman_list = []
    recall_lists = {f"ratio_{int(ratio * 100)}": [] for ratio in elite_ratios}

    # Print result of each group
    for res in all_results:
        if res['status'] != 'success':
            continue

        gid = res['group_id']
        metrics = res['metrics']

        # Extract data
        sample_count = res['meta']['sample_count']
        spearman_val = metrics['spearman']

        # Collect data for average calculation
        spearman_list.append(spearman_val)

        # Format output row
        row = f"{gid:<10} | {sample_count:<12} | {spearman_val:<10.4f} "

        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            recall_val = metrics[ratio_key]["recall_rate"]
            recall_lists[ratio_key].append(recall_val)
            row += f"| {recall_val * 100:>10.2f}%  "

        print(row)

    print("-" * 100)

    # Calculate and print average value
    if spearman_list:
        avg_spearman = np.mean(spearman_list)

        # Calculate average recall rate for each ratio
        avg_recalls = {}
        for ratio_key, recall_list in recall_lists.items():
            if recall_list:
                avg_recalls[ratio_key] = np.mean(recall_list)

        # Print average row
        avg_row = f"{'AVERAGE':<10} | {'-':<12} | {avg_spearman:<10.4f} "
        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            avg_recall = avg_recalls.get(ratio_key, 0)
            avg_row += f"| {avg_recall * 100:>10.2f}%  "

        print(avg_row)

    print("=" * 100 + "\n")

    # Return statistics data for saving
    stats_summary = {
        "spearman_mean": float(np.mean(spearman_list)) if spearman_list else 0.0,
        "recall_means": {
            ratio_key: float(np.mean(recall_list)) if recall_list else 0.0
            for ratio_key, recall_list in recall_lists.items()
        },
        "total_groups": len(all_results),
        "successful_groups": len([r for r in all_results if r['status'] == 'success'])
    }

    return stats_summary


def save_all_results_to_json(all_results: List[Dict], output_path: str):
    """
    Save all results to JSON file
    """
    # Prepare data for saving
    save_data = {
        "timestamp": np.datetime64('now').astype(str),
        "total_groups": len(all_results),
        "successful_groups": len([r for r in all_results if r['status'] == 'success']),
        "all_results": all_results,
        "summary_statistics": {}
    }

    # Calculate summary statistics (if all groups succeed)
    successful_results = [r for r in all_results if r['status'] == 'success']
    if successful_results:
        # Extract all ratios
        elite_ratios = successful_results[0]['meta']['elite_ratios']

        # Initialize statistics dictionary
        spearman_list = []
        recall_stats = {f"ratio_{int(ratio * 100)}": [] for ratio in elite_ratios}

        # Collect all data
        for res in successful_results:
            metrics = res['metrics']
            spearman_list.append(metrics['spearman'])

            for ratio in elite_ratios:
                ratio_key = f"ratio_{int(ratio * 100)}"
                if ratio_key in metrics:
                    recall_stats[ratio_key].append(metrics[ratio_key]["recall_rate"])

        # Calculate statistics
        summary_stats = {
            "spearman": {
                "mean": float(np.mean(spearman_list)) if spearman_list else 0.0,
                "std": float(np.std(spearman_list)) if spearman_list else 0.0,
                "min": float(np.min(spearman_list)) if spearman_list else 0.0,
                "max": float(np.max(spearman_list)) if spearman_list else 0.0
            },
            "recall_rates": {}
        }

        for ratio_key, recall_list in recall_stats.items():
            if recall_list:
                summary_stats["recall_rates"][ratio_key] = {
                    "mean": float(np.mean(recall_list)),
                    "std": float(np.std(recall_list)),
                    "min": float(np.min(recall_list)),
                    "max": float(np.max(recall_list))
                }

        save_data["summary_statistics"] = summary_stats

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"✅ All results saved to: {output_path}")
    return output_path


def batch_analyze_all_groups_multi_ratios(
        base_path_str: str,
        elite_ratios: List[float] = [0.2, 0.3, 0.4],
        save_results: bool = True,
        output_file: str = "all_results_summary.json"
) -> Dict[str, Any]:
    """
    Traverse all numeric folders under base_path, calculate metrics of multiple elite ratios respectively, and finally summarize.

    Parameters:
        base_path_str: Base path
        elite_ratios: Elite ratio list, default is [0.2, 0.3, 0.4]
        save_results: Whether to save all results to JSON file
        output_file: Filename for saving results

    Returns:
        Dictionary containing all results
    """
    base_path = Path(base_path_str)
    if not base_path.exists():
        print(f"❌ Path does not exist: {base_path_str}")
        return {"status": "error", "msg": f"Path does not exist: {base_path_str}"}

    # 1. Scan all numeric folders (1, 2, 3...)
    group_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    if not group_dirs:
        print("❌ No numeric-named group folders found (e.g., '1', '2'...)")
        return {"status": "error", "msg": "No numeric-named group folders found"}

    print(f"📂 Detected {len(group_dirs)} experimental data groups, starting batch analysis...")
    print(f"📊 Elite ratios analyzed: {[f'Top {int(ratio * 100)}%' for ratio in elite_ratios]}\n")

    all_results = []
    valid_results = []

    # 2. Analyze one by one
    for group_dir in group_dirs:
        group_id = group_dir.name
        path_simple = group_dir / "simple"
        path_complete = group_dir / "complete"

        # Check if subfolders exist
        if not (path_simple.exists() and path_complete.exists()):
            print(f"⚠️  Skipping group {group_id}: Missing 'simple' or 'complete' subfolders")
            all_results.append({
                "group_id": group_id,
                "status": "error",
                "msg": "Missing 'simple' or 'complete' subfolders"
            })
            continue

        # Call core logic
        print(f"🔍 Analyzing group {group_id}...")
        res = verify_elite_group_retention_multi_ratios(
            str(path_simple),
            str(path_complete),
            elite_ratios
        )

        if res['status'] == 'error':
            print(f"⚠️  Skipping group {group_id}: {res['msg']}")
            res['group_id'] = group_id
            all_results.append(res)
            continue

        # Success, record result
        res['group_id'] = group_id
        valid_results.append(res)
        all_results.append(res)

        # Print single group brief report
        m = res['metrics']
        recall_strs = []
        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            recall_strs.append(f"Top {int(ratio * 100)}% Recall={m[ratio_key]['recall_rate']:.2f}")

        print(f"✅  Group {group_id} complete: Spearman={m['spearman']:.2f}, {', '.join(recall_strs)}")

    # 3. Print final summary table
    if valid_results:
        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print("=" * 60)

        stats_summary = print_batch_summary_table_multi_ratios(valid_results, elite_ratios)

        # 4. Save all results to file
        if save_results:
            # Determine output path
            if not Path(output_file).is_absolute():
                # If output file is relative path, save under base path
                output_path = base_path / output_file
            else:
                output_path = Path(output_file)

            saved_file = save_all_results_to_json(all_results, str(output_path))

            # Print saved file information
            print(f"💾 All detailed results saved to: {saved_file}")
            print(f"📁 Contains {len(all_results)} groups of data, of which {len(valid_results)} groups were successfully analyzed")

            # Return results containing the saved file path
            return {
                "status": "success",
                "total_groups": len(all_results),
                "successful_groups": len(valid_results),
                "summary_statistics": stats_summary,
                "saved_file": str(saved_file),
                "all_results": all_results
            }
        else:
            return {
                "status": "success",
                "total_groups": len(all_results),
                "successful_groups": len(valid_results),
                "summary_statistics": stats_summary,
                "all_results": all_results
            }
    else:
        print("❌ No valid results produced.")
        return {
            "status": "error",
            "msg": "No valid results produced",
            "total_groups": len(all_results),
            "successful_groups": 0,
            "all_results": all_results
        }


def print_detailed_statistics(all_results: List[Dict]):
    """
    Print detailed statistical data
    """
    successful_results = [r for r in all_results if r['status'] == 'success']

    if not successful_results:
        print("No successfully analyzed data")
        return

    print("\n" + "=" * 60)
    print("Detailed Statistical Data:")
    print("=" * 60)

    # Extract ratio settings of the first group of data
    elite_ratios = successful_results[0]['meta']['elite_ratios']

    # Collect all data
    spearman_values = []
    recall_values = {f"ratio_{int(ratio * 100)}": [] for ratio in elite_ratios}
    sample_counts = []

    for res in successful_results:
        metrics = res['metrics']
        spearman_values.append(metrics['spearman'])
        sample_counts.append(res['meta']['sample_count'])

        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            if ratio_key in metrics:
                recall_values[ratio_key].append(metrics[ratio_key]["recall_rate"])

    # Print Spearman statistics
    print(f"\n📈 Spearman correlation coefficient statistics (Total {len(spearman_values)} groups):")
    print(f"   Average value: {np.mean(spearman_values):.4f}")
    print(f"   Standard deviation: {np.std(spearman_values):.4f}")
    print(f"   Minimum value: {np.min(spearman_values):.4f}")
    print(f"   Maximum value: {np.max(spearman_values):.4f}")

    # Print recall rate statistics for each ratio
    for ratio in elite_ratios:
        ratio_key = f"ratio_{int(ratio * 100)}"
        ratio_name = f"Top {int(ratio * 100)}%"

        if recall_values[ratio_key]:
            print(f"\n🎯 {ratio_name} Recall Rate Statistics:")
            print(
                f"   Average value: {np.mean(recall_values[ratio_key]):.4f} ({np.mean(recall_values[ratio_key]) * 100:.2f}%)")
            print(f"   Standard deviation: {np.std(recall_values[ratio_key]):.4f}")
            print(f"   Minimum value: {np.min(recall_values[ratio_key]):.4f} ({np.min(recall_values[ratio_key]) * 100:.2f}%)")
            print(f"   Maximum value: {np.max(recall_values[ratio_key]):.4f} ({np.max(recall_values[ratio_key]) * 100:.2f}%)")

    # Print sample count statistics
    print(f"\n📊 Sample Count Statistics:")
    print(f"   Average sample count: {np.mean(sample_counts):.1f}")
    print(f"   Sample count range: {np.min(sample_counts)} - {np.max(sample_counts)}")

    print("=" * 60)


def calculate_top_percentage_spearman(lf_rank_list, hf_rank_list, common, top_ratio=0.4):
    # Take top k strategies
    k = max(1, int(len(common) * top_ratio))

    # High-fidelity top k strategies
    hf_top = hf_rank_list[:k]

    # Get positions of these strategies in the two rankings
    s_indices = [lf_rank_list.index(p) for p in hf_top]
    c_indices = [hf_rank_list.index(p) for p in hf_top]

    # Calculate correlation
    corr, _ = stats.spearmanr(s_indices, c_indices)
    return round(corr if not np.isnan(corr) else 0, 4)


if __name__ == '__main__':
    # Configure parameters
    base_result_dir = r"experiment\effectiveness_validation_of_low_granularity_model_screening\passed_validation\result"

    # Define elite ratios to be analyzed
    custom_elite_ratios = [0.2, 0.3, 0.4]  # Top 20%, 30%, 40%

    # Run batch analysis
    result = batch_analyze_all_groups_multi_ratios(
        base_path_str=base_result_dir,
        elite_ratios=custom_elite_ratios,
        save_results=True,
        output_file="all_results_summary.json"
    )

    # Print detailed statistics
    if result['status'] == 'success' and 'all_results' in result:
        print_detailed_statistics(result['all_results'])

    print("\n✨ Analysis complete!")