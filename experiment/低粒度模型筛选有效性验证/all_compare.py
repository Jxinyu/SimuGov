import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import scipy.stats as stats
import numpy as np

# 配置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================================
# 1. 核心计算逻辑 (保持不变)
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

            # 格式化 Key 以便对齐
            sig = f"edu_{pol.get('e_edu')}_ai_{float(pol.get('ai_threshold', 0)):.2f}_f_{float(pol.get('f_penalty', 0)):.2f}"
            results[sig] = compute_final_metrics(kpi)
        except:
            continue
    return results


# =========================================================================
# 2. 排名与验证逻辑 (保持不变)
# =========================================================================

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
    同时计算多个精英比例的结果
    """
    # 1. 提取与对齐
    s_metrics = _extract_metrics_from_root(simple_data_path)
    c_metrics = _extract_metrics_from_root(complete_data_path)
    common = set(s_metrics.keys()) & set(c_metrics.keys())
    count = len(common)

    if count < 3:
        return {"status": "error", "msg": f"样本太少(n={count})"}

    # 2. 计算排名
    s_data = {k: v for k, v in s_metrics.items() if k in common}
    c_data = {k: v for k, v in c_metrics.items() if k in common}

    lf_rank_list = _calculate_pareto_rank(s_data)
    hf_rank_list = _calculate_pareto_rank(c_data)

    # 计算Spearman相关系数（只需要计算一次，与精英比例无关）
    s_indices = [lf_rank_list.index(p) for p in common]
    c_indices = [hf_rank_list.index(p) for p in common]
    corr, _ = stats.spearmanr(s_indices, c_indices)
    spearman_value = round(corr if not np.isnan(corr) else 0, 4)
    top40_spearman = calculate_top_percentage_spearman(lf_rank_list, hf_rank_list, common, 0.4)
    print(f"Spearman: {spearman_value}, Top40 Spearman: {top40_spearman}")

    # 3. 分析多个精英比例
    all_ratios_results = {}
    elite_analysis_dict = {}

    for ratio in elite_ratios:
        # 精英定义
        k_elites = max(1, int(count * ratio))
        true_elites = hf_rank_list[:k_elites]

        # 分析
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

        # 保存该比例的结果
        ratio_key = f"ratio_{int(ratio * 100)}"
        all_ratios_results[ratio_key] = {
            "recall_rate": round(recall_rate, 4),
            "elite_count": k_elites,
            "sample_count": count
        }

        # 保存详细分析
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


# =========================================================================
# 3. 批量处理与报告生成 (修改部分)
# =========================================================================

def print_batch_summary_table_multi_ratios(all_results: List[Dict], elite_ratios: List[float]):
    """
    打印所有组的汇总表格，支持多个精英比例
    """
    print("\n" + "=" * 100)

    # 生成标题
    ratio_strs = [f"Top {int(ratio * 100)}%" for ratio in elite_ratios]
    title = f"【批量实验汇总报告 - 多精英比例分析】"
    print(f"{title:^90}")
    print("=" * 100)

    # 表头
    header = f"{'Group ID':<10} | {'样本数':<8} | {'Spearman':<10} "
    for ratio_str in ratio_strs:
        header += f"| {ratio_str + ' Recall':<12} "
    print(header)
    print("-" * 100)

    # 初始化统计列表
    spearman_list = []
    recall_lists = {f"ratio_{int(ratio * 100)}": [] for ratio in elite_ratios}

    # 打印每一组的结果
    for res in all_results:
        if res['status'] != 'success':
            continue

        gid = res['group_id']
        metrics = res['metrics']

        # 提取数据
        sample_count = res['meta']['sample_count']
        spearman_val = metrics['spearman']

        # 收集数据用于平均计算
        spearman_list.append(spearman_val)

        # 格式化输出行
        row = f"{gid:<10} | {sample_count:<8} | {spearman_val:<10.4f} "

        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            recall_val = metrics[ratio_key]["recall_rate"]
            recall_lists[ratio_key].append(recall_val)
            row += f"| {recall_val * 100:>10.2f}%  "

        print(row)

    print("-" * 100)

    # 计算并打印平均值
    if spearman_list:
        avg_spearman = np.mean(spearman_list)

        # 计算每个比例的平均召回率
        avg_recalls = {}
        for ratio_key, recall_list in recall_lists.items():
            if recall_list:
                avg_recalls[ratio_key] = np.mean(recall_list)

        # 打印平均行
        avg_row = f"{'AVERAGE':<10} | {'-':<8} | {avg_spearman:<10.4f} "
        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            avg_recall = avg_recalls.get(ratio_key, 0)
            avg_row += f"| {avg_recall * 100:>10.2f}%  "

        print(avg_row)

    print("=" * 100 + "\n")

    # 返回统计数据用于保存
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
    将所有结果保存到JSON文件
    """
    # 准备保存的数据
    save_data = {
        "timestamp": np.datetime64('now').astype(str),
        "total_groups": len(all_results),
        "successful_groups": len([r for r in all_results if r['status'] == 'success']),
        "all_results": all_results,
        "summary_statistics": {}
    }

    # 计算汇总统计（如果所有组都成功）
    successful_results = [r for r in all_results if r['status'] == 'success']
    if successful_results:
        # 提取所有比例
        elite_ratios = successful_results[0]['meta']['elite_ratios']

        # 初始化统计字典
        spearman_list = []
        recall_stats = {f"ratio_{int(ratio * 100)}": [] for ratio in elite_ratios}

        # 收集所有数据
        for res in successful_results:
            metrics = res['metrics']
            spearman_list.append(metrics['spearman'])

            for ratio in elite_ratios:
                ratio_key = f"ratio_{int(ratio * 100)}"
                if ratio_key in metrics:
                    recall_stats[ratio_key].append(metrics[ratio_key]["recall_rate"])

        # 计算统计量
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

    # 保存到文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 所有结果已保存到: {output_path}")
    return output_path


def batch_analyze_all_groups_multi_ratios(
        base_path_str: str,
        elite_ratios: List[float] = [0.2, 0.3, 0.4],
        save_results: bool = True,
        output_file: str = "all_results_summary.json"
) -> Dict[str, Any]:
    """
    遍历 base_path 下所有的数字文件夹，分别计算多个精英比例的指标，最后汇总。

    参数:
        base_path_str: 基础路径
        elite_ratios: 精英比例列表，默认为[0.2, 0.3, 0.4]
        save_results: 是否保存所有结果到JSON文件
        output_file: 结果保存的文件名

    返回:
        包含所有结果的字典
    """
    base_path = Path(base_path_str)
    if not base_path.exists():
        print(f"❌ 路径不存在: {base_path_str}")
        return {"status": "error", "msg": f"路径不存在: {base_path_str}"}

    # 1. 扫描所有数字文件夹 (1, 2, 3...)
    group_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    if not group_dirs:
        print("❌ 未找到任何以数字命名的分组文件夹 (例如 '1', '2'...)")
        return {"status": "error", "msg": "未找到任何以数字命名的分组文件夹"}

    print(f"📂 检测到 {len(group_dirs)} 组实验数据，开始批量分析...")
    print(f"📊 分析的精英比例: {[f'Top {int(ratio * 100)}%' for ratio in elite_ratios]}\n")

    all_results = []
    valid_results = []

    # 2. 逐个分析
    for group_dir in group_dirs:
        group_id = group_dir.name
        path_simple = group_dir / "simple"
        path_complete = group_dir / "complete"

        # 检查子文件夹是否存在
        if not (path_simple.exists() and path_complete.exists()):
            print(f"⚠️  跳过组 {group_id}: 缺少 'simple' 或 'complete' 子文件夹")
            all_results.append({
                "group_id": group_id,
                "status": "error",
                "msg": "缺少 'simple' 或 'complete' 子文件夹"
            })
            continue

        # 调用核心逻辑
        print(f"🔍 正在分析组 {group_id}...")
        res = verify_elite_group_retention_multi_ratios(
            str(path_simple),
            str(path_complete),
            elite_ratios
        )

        if res['status'] == 'error':
            print(f"⚠️  跳过组 {group_id}: {res['msg']}")
            res['group_id'] = group_id
            all_results.append(res)
            continue

        # 成功，记录结果
        res['group_id'] = group_id
        valid_results.append(res)
        all_results.append(res)

        # 打印单组简报
        m = res['metrics']
        recall_strs = []
        for ratio in elite_ratios:
            ratio_key = f"ratio_{int(ratio * 100)}"
            recall_strs.append(f"Top {int(ratio * 100)}% Recall={m[ratio_key]['recall_rate']:.2f}")

        print(f"✅  组 {group_id} 完成: Spearman={m['spearman']:.2f}, {', '.join(recall_strs)}")

    # 3. 打印最终汇总表格
    if valid_results:
        print("\n" + "=" * 60)
        print("汇总统计:")
        print("=" * 60)

        stats_summary = print_batch_summary_table_multi_ratios(valid_results, elite_ratios)

        # 4. 保存所有结果到文件
        if save_results:
            # 确定输出路径
            if not Path(output_file).is_absolute():
                # 如果输出文件是相对路径，则保存在基础路径下
                output_path = base_path / output_file
            else:
                output_path = Path(output_file)

            saved_file = save_all_results_to_json(all_results, str(output_path))

            # 打印保存的文件信息
            print(f"💾 所有详细结果已保存到: {saved_file}")
            print(f"📁 包含 {len(all_results)} 组数据，其中 {len(valid_results)} 组分析成功")

            # 返回包含保存文件路径的结果
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
        print("❌ 没有产生任何有效结果。")
        return {
            "status": "error",
            "msg": "没有产生任何有效结果",
            "total_groups": len(all_results),
            "successful_groups": 0,
            "all_results": all_results
        }


def print_detailed_statistics(all_results: List[Dict]):
    """
    打印详细的统计数据
    """
    successful_results = [r for r in all_results if r['status'] == 'success']

    if not successful_results:
        print("没有成功分析的数据")
        return

    print("\n" + "=" * 60)
    print("详细统计数据:")
    print("=" * 60)

    # 提取第一组数据的比例设置
    elite_ratios = successful_results[0]['meta']['elite_ratios']

    # 收集所有数据
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

    # 打印Spearman统计
    print(f"\n📈 Spearman相关系数统计 (共{len(spearman_values)}组):")
    print(f"   平均值: {np.mean(spearman_values):.4f}")
    print(f"   标准差: {np.std(spearman_values):.4f}")
    print(f"   最小值: {np.min(spearman_values):.4f}")
    print(f"   最大值: {np.max(spearman_values):.4f}")

    # 打印每个比例的召回率统计
    for ratio in elite_ratios:
        ratio_key = f"ratio_{int(ratio * 100)}"
        ratio_name = f"Top {int(ratio * 100)}%"

        if recall_values[ratio_key]:
            print(f"\n🎯 {ratio_name} 召回率统计:")
            print(
                f"   平均值: {np.mean(recall_values[ratio_key]):.4f} ({np.mean(recall_values[ratio_key]) * 100:.2f}%)")
            print(f"   标准差: {np.std(recall_values[ratio_key]):.4f}")
            print(f"   最小值: {np.min(recall_values[ratio_key]):.4f} ({np.min(recall_values[ratio_key]) * 100:.2f}%)")
            print(f"   最大值: {np.max(recall_values[ratio_key]):.4f} ({np.max(recall_values[ratio_key]) * 100:.2f}%)")

    # 打印样本数统计
    print(f"\n📊 样本数统计:")
    print(f"   平均样本数: {np.mean(sample_counts):.1f}")
    print(f"   样本数范围: {np.min(sample_counts)} - {np.max(sample_counts)}")

    print("=" * 60)


def calculate_top_percentage_spearman(lf_rank_list, hf_rank_list, common, top_ratio=0.4):
    # 取前k个策略
    k = max(1, int(len(common) * top_ratio))

    # 高保真前k个策略
    hf_top = hf_rank_list[:k]

    # 获取这些策略在两个排名中的位置
    s_indices = [lf_rank_list.index(p) for p in hf_top]
    c_indices = [hf_rank_list.index(p) for p in hf_top]

    # 计算相关性
    corr, _ = stats.spearmanr(s_indices, c_indices)
    return round(corr if not np.isnan(corr) else 0, 4)


if __name__ == '__main__':
    # 配置参数
    base_result_dir = r"experiment\低粒度模型筛选有效性验证\验证通过\result"

    # 定义要分析的精英比例
    custom_elite_ratios = [0.2, 0.3, 0.4]  # Top 20%, 30%, 40%

    # 运行批量分析
    result = batch_analyze_all_groups_multi_ratios(
        base_path_str=base_result_dir,
        elite_ratios=custom_elite_ratios,
        save_results=True,
        output_file="all_results_summary.json"
    )

    # 打印详细统计
    if result['status'] == 'success' and 'all_results' in result:
        print_detailed_statistics(result['all_results'])

    print("\n✨ 分析完成！")