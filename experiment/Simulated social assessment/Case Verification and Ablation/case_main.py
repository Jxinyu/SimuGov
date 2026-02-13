import json
import os
import math
from datetime import datetime
from typing import List

import numpy as np
from case_validator import CaseValidator


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


def build_simulation_data_strict_window(input_file_root, end_day, capacity=30, lookback_days=3):
    """
    模拟真实的社交平台热榜机制：
    1. 容量有限 (Top K)
    2. 时效性强 (Lookback Window) - 超过 N 天的内容强制过期，无法上首页。

    :param lookback_days: 回溯天数。建议设为 2 或 3。这意味着3天前的内容自动从首页消失。
    """
    target_dir = os.path.join(input_file_root, f"day_time_{end_day}")
    content_path = os.path.join(target_dir, "output_contents.json")
    persona_path = os.path.join(target_dir, "output_personas.json")

    if not os.path.exists(content_path):
        raise FileNotFoundError(f"找不到数据文件: {content_path}")

    print(f"📖 读取数据: {target_dir}")
    with open(content_path, 'r', encoding='utf-8') as f:
        all_contents = json.load(f)
    with open(persona_path, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)

    sim_protest_ratios = []
    sim_satisfaction = []

    print(f"🔄 计算首页: 容量={capacity}, 有效期=最近{lookback_days}天...")

    # === 遍历每一天 ===
    for day in range(1, end_day + 1):
        window_start = max(1, day - lookback_days + 1)

        valid_pool = [
            c for c in all_contents
            if window_start <= c['time'] <= day
        ]

        # 按时间倒序排列 (最新的在最上面)
        # 兜底排序：如果时间相同，按ID倒序
        valid_pool.sort(key=lambda x: (x['time'], str(x['id'])), reverse=True)

        # 截取首页容量 (模拟用户能看到的范围)
        homepage = valid_pool[:capacity]

        # 统计抗议内容
        protest_count = 0
        for c in homepage:
            topic = str(c.get('topic', '')).upper()
            detail = str(c.get('content_detail', '')).upper()

            # 判定标准
            if 'NO AI' in topic or 'PROTEST' in topic or 'NO AI' in detail:
                protest_count += 1

        # 计算占比
        # 分母：实际首页展示数 (初期可能小于 capacity)
        denominator = len(homepage)
        ratio = protest_count / denominator if denominator > 0 else 0.0
        sim_protest_ratios.append(ratio)

        # -----------------------------------
        # 2. 满意度计算
        # -----------------------------------
        sat_sum = 0
        count = 0
        for p in all_personas:
            if p['type'] == '合规创作者':
                history = p.get('satisfaction', [])
                idx = day - 1
                if 0 <= idx < len(history):
                    sat_sum += history[idx]
                    count += 1

        avg_sat = sat_sum / count if count > 0 else 0
        sim_satisfaction.append(avg_sat)

    return sim_protest_ratios, sim_satisfaction


def load_multi_run_data(run_dirs: List[str], end_day=30, capacity=40, lookback_days=3):
    """
    加载多个运行文件夹的数据，返回矩阵。
    """
    all_runs_ratios = []
    all_runs_sats = []

    print(f"🚀 开始聚合 {len(run_dirs)} 次运行结果...")

    for i, root_dir in enumerate(run_dirs):
        print(f"   Reading Run #{i + 1}: {root_dir}")
        try:
            # 复用之前的 build_simulation_data_strict_window 函数
            ratios, sats = build_simulation_data_strict_window(
                root_dir, end_day, capacity, lookback_days
            )
            all_runs_ratios.append(ratios)
            all_runs_sats.append(sats)
        except Exception as e:
            print(f"   ⚠️ Run #{i + 1} 加载失败: {e}")

    # 转换为 Numpy 矩阵 [N_runs, N_days]
    # 注意：确保所有 runs 的天数长度一致，如果不一致需要截断
    min_len = min([len(r) for r in all_runs_ratios])

    ratio_matrix = np.array([r[:min_len] for r in all_runs_ratios])
    sat_matrix = np.array([s[:min_len] for s in all_runs_sats])

    return ratio_matrix, sat_matrix


def plot_confidence_interval(run_directories):
    """
    绘制多组运行结果的置信区间图
    :return:
    """
    ground_truth_ratios = [
        0, 0, 0, 0, 0, 0, 0,
        0.10, 0.09, 0.28, 0.30, 0.34,
        0.35, 0.33, 0.42, 0.42, 0.31,
        0.27, 0.13, 0.04, 0.04, 0.03,
        0.01, 0.01, 0.01, 0, 0.01, 0, 0, 0
    ]

    try:
        # 加载数据矩阵
        ratio_matrix, sat_matrix = load_multi_run_data(
            run_directories,
            end_day=30,
            capacity=40,
            lookback_days=3
        )

        # 计算均值曲线用于指标验证
        sim_mean_curve = np.mean(ratio_matrix, axis=0).tolist()
        sat_mean_curve = np.mean(sat_matrix, axis=0).tolist()

        # 数据对齐
        min_len = min(len(ground_truth_ratios), len(sim_mean_curve))
        gt_aligned = ground_truth_ratios[:min_len]
        sim_mean_aligned = sim_mean_curve[:min_len]

        # --- 验证 (使用均值曲线进行打分) ---
        print("\n" + "=" * 40)
        print(f"🔬 多次运行聚合验证报告 (N={len(run_directories)})")
        print("=" * 40)

        # 1. 趋势验证 (对比均值)
        trend_res = CaseValidator.validate_trend_correlation(gt_aligned, sim_mean_aligned)
        print(f"📈 [均值趋势] Pearson: {trend_res['pearson']['value']} | Spearman: {trend_res['spearman']['value']}")

        # 2. 鲁棒性验证 (新增：计算标准差的平均值)
        avg_std = np.mean(np.std(ratio_matrix, axis=0))
        print(f"🛡️ [鲁棒性] 平均标准差 (Avg STD): {avg_std:.4f} (越低越稳定)")

        # 3. 绘图 (带置信区间)
        save_path = CaseValidator.plot_trend_with_ci(
            gt_aligned,
            ratio_matrix,  # 传入矩阵
            title=f"ArtStation Simulation Robustness (N={len(run_directories)})"
        )
        print(f"🖼️ 置信区间图已保存: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def analysis_single_data(input_file_root):
    """
    分析单次仿真运行结果
    :return:
    """
    # 1. 真实历史数据
    ground_truth_ratios = [
        0, 0, 0, 0, 0, 0, 0,
        0.10, 0.09, 0.28, 0.30, 0.34,
        0.35, 0.33, 0.42, 0.42, 0.31,
        0.27, 0.13, 0.04, 0.04, 0.03,
        0.01, 0.01, 0.01, 0, 0.01, 0, 0, 0
    ]

    # 3. 参数
    HOMEPAGE_CAPACITY = 40
    CONTENT_EXPIRATION_DAYS = 3

    try:
        sim_ratios, sim_sat = build_simulation_data_strict_window(
            input_file_root,
            end_day=30,
            capacity=HOMEPAGE_CAPACITY,
            lookback_days=CONTENT_EXPIRATION_DAYS
        )

        # 数据对齐
        min_len = min(len(ground_truth_ratios), len(sim_ratios))
        gt_aligned = ground_truth_ratios[:min_len]
        sim_aligned = sim_ratios[:min_len]
        sat_aligned = sim_sat[:min_len]

        # --- 输出报告 ---
        print("\n" + "=" * 40)
        print("🔬 案例验证报告 (时效性窗口模型)")
        print("=" * 40)

        # 这里 CaseValidator 内部已经打印了详细的带评级的报告
        trend_res = CaseValidator.validate_trend_correlation(gt_aligned, sim_aligned)
        mech_res = CaseValidator.validate_mechanism_causality(sat_aligned, sim_aligned)
        peak_res = CaseValidator.validate_peak_alignment(gt_aligned, sim_aligned)

        # --- 【修复点】下面这几行简单的汇总打印，键名需要更新 ---
        print(f"\n📝 简要汇总:")
        print(f"📈 [趋势拟合] Pearson: {trend_res['pearson']['value']} | Spearman: {trend_res['spearman']['value']}")
        print(f"🏔️ [时序同步] 滞后: {peak_res['lag_days']} 天 (Sim Peak: Day {peak_res['sim_peak_day']})")
        print(f"⚙️ [机制验证] 相关性: {mech_res['correlation']}")

        # 绘图
        output_dir = r'experiment\仿真社会评估\案例验证\output'
        save_path = CaseValidator.plot_trend_comparison(
            gt_aligned,
            sim_aligned,
            title=f"ArtStation验证: 首页占比 (Top{HOMEPAGE_CAPACITY})",
            output_dir=output_dir
        )
        print(f"图表已保存: {save_path}")

        # 保存JSON
        res_data = {
            "metrics": {"trend": trend_res, "peak": peak_res, "mechanism": mech_res},
            "data": {"ground_truth": gt_aligned, "simulation": sim_aligned, "satisfaction": sat_aligned}
        }
        json_path = os.path.join(output_dir, f"validation_report_{datetime.now().strftime('%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(res_data, f, ensure_ascii=False, indent=4, cls=NpEncoder)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    # 绘制置信区间
    # 2. 多次运行的目录列表
    # directories = [
    #     r'experiment\Simulated social assessment\Case Verification and Ablation\data\通过\155658\惩罚0_01_教育低_ai_threshold_0_8',
    #     r'experiment\Simulated social assessment\Case Verification and Ablation\data\t7\case_validation',
    # ]
    # plot_confidence_interval(directories)
    # 分析单次运行结果
    # 2. 输入路径
    input_file = r'experiment\仿真社会评估\案例验证\data\开启心理参数\case_validation'
    analysis_single_data(input_file)














