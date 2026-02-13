import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Any
from scipy import stats

# 设置绘图风格和中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 适配不同系统
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def evaluate_extreme_psychology_experiment(
        group_low_rebel_data: Dict[str, List[float]],
        group_high_rebel_data: Dict[str, List[float]],
        output_dir: str
):
    """
    评估心理参数极端对照实验的结果（顺从组 vs 反叛组）。
    验证核心假设：在高压环境下，高逆反/高敏感组的安全性与满意度应显著低于低逆反/低敏感组。

    Args:
        group_low_rebel_data (dict): "低逆反/顺从组" (Low Beta/Sensitivity) 的仿真数据。
            数据格式要求:
            {
                "safety": [0.99, 0.98, 0.99, ...],       # List[float]: 每日安全性KPI
                "satisfaction": [0.6, 0.55, 0.58, ...],  # List[float]: 每日满意度KPI
            }

        group_high_rebel_data (dict): "高逆反/反叛组" (High Beta/Sensitivity) 的仿真数据。
            数据格式要求同上。

        output_dir (str): 图表和报告的保存路径。

    Returns:
        Dict[str, Any]: 包含评估结果的字典。
    """

    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 提取核心指标
    low_safety = np.array(group_low_rebel_data['safety'])
    high_safety = np.array(group_high_rebel_data['safety'])

    low_sat = np.array(group_low_rebel_data['satisfaction'])
    high_sat = np.array(group_high_rebel_data['satisfaction'])

    days = range(1, len(low_safety) + 1)

    # 3. 计算统计量 (均值)
    mean_safe_low = np.mean(low_safety)
    mean_safe_high = np.mean(high_safety)

    mean_sat_low = np.mean(low_sat)
    mean_sat_high = np.mean(high_sat)

    # 差异值 (Low - High)，预期应为正数
    safety_diff = mean_safe_low - mean_safe_high
    sat_diff = mean_sat_low - mean_sat_high

    # 4. 假设检验 (T-test) - 可选，用于增强科学性
    # 检验两组数据的均值是否存在显著差异
    t_stat_safe, p_val_safe = stats.ttest_ind(low_safety, high_safety, equal_var=False)
    t_stat_sat, p_val_sat = stats.ttest_ind(low_sat, high_sat, equal_var=False)

    # 5. 判定逻辑
    # 通过标准：
    # 1. 低逆反组的安全性 显著高于 高逆反组 (diff > 0.05 且 p < 0.05)
    # 2. 低逆反组的满意度 显著高于 高逆反组 (因为高压下高敏感组会愤怒)

    is_safety_consistent = (safety_diff > 0.05) and (p_val_safe < 0.05)
    is_sat_consistent = (sat_diff > 0.05) and (p_val_sat < 0.05)

    is_passed = is_safety_consistent and is_sat_consistent

    # 6. 绘制对比图 (修改为 3 行 1 列)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), sharex=True)

    # 子图1：安全性对比
    ax1.plot(days, low_safety, 'b-o', label=f'低逆反组 (均值={mean_safe_low:.2f})', linewidth=2)
    ax1.plot(days, high_safety, 'r-^', label=f'高逆反组 (均值={mean_safe_high:.2f})', linewidth=2)
    ax1.set_title('Safety 对比验证', fontsize=14)
    ax1.set_ylabel('KPI 值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    # 标注差异
    ax1.text(0.06, 0.03, f'差异: {safety_diff:.2f}\nP值: {p_val_safe:.4f}',
             transform=ax1.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # 子图2：满意度对比
    ax2.plot(days, low_sat, 'b-o', label=f'低敏感组 (均值={mean_sat_low:.2f})', linewidth=2)
    ax2.plot(days, high_sat, 'r-^', label=f'高敏感组 (均值={mean_sat_high:.2f})', linewidth=2)
    ax2.set_title('Satisfaction 对比验证', fontsize=14)
    ax2.set_ylabel('KPI 值')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    # 标注差异
    ax2.text(0.06, 0.03, f'差异: {sat_diff:.2f}\nP值: {p_val_sat:.4f}',
             transform=ax2.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(f'高低逆反组对比验证: {"通过" if is_passed else "未通过"}', fontsize=16, fontweight='bold',
                 color='green' if is_passed else 'red')

    save_path = os.path.join(output_dir, 'consistency_verification_result.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 打印控制台报告
    print("\n" + "=" * 50)
    print(" >>> 内部一致性验证报告 (Internal Consistency Report) <<<")
    print("=" * 50)
    print(f"验证状态: {'✅ 通过' if is_passed else '❌ 失败'}")
    print("-" * 30)
    print(f"[安全性 Safety]")
    print(f"  - 顺从组均值: {mean_safe_low:.4f}")
    print(f"  - 反叛组均值: {mean_safe_high:.4f}")
    print(f"  - 差异 (预期>0): {safety_diff:.4f}")
    print(f"  - 统计显著性 (p-value): {p_val_safe:.4e}")
    print("-" * 30)
    print(f"[满意度 Satisfaction]")
    print(f"  - 顺从组均值: {mean_sat_low:.4f}")
    print(f"  - 反叛组均值: {mean_sat_high:.4f}")
    print(f"  - 差异 (预期>0): {sat_diff:.4f}")
    print(f"  - 统计显著性 (p-value): {p_val_sat:.4e}")
    print("=" * 50 + "\n")

    save_path = os.path.join(output_dir, 'result.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "is_passed (是否通过一致性验证)": str(is_passed),
                "safety_diff ( 安全性均值差 (Low - High))": float(safety_diff),
                "satisfaction_diff (满意度均值差 (Low - High))": float(sat_diff),
                "p_value_safety": float(p_val_safe),
                "p_value_satisfaction": float(p_val_sat),
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"结果保存在 {save_path}")


def construct_data(file_path_1, file_path_2, output_dir, day_time):
    """
    构造数据
    file_path_1: experiment\仿真社会评估\内部一致性验证\data\12-28-政策3-时间20\惩罚0_01_教育低_ai_threshold_0_99
    :return:
    """
    with open(file_path_1 + f'/day_time_{day_time}/output_system_kpi.json', 'r', encoding='utf-8') as f:
        data_1 = json.load(f)
    with open(file_path_2 + f'/day_time_{day_time}/output_system_kpi.json', 'r', encoding='utf-8') as f:
        data_2 = json.load(f)
    low_data = {
        'safety': data_1['safety'],
        'satisfaction': data_1['satisfaction'],
        'creativity': data_1['creativity'],
    }
    high_data = {
        'safety': data_2['safety'],
        'satisfaction': data_2['satisfaction'],
        'creativity': data_2['creativity'],
    }
    evaluate_extreme_psychology_experiment(low_data, high_data, output_dir)


def eva_compare(low_beta_file, high_beta_file, output_dir, day_time):
    """
    对比
    :param low_beta_file: r'experiment\仿真社会评估\内部一致性验证\data\12-28-政策3-时间20\惩罚0_01_教育低_ai_threshold_0_99'
    :param high_beta_file: r'experiment\仿真社会评估\内部一致性验证\data\12-28-政策3-时间20\惩罚0_99_教育高_ai_threshold_0_01'
    :param output_dir: r'experiment\仿真社会评估\内部一致性验证\output'
    :param day_time:
    :return:
    """
    output_dir += '/low_high'
    try:
        os.makedirs(output_dir)
    except:
        pass
    construct_data(low_beta_file, high_beta_file, output_dir, day_time)
