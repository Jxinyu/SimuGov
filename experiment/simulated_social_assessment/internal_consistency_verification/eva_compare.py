import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Any
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']  # Adaptation for different systems
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def evaluate_extreme_psychology_experiment(
        group_low_rebel_data: Dict[str, List[float]],
        group_high_rebel_data: Dict[str, List[float]],
        output_dir: str
):
    """
    Evaluate results of extreme psychological parameter control experiment (Compliant group vs Rebel group).
    Verify core hypothesis: in high-pressure environments, safety and satisfaction of high-reactance/high-sensitivity group should be significantly lower than low-reactance/low-sensitivity group.

    Args:
        group_low_rebel_data (dict): Simulation data of "low-reactance/compliant group" (Low Beta/Sensitivity).
            Data format requirements:
            {
                "safety": [0.99, 0.98, 0.99, ...],       # List[float]: Daily safety KPI
                "satisfaction": [0.6, 0.55, 0.58, ...],  # List[float]: Daily satisfaction KPI
            }

        group_high_rebel_data (dict): Simulation data of "high-reactance/rebel group" (High Beta/Sensitivity).
            Data format requirements same as above.

        output_dir (str): Save path for charts and reports.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results.
    """

    # 1. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Extract core metrics
    low_safety = np.array(group_low_rebel_data['safety'])
    high_safety = np.array(group_high_rebel_data['safety'])

    low_sat = np.array(group_low_rebel_data['satisfaction'])
    high_sat = np.array(group_high_rebel_data['satisfaction'])

    days = range(1, len(low_safety) + 1)

    # 3. Calculate statistics (mean)
    mean_safe_low = np.mean(low_safety)
    mean_safe_high = np.mean(high_safety)

    mean_sat_low = np.mean(low_sat)
    mean_sat_high = np.mean(high_sat)

    # Difference value (Low - High), expected to be positive
    safety_diff = mean_safe_low - mean_safe_high
    sat_diff = mean_sat_low - mean_sat_high

    # 4. Hypothesis testing (T-test) - optional, used to enhance scientific rigor
    # Test if there is a significant difference between the means of the two groups
    t_stat_safe, p_val_safe = stats.ttest_ind(low_safety, high_safety, equal_var=False)
    t_stat_sat, p_val_sat = stats.ttest_ind(low_sat, high_sat, equal_var=False)

    # 5. Decision logic
    # Passing criteria:
    # 1. Safety of low-reactance group is significantly higher than high-reactance group (diff > 0.05 and p < 0.05)
    # 2. Satisfaction of low-reactance group is significantly higher than high-reactance group (because high-sensitivity group will be angry under high pressure)

    is_safety_consistent = (safety_diff > 0.05) and (p_val_safe < 0.05)
    is_sat_consistent = (sat_diff > 0.05) and (p_val_sat < 0.05)

    is_passed = is_safety_consistent and is_sat_consistent

    # 6. Draw comparison chart (modified to 3 rows and 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), sharex=True)

    # Subplot 1: Safety comparison
    ax1.plot(days, low_safety, 'b-o', label=f'Low-reactance group (Mean={mean_safe_low:.2f})', linewidth=2)
    ax1.plot(days, high_safety, 'r-^', label=f'High-reactance group (Mean={mean_safe_high:.2f})', linewidth=2)
    ax1.set_title('Safety comparison validation', fontsize=14)
    ax1.set_ylabel('KPI Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Annotate difference
    ax1.text(0.06, 0.03, f'Difference: {safety_diff:.2f}\nP-value: {p_val_safe:.4f}',
             transform=ax1.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Subplot 2: Satisfaction comparison
    ax2.plot(days, low_sat, 'b-o', label=f'Low-sensitivity group (Mean={mean_sat_low:.2f})', linewidth=2)
    ax2.plot(days, high_sat, 'r-^', label=f'High-sensitivity group (Mean={mean_sat_high:.2f})', linewidth=2)
    ax2.set_title('Satisfaction comparison validation', fontsize=14)
    ax2.set_ylabel('KPI Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    # Annotate difference
    ax2.text(0.06, 0.03, f'Difference: {sat_diff:.2f}\nP-value: {p_val_sat:.4f}',
             transform=ax2.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(f'High-Low reactance group comparison validation: {"Passed" if is_passed else "Failed"}', fontsize=16, fontweight='bold',
                 color='green' if is_passed else 'red')

    save_path = os.path.join(output_dir, 'consistency_verification_result.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Print console report
    print("\n" + "=" * 50)
    print(" >>> Internal Consistency Report <<<")
    print("=" * 50)
    print(f"Validation status: {'✅ Passed' if is_passed else '❌ Failed'}")
    print("-" * 30)
    print(f"[Safety]")
    print(f"  - Compliant group mean: {mean_safe_low:.4f}")
    print(f"  - Rebel group mean: {mean_safe_high:.4f}")
    print(f"  - Difference (Expected>0): {safety_diff:.4f}")
    print(f"  - Statistical significance (p-value): {p_val_safe:.4e}")
    print("-" * 30)
    print(f"[Satisfaction]")
    print(f"  - Compliant group mean: {mean_sat_low:.4f}")
    print(f"  - Rebel group mean: {mean_sat_high:.4f}")
    print(f"  - Difference (Expected>0): {sat_diff:.4f}")
    print(f"  - Statistical significance (p-value): {p_val_sat:.4e}")
    print("=" * 50 + "\n")

    save_path = os.path.join(output_dir, 'result.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "is_passed (Whether consistency validation passed)": str(is_passed),
                "safety_diff (Safety mean difference (Low - High))": float(safety_diff),
                "satisfaction_diff (Satisfaction mean difference (Low - High))": float(sat_diff),
                "p_value_safety": float(p_val_safe),
                "p_value_satisfaction": float(p_val_sat),
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Results saved at {save_path}")


def construct_data(file_path_1, file_path_2, output_dir, day_time):
    """
    Construct data
    file_path_1: experiment/simulation_social_assessment/internal_consistency_validation/data/12-28-Policy3-Time20/Punishment0_01_LowEducation_ai_threshold_0_99
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
    Compare
    :param low_beta_file: r'experiment/simulation_social_assessment/internal_consistency_validation/data/12-28-Policy3-Time20/Punishment0_01_LowEducation_ai_threshold_0_99'
    :param high_beta_file: r'experiment/simulation_social_assessment/internal_consistency_validation/data/12-28-Policy3-Time20/Punishment0_99_HighEducation_ai_threshold_0_01'
    :param output_dir: r'experiment/simulation_social_assessment/internal_consistency_validation/output'
    :param day_time:
    :return:
    """
    output_dir += '/low_high'
    try:
        os.makedirs(output_dir)
    except:
        pass
    construct_data(low_beta_file, high_beta_file, output_dir, day_time)