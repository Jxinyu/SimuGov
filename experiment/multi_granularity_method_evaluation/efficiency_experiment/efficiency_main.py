import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pygmo as pg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def load_experiment_data(base_dir: str) -> Dict[str, Any]:
    """
    Loads the experiment_results.json summary file under the root directory.
    """
    path = Path(base_dir)
    summary_path = path / "experiment_results.json"

    data = {}

    if not summary_path.exists():
        log.error(f"❌ Cannot find summary file: {summary_path}")
        return {}

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)

        # 1. Extract experiment parameters
        data["parameters"] = raw_json.get("parameters", {})

        # 2. Extract Group A data
        # Note: Use get here to avoid errors if the key does not exist, giving a default empty dictionary
        groups = raw_json.get("groups", {})
        data["group_a"] = groups.get("Group_A_Baseline", {})
        data["group_b"] = groups.get("Group_B_Proposed", {})

        # Simple validation
        if not data["group_a"] or not data["group_b"]:
            log.warning("⚠️ Warning: Group A or Group B data is missing in the summary file, the analysis might be incomplete.")

    except Exception as e:
        log.error(f"❌ Failed to read JSON file: {e}")
        return {}

    return data


def calculate_efficiency_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate efficiency metrics"""
    time_a = data["group_a"].get("time_minutes", 0)
    time_b = data["group_b"].get("time_minutes", 0)
    token_a = data["group_a"].get("token_cost", 0)
    token_b = data["group_b"].get("token_cost", 0)

    pop_size = data["parameters"].get("population", 1)
    generations = data["parameters"].get("generations", 1)
    total_evaluations = pop_size * generations

    metrics = {
        "time_a": time_a,
        "time_b": time_b,
        "token_a": token_a,
        "token_b": token_b
    }

    # Calculate metrics
    metrics["speedup"] = time_a / time_b if time_b > 0 else 0.0
    metrics["token_saving"] = (1 - (token_b / token_a)) * 100 if token_a > 0 else 0.0

    if total_evaluations > 0:
        metrics["latency_a"] = (time_a * 60) / total_evaluations
        metrics["latency_b"] = (time_b * 60) / total_evaluations
    else:
        metrics["latency_a"] = 0.0
        metrics["latency_b"] = 0.0

    return metrics


def calculate_quality_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quality metrics"""

    def extract_kpi_points(group_data: Dict) -> List[List[float]]:
        # --- Fix: Compatible with different Key names ---
        elites = group_data.get("elite_solutions") or \
                 group_data.get("elite_solutions_results") or \
                 []

        points = []
        for ind in elites:
            # Prioritize taking the high-granularity verification value complete_kpi
            kpi_dict = ind.get("complete_kpi")
            if not kpi_dict:
                kpi_dict = ind.get("kpi", {})

            if not kpi_dict: continue

            # Convert to positive numbers [0,1]
            s = abs(kpi_dict.get("safety", 0))
            c = abs(kpi_dict.get("creativity", 0))
            sat = abs(kpi_dict.get("satisfaction", 0))
            points.append([s, c, sat])
        return points

    points_a = extract_kpi_points(data.get("group_a", {}))
    points_b = extract_kpi_points(data.get("group_b", {}))

    metrics = {
        "count_a": len(points_a),
        "count_b": len(points_b)
    }

    # Calculate HV (minimize target [-s, -c, -sat], reference point [0,0,0])
    ref_point = [0.0, 0.0, 0.0]

    def compute_hv(points):
        if not points: return 0.0
        neg_points = [[-p[0], -p[1], -p[2]] for p in points]
        return pg.hypervolume(neg_points).compute(ref_point)

    metrics["hv_a"] = compute_hv(points_a)
    metrics["hv_b"] = compute_hv(points_b)
    metrics["hv_ratio"] = metrics["hv_b"] / metrics["hv_a"] if metrics["hv_a"] > 0 else 0.0

    return metrics


def save_analysis_summary_human_readable(efficiency: Dict, quality: Dict, output_dir: str):
    """
    Save analysis report.
    """

    # Construct detailed report structure
    report = {
        "0_Meta_Information": {
            "Generation_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Data_Source": output_dir
        },

        "1_Efficiency_Evaluation": {
            "1.1_Time_Cost_Comparison": {
                "All_High_Group_Time_Cost": f"{efficiency['time_a']:.2f} minutes",
                "High_Low_Group_Time_Cost": f"{efficiency['time_b']:.2f} minutes",
                "Speedup": {
                    "Value": f"{efficiency['speedup']:.2f} x",
                    "Explanation": "How many times faster the experimental group is than the baseline group (the higher the better)"
                }
            },
            "1.2_Computing_Power_Cost_Comparison": {
                "All_High_Group_Tokens": f"{int(efficiency['token_a']):,} Tokens",
                "High_Low_Group_Tokens": f"{int(efficiency['token_b']):,} Tokens",
                "Saving_Rate": {
                    "Value": f"{efficiency['token_saving']:.2f} %",
                    "Explanation": "What percentage of Token cost was saved (the higher the better)"
                }
            },
            "1.3_Scalability_Metrics": {
                "All_High_Group_Single_Strategy_Latency": f"{efficiency['latency_a']:.2f} seconds/strategy",
                "High_Low_Group_Single_Strategy_Latency": f"{efficiency['latency_b']:.2f} seconds/strategy",
                "Explanation": "Average physical time required to evaluate one strategy (excluding experiment scale impact)"
            }
        },

        "2_Quality_Evaluation": {
            "2.1_Solution_Set_Quantity": {
                "All_High_Group_Elite_Count": efficiency.get('elite_count_a', quality['count_a']),  # Compatible writing
                "High_Low_Group_Elite_Count": efficiency.get('elite_count_b', quality['count_b'])
            },
            "2.2_Hypervolume": {
                "All_High_Group_HV": f"{quality['hv_a']:.4f}",
                "High_Low_Group_HV": f"{quality['hv_b']:.4f}",
                "Explanation": "Measure the comprehensive quality of the Pareto frontier (convergence + diversity)"
            },
            "2.3_Consistency_Metrics": {
                "HV_Retention_Rate": {
                    "Value": f"{quality['hv_ratio'] * 100:.2f} %",
                    "Explanation": "How much quality the experimental group reproduced from the baseline group. Close to 100% means perfect, >85% is excellent."
                }
            }
        },

        "3_One_Sentence_Conclusion": (
            f"In this experiment, the high-low granularity coupling framework achieved {efficiency['speedup']:.1f} times time speedup, "
            f"and saved {efficiency['token_saving']:.1f}% of computing power cost. "
            f"The solution set quality retention rate is {quality['hv_ratio'] * 100:.1f}%."
        )
    }

    target_path = Path(output_dir) / f"analysis_summary_{str(datetime.now()).replace(':', '-').replace('.', '-')}.json"
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            # indent=4 makes the format aesthetic
            json.dump(report, f, indent=4, ensure_ascii=False)
        log.info(f"✅ Human-readable report has been saved to: {target_path}")
    except Exception as e:
        log.error(f"❌ Failed to save report: {e}")


def print_analysis_report(
        efficiency: Dict[str, float],
        quality: Dict[str, Any],
        path: str
):
    """
    Prints the formatted analysis report.
    """
    print("\n" + "=" * 60)
    print(f"📊 Framework Efficiency Experiment Analysis Report")
    print(f"📁 Data Source: {path}")
    print("=" * 60)

    print(f"\n[1. Efficiency Evaluation]")
    print(f"--------------------------------------------------")
    print(f"⏱️  Total Time Cost Comparison:")
    print(f"    - Baseline (A): {efficiency['time_a']:.2f} minutes")
    print(f"    - Proposed (B): {efficiency['time_b']:.2f} minutes")
    print(f"🚀 Time Speedup: {efficiency['speedup']:.2f} x")

    print(f"\n💰 Computing Power Consumption Comparison:")
    print(f"    - Baseline (A): {efficiency['token_a']:,} Tokens")
    print(f"    - Proposed (B): {efficiency['token_b']:,} Tokens")
    print(f"📉 Token Saving Rate: {efficiency['token_saving']:.2f} %")

    print(f"\n⚡ Single Strategy Inference Latency:")
    print(f"    - Baseline: {efficiency['latency_a']:.2f} seconds/strategy")
    print(f"    - Proposed: {efficiency['latency_b']:.2f} seconds/strategy")

    print(f"\n[2. Quality Evaluation]")
    print(f"--------------------------------------------------")
    print(f"🎯 Elite Solution Count:")
    print(f"    - Baseline: {quality['count_a']} units")
    print(f"    - Proposed: {quality['count_b']} units")

    print(f"\n📐 Hypervolume:")
    print(f"    - Baseline HV: {quality['hv_a']:.4f}")
    print(f"    - Proposed HV: {quality['hv_b']:.4f}")
    print(f"⚖️  HV Retention Rate: {quality['hv_ratio'] * 100:.2f} %")

    print("=" * 60)
    print("Conclusion and Suggestions:")
    if efficiency['speedup'] > 10 and quality['hv_ratio'] > 0.8:
        print("✅ Experiment successful: Achieved over 10x efficiency improvement while maintaining over 80% solution quality.")
    elif efficiency['speedup'] < 5:
        print("⚠️ Warning: Speedup is not significant, please check if Group B actually followed the low-granularity path.")
    elif quality['hv_ratio'] < 0.5:
        print("⚠️ Warning: Severe loss in solution quality, the low-granularity model might have failed to correctly capture the gradient direction.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # r'result_data\20260107\165500'
    input_path = r'experiment\Multi-granularity method evaluation\Efficiency Experiment\data\155812'
    output_path = r'experiment\Multi-granularity method evaluation\Efficiency Experiment\output'

    if os.path.exists(input_path):
        # 1. Load data
        raw_data = load_experiment_data(input_path)

        # 2. Calculate metrics
        eff_metrics = calculate_efficiency_metrics(raw_data)
        qual_metrics = calculate_quality_metrics(raw_data)

        # 3. Print report (Visible in console)
        print_analysis_report(eff_metrics, qual_metrics, input_path)

        # 4. [New] Save results to file
        save_analysis_summary_human_readable(eff_metrics, qual_metrics, output_path)
    else:
        print(f"❌ Path does not exist: {input_path}")