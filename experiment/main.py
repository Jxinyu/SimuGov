import os
import sys
import json
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.stats as stats
import pygmo as pg

EXP_DIR = Path(__file__).resolve().parent


def auto_append_paths(root_path):
    """Recursively add subdirectories to sys.path for internal calls."""
    for root, dirs, files in os.walk(root_path):
        if "__pycache__" in root or ".git" in root: continue
        if root not in sys.path: sys.path.append(root)


auto_append_paths(str(EXP_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimuGov-Aggregator")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray, np.bool_)):
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj.item()
        if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records')
        return super(NpEncoder, self).default(obj)


# Translation map for converting internal labels to English
TRANSLATION_MAP = {
    "安全性": "Safety", "创造力": "Creativity", "满意度": "Satisfaction",
    "惩罚": "Penalty", "教育": "Education",
    "ratio_20": "Recall_Top20", "ratio_30": "Recall_Top30", "ratio_40": "Recall_Top40"
}


def translate(obj):
    if isinstance(obj, dict):
        return {TRANSLATION_MAP.get(k, k): translate(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate(i) for i in obj]
    return obj


def find_file_recursive(base_path: Path, filename: str) -> Path:
    if not base_path or not base_path.exists(): return None
    matches = list(base_path.rglob(filename))
    if matches:
        matches.sort(key=lambda p: len(str(p)), reverse=True)
        return matches[0]
    return None


def run_rq1_ablation():
    """RQ1: Ablation Study (Aggregating 5 runs)"""
    from case_main import build_simulation_data_strict_window
    from scipy.stats import pearsonr

    logger.info("Analyzing RQ1: PAEP Ablation Study...")
    base_path = EXP_DIR / "simulated_social_assessment" / "case_verification_and_ablation" / "verification_passed"
    gt = [0, 0, 0, 0, 0, 0, 0, 0.1, 0.09, 0.28, 0.3, 0.34, 0.35, 0.33, 0.42, 0.42, 0.31, 0.27, 0.13, 0.04, 0.04, 0.03,
          0.01, 0.01, 0.01, 0, 0.01, 0, 0, 0]

    def process_group(folder_name):
        group_dir = base_path / folder_name
        if not group_dir.exists(): return None
        pearsons, trajectories = [], []
        for i in range(1, 6):
            run_dir = group_dir / str(i) / "case_validation"
            try:
                sim_r, _ = build_simulation_data_strict_window(str(run_dir), 30, 40, 3)
                ml = min(len(gt), len(sim_r))
                if ml >= 20:
                    p_val = 0.0 if np.std(sim_r[:ml]) == 0 else float(pearsonr(gt[:ml], sim_r[:ml])[0])
                    pearsons.append(p_val);
                    trajectories.append(sim_r[:ml])
            except:
                continue
        return {"mean": np.mean(pearsons), "std": np.std(pearsons),
                "trajectory_30d": np.mean(trajectories, axis=0).tolist()} if pearsons else None

    return {"SimuGov_Ours": process_group("Turn on PAEP"), "Baseline_Off": process_group("Turn off PAEP")}


def run_rq2_comprehensive():
    """RQ2: Fidelity and Efficiency Analysis"""
    logger.info("Analyzing RQ2: RSC Proxy Fidelity & Scalability...")

    # 1. Fidelity
    fidelity = None
    from all_compare import batch_analyze_all_groups_multi_ratios
    fid_path = EXP_DIR / "validation_of_the_effectiveness_of_low_granularity_model_screening" / "verification_passed" / "result"
    if fid_path.exists():
        res = batch_analyze_all_groups_multi_ratios(str(fid_path), [0.2, 0.3, 0.4], False)
        if res.get('status') == 'success':
            fidelity = translate({
                "spearman_mean": res["summary_statistics"].get("spearman_mean"),
                "recall_means": res["summary_statistics"].get("recall_means")
            })

    # 2. Efficiency Scalability
    efficiency = None
    csv_path = find_file_recursive(EXP_DIR / "multi_granularity_method_evaluation" / "efficiency_experiment",
                                   "scalability_metrics.csv")
    if csv_path:
        df = pd.read_csv(csv_path)
        efficiency = {
            "metrics_table": df.to_dict(orient='records'),
            "max_scale_summary": {
                "speedup": float(df.iloc[-1]['time_a'] / df.iloc[-1]['time_b']),
                "cost_saving_percent": float(df.iloc[-1]['saving'])
            }
        }
    return {"Fidelity": fidelity, "Efficiency_Scalability": efficiency}


def run_rq3_convergence():
    """RQ3-1: Convergence Analysis"""
    logger.info("Analyzing RQ3-1: Convergence History...")
    path = EXP_DIR / "multi_granularity_method_evaluation" / "closed_loop_effectiveness_experiment" / "Verification_passed" / "Convergence verification passed" / "exported_data" / "evolution_performance_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[['generation', 'pareto_front_size', 'hypervolume']].to_dict(orient='records')
    return []


def run_rq3_baseline_compare():
    """RQ3-2: Baseline Benchmarking"""
    logger.info("Analyzing RQ3-2: Baseline Comparison...")
    path = EXP_DIR / "multi_granularity_method_evaluation" / "closed_loop_effectiveness_experiment" / "Verification_passed" / "Benchmark comparison verification passed" / "1" / "20260118_193457" / "benchmarking_data.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            "hv_improvement_percent": ((data['metrics']['hv_elite'] / data['metrics']['hv_base']) - 1) * 100,
            "best_elite_vector": translate(data['metrics']['best_elite_robust_vector']),
            "benchmarking_details": translate(data['benchmarking'])
        }
    return None


def run_rq3_adaptability_fixed():
    """RQ3-3: Adaptability Analysis (Physical File Loading)"""
    logger.info("Analyzing RQ3-3: Adaptability from all_total and KPI.json...")

    # Define physical base path for adaptive experiment
    base_data_path = EXP_DIR / "multi_granularity_method_evaluation" / "closed_loop_effectiveness_experiment" / "Verification_passed" / "Adaptive experiment passed" / "2" / "data"
    all_total_file = base_data_path.parent / "all_total"

    if not all_total_file.exists():
        return {"error": "all_total file not found"}

    try:
        with open(all_total_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex patterns to extract Policy IDs from each section
        def get_id(pattern):
            match = re.search(pattern, content, re.DOTALL)
            if match:
                id_m = re.search(r"最佳策略ID:\s*(.*)", match.group(1))
                return id_m.group(1).strip() if id_m else None
            return None

        # Configuration for folder mapping
        group_configs = {
            "Group_A_Compliance": {"sub": "低逆反/elite", "id": get_id(r"低逆反最优解\s*={10,}(.*?)(?=={10,}|$)")},
            "Group_B_Radical": {"sub": "高逆反/elite", "id": get_id(r"高逆反最优解\s*={10,}(.*?)(?=={10,}|$)")},
            "Group_C_Mismatch": {"sub": "低入高/运行数据", "id": get_id(r"低入高结果\s*={10,}(.*?)(?=={10,}|$)")}
        }

        results = {}
        for key, cfg in group_configs.items():
            sid = cfg["id"]
            if not sid:
                results[key] = "ID not found in all_total"
                continue

            # Locate actual KPI file based on ID
            kpi_file = base_data_path / cfg["sub"] / sid / "day_time_15" / "output_system_kpi.json"

            if not kpi_file.exists():  # Fallback search
                policy_dir = base_data_path / cfg["sub"] / sid
                day_dirs = [d for d in policy_dir.iterdir() if
                            d.is_dir() and "day_time" in d.name] if policy_dir.exists() else []
                if day_dirs:
                    latest_day = max(day_dirs, key=lambda x: int(x.name.split('_')[-1]))
                    kpi_file = latest_day / "output_system_kpi.json"

            if kpi_file.exists():
                with open(kpi_file, 'r', encoding='utf-8') as kf:
                    kpi = json.load(kf)

                results[key] = {
                    "policy_id": sid,
                    "metrics": {
                        "theta_jitter": float(np.mean(np.abs(np.diff(kpi["theta"])))) if "theta" in kpi and len(
                            kpi["theta"]) > 1 else 0.0,
                        "final_safety": kpi["safety"][-1] if "safety" in kpi else 0.0
                    },
                    "time_series": {
                        "safety": kpi.get("safety", []), "satisfaction": kpi.get("satisfaction", []),
                        "creativity": kpi.get("creativity", []), "theta": kpi.get("theta", [])
                    }
                }
            else:
                results[key] = f"KPI.json not found for ID {sid}"

        # Evaluation validation
        verification = {}
        try:
            ja = results["Group_A_Compliance"]["metrics"]["theta_jitter"]
            jb = results["Group_B_Radical"]["metrics"]["theta_jitter"]
            verification[
                "Cost_Asymmetry"] = f"Radical({jb:.4f}) > Compliance({ja:.4f}) -> {'PASS' if jb > ja else 'FAIL'}"
            sc = results["Group_C_Mismatch"]["metrics"]["final_safety"]
            verification["System_Collapse"] = f"Safety({sc:.4f}) -> {'PASS' if sc < 0.1 else 'FAIL'}"
        except:
            pass

        return {"results": results, "verification": verification}
    except Exception as e:
        return {"error": str(e)}


def main():
    logger.info("=" * 60)
    logger.info("SimuGov Full Experiment Data Aggregator")
    logger.info("=" * 60)

    results = {
        "RQ1_Ablation": run_rq1_ablation(),
        "RQ2_Proxy_Evaluation": run_rq2_comprehensive(),
        "RQ3_Optimization_System": {
            "Sub1_Convergence": run_rq3_convergence(),
            "Sub2_Baseline_Comparison": run_rq3_baseline_compare(),
            "Sub3_Adaptability": run_rq3_adaptability_fixed()
        }
    }

    final_output = {
        "metadata": {
            "project": "SimuGov (Topic-1 Framework Evaluation)",
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "All Research Questions Included"
        },
        "results": results
    }

    # Save to JSON
    output_path = EXP_DIR / "SimuGov_Consolidated_Report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, cls=NpEncoder, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"✅ Aggregation Completed Successfully!")

    # Terminal Summary Output
    try:
        r = final_output["results"]
        # RQ1
        print(f"1. RQ1 Pearson (Ours): {r['RQ1_Ablation']['SimuGov_Ours']['pearson_mean']:.4f}")
        # RQ2
        print(
            f"2. RQ2 Speedup: {r['RQ2_Proxy_Evaluation']['Efficiency_Scalability']['max_scale_summary']['speedup']:.2f}x")
        # RQ3
        sub2 = r["RQ3_Optimization"]["Sub2_Baseline_Comparison"]
        print(f"3. RQ3 HV Expansion: {sub2['hv_improvement_percent']:.2f}%")
        sub3 = r["RQ3_Optimization"]["Sub3_Adaptability"]
        print(f"4. RQ3 Adaptability Verification: {sub3.get('verification')}")
    except Exception as e:
        print(f"Summary print error: {e}")

    print(f"Full JSON report saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()