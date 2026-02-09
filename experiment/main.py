import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. 核心路径处理 (解决 ModuleNotFoundError)
# ==========================================

# 获取 experiment 目录的绝对路径
CURRENT_DIR = Path(__file__).resolve().parent


def auto_append_paths(root_path):
    """
    递归地将所有子目录添加到 sys.path 中。
    这样子目录中的脚本相互调用（如 from ana_utils import ...）时就不会报错。
    """
    for root, dirs, files in os.walk(root_path):
        if "__pycache__" in root:
            continue
        if root not in sys.path:
            sys.path.append(root)
            # logger 不能在这里用，因为还没配置，先用 print
            # print(f"Added to path: {root}")


# 执行路径自动添加
auto_append_paths(str(CURRENT_DIR))

# ==========================================
# 2. 配置与日志
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperimentAggregator")

# --- 实验数据路径配置 ---
CONFIG = {
    "OUTPUT_FILE": "final_experiment_summary.json",

    "MACRO_DATA": r'experiment\仿真社会评估\宏观行为验证\data\case_policy',
    "INTERNAL_CONSISTENCY_LOW": r'experiment\仿真社会评估\内部一致性验证\data\eva_compare\test-2\low\惩罚0_99_教育低_ai_threshold_0_01',
    "INTERNAL_CONSISTENCY_HIGH": r'experiment\仿真社会评估\内部一致性验证\data\eva_compare\test-2\high\惩罚0_99_教育低_ai_threshold_0_01',
    "ROBUSTNESS_DATA": r'experiment\仿真社会评估\内部一致性验证\data\eva_robustness',
    "CASE_DATA": r'experiment\仿真社会评估\案例验证\data\开启心理参数\case_validation',
    "LOW_FIDELITY_DATA": r"experiment\低粒度模型筛选有效性验证\验证通过\result",
    "EFFICIENCY_DATA": r'experiment\多粒度方法评估\效率实验\data\155812',
    "SENSITIVITY_DATA": r'experiment\多粒度方法评估\机制敏感度实验\data',
    "CLOSED_LOOP_DATA": r"experiment\多粒度方法评估\闭环有效性实验\data"
}


# ==========================================
# 3. 工具类
# ==========================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records')
        return super(NpEncoder, self).default(obj)


# ==========================================
# 4. 实验适配器 (逻辑保持不变)
# ==========================================

def run_macro_behavior_analysis():
    logger.info("正在执行: 宏观行为验证分析...")
    # 此时 sys.path 已经包含了该目录，直接 import 不会报错
    try:
        from mac_main import ExperimentAutomator
        path = CONFIG["MACRO_DATA"]
        if not os.path.exists(path): return None

        # 这种方式是为了不修改原有的 mac_main 代码，复用其内部逻辑
        automator = ExperimentAutomator(path, os.path.join(CURRENT_DIR, "temp_macro"))
        summary_report = {}
        top_sub_dirs = [d for d in os.listdir(automator.data_root) if
                        os.path.isdir(os.path.join(automator.data_root, d))]
        is_2_layer = any(automator._get_max_runtime_day(os.path.join(automator.data_root, d)) > 0 for d in top_sub_dirs)
        if is_2_layer:
            summary_report["Direct_Base"] = automator._process_group("Direct_Base", automator.data_root)
        else:
            for group in top_sub_dirs:
                summary_report[group] = automator._process_group(group, os.path.join(automator.data_root, group))
        return summary_report
    except Exception as e:
        logger.error(f"宏观行为验证解析失败: {e}")
        return None


def run_internal_consistency_analysis():
    logger.info("正在执行: 内部一致性与鲁棒性分析...")
    try:
        import scipy.stats as stats
        from eva_compare import evaluate_extreme_psychology_experiment
        from eva_robustness import load_robustness_kpi_data

        results = {}
        # 极端对比
        p_low, p_high = CONFIG["INTERNAL_CONSISTENCY_LOW"], CONFIG["INTERNAL_CONSISTENCY_HIGH"]
        if os.path.exists(p_low) and os.path.exists(p_high):
            def load_last_kpi(p):
                dirs = [d for d in os.listdir(p) if d.startswith("day_time_")]
                if not dirs: return None
                md = max(dirs, key=lambda x: int(x.split('_')[-1]))
                with open(os.path.join(p, md, 'output_system_kpi.json'), 'r', encoding='utf-8') as f:
                    return json.load(f)

            l_d, h_d = load_last_kpi(p_low), load_last_kpi(p_high)
            if l_d and h_d:
                c_res = {}
                for m in ['safety', 'satisfaction']:
                    t_s, p_v = stats.ttest_ind(l_d[m], h_d[m], equal_var=False)
                    c_res[m] = {"low_mean": np.mean(l_d[m]), "high_mean": np.mean(h_d[m]),
                                "diff": np.mean(l_d[m]) - np.mean(h_d[m]), "p_value": p_v}
                results["extreme_comparison"] = c_res

        # 鲁棒性
        p_rob = CONFIG["ROBUSTNESS_DATA"]
        if os.path.exists(p_rob):
            all_runs = load_robustness_kpi_data(p_rob)
            if all_runs:
                rob_res = {}
                for m in ['safety', 'satisfaction', 'creativity']:
                    vals = [r[m][-1] for r in all_runs]
                    mean_v = np.mean(vals)
                    rob_res[m] = {"final_mean": mean_v, "cv": np.std(vals) / mean_v if mean_v != 0 else 0}
                results["robustness"] = rob_res
        return results
    except Exception as e:
        logger.error(f"内部一致性解析失败: {e}")
        return None


def run_case_validation_analysis():
    logger.info("正在执行: 案例验证分析...")
    try:
        from case_main import build_simulation_data_strict_window
        from case_validator import CaseValidator
        path = CONFIG["CASE_DATA"]
        if not os.path.exists(path): return None
        gt = [0, 0, 0, 0, 0, 0, 0, 0.1, 0.09, 0.28, 0.3, 0.34, 0.35, 0.33, 0.42, 0.42, 0.31, 0.27, 0.13, 0.04, 0.04,
              0.03, 0.01, 0.01, 0.01, 0, 0.01, 0, 0, 0]
        sim_r, sim_s = build_simulation_data_strict_window(path, 30, 40, 3)
        ml = min(len(gt), len(sim_r))
        return {
            "metrics": {
                "trend": CaseValidator.validate_trend_correlation(gt[:ml], sim_r[:ml]),
                "mechanism": CaseValidator.validate_mechanism_causality(sim_s[:ml], sim_r[:ml]),
                "peak": CaseValidator.validate_peak_alignment(gt[:ml], sim_r[:ml])
            }
        }
    except Exception as e:
        logger.error(f"案例验证解析失败: {e}")
        return None


def run_low_fidelity_analysis():
    logger.info("正在执行: 低粒度模型筛选有效性分析...")
    try:
        from all_compare import batch_analyze_all_groups_multi_ratios
        path = CONFIG["LOW_FIDELITY_DATA"]
        if not os.path.exists(path): return None
        res = batch_analyze_all_groups_multi_ratios(path, [0.2, 0.3, 0.4], False)
        if res.get('status') == 'success':
            return {"summary": res.get("summary_statistics"),
                    "groups": [{"id": r['group_id'], "metrics": r['metrics']} for r in res.get("all_results", []) if
                               r.get('status') == 'success']}
        return res
    except Exception as e:
        logger.error(f"低粒度筛选解析失败: {e}")
        return None


def run_efficiency_analysis():
    logger.info("正在执行: 效率实验分析...")
    try:
        from efficiency_main import load_experiment_data, calculate_efficiency_metrics, calculate_quality_metrics
        path = CONFIG["EFFICIENCY_DATA"]
        if not os.path.exists(path): return None
        raw = load_experiment_data(path)
        return {"efficiency": calculate_efficiency_metrics(raw),
                "quality": calculate_quality_metrics(raw)} if raw else None
    except Exception as e:
        logger.error(f"效率分析失败: {e}")
        return None


def run_sensitivity_analysis():
    logger.info("正在执行: 机制敏感度分析...")
    try:
        import sen_main
        sen_main.DATA_ROOT = Path(CONFIG["SENSITIVITY_DATA"])
        df = sen_main.load_results()
        return df.groupby("Group").mean(numeric_only=True).to_dict(orient="index") if not df.empty else None
    except Exception as e:
        logger.error(f"敏感度分析失败: {e}")
        return None


def run_closed_loop_analysis():
    logger.info("正在执行: 闭环有效性分析...")
    try:
        from base_comparison_main import load_all_experiment_data, get_robust_vector
        import pygmo as pg
        path = Path(CONFIG["CLOSED_LOOP_DATA"])
        if not path.exists(): return None
        data = load_all_experiment_data(path)
        if not data['elites'] or not data['baselines']: return None
        e_vecs = np.array([get_robust_vector(e) for e in data['elites']])
        b_vecs = np.array([get_robust_vector(b) for b in data['baselines']])
        best_idx = np.argmax(np.sum(e_vecs, axis=1))

        def chv(m):
            try:
                return float(pg.hypervolume(-1.0 * m).compute([0.01, 0.01, 0.01]))
            except:
                return 0.0

        return {
            "metrics": {"hv_elite": chv(e_vecs), "hv_base": chv(b_vecs),
                        "improvement": (chv(e_vecs) / chv(b_vecs) - 1) if chv(b_vecs) > 0 else 0},
            "best_elite": {"id": data['elites'][best_idx]['id'], "vector": e_vecs[best_idx].tolist()},
            "base_avg": np.mean(b_vecs, axis=0).tolist()
        }
    except Exception as e:
        logger.error(f"闭环分析失败: {e}")
        return None


# ==========================================
# 5. 主执行逻辑
# ==========================================

def main():
    logger.info("=" * 50)
    logger.info("开始汇总所有子实验结果数据")
    logger.info("=" * 50)

    final_output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project": "Topic-1 Framework Evaluation"
        },
        "results": {
            "social_assessment": {
                "macro_behavior": run_macro_behavior_analysis(),
                "internal_consistency": run_internal_consistency_analysis(),
                "case_validation": run_case_validation_analysis()
            },
            "low_fidelity_efficiency": run_low_fidelity_analysis(),
            "method_evaluation": {
                "efficiency": run_efficiency_analysis(),
                "sensitivity": run_sensitivity_analysis(),
                "closed_loop": run_closed_loop_analysis()
            }
        }
    }

    out_path = os.path.join(CURRENT_DIR, CONFIG["OUTPUT_FILE"])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    logger.info("=" * 50)
    logger.info(f"汇总完成！数据文件: {out_path}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()