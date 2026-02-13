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
    Simulate the real social platform hot list mechanism:
    1. Limited capacity (Top K)
    2. Strong timeliness (Lookback Window) - Content older than N days is forced to expire and cannot reach the front page.

    :param lookback_days: Lookback days. Suggested to set to 2 or 3. This means content from 3 days ago automatically disappears from the front page.
    """
    target_dir = os.path.join(input_file_root, f"day_time_{end_day}")
    content_path = os.path.join(target_dir, "output_contents.json")
    persona_path = os.path.join(target_dir, "output_personas.json")

    if not os.path.exists(content_path):
        raise FileNotFoundError(f"Cannot find data file: {content_path}")

    print(f"📖 Reading data: {target_dir}")
    with open(content_path, 'r', encoding='utf-8') as f:
        all_contents = json.load(f)
    with open(persona_path, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)

    sim_protest_ratios = []
    sim_satisfaction = []

    print(f"🔄 Calculating front page: Capacity={capacity}, Validity=Past {lookback_days} days...")

    # === Traverse each day ===
    for day in range(1, end_day + 1):
        window_start = max(1, day - lookback_days + 1)

        valid_pool = [
            c for c in all_contents
            if window_start <= c['time'] <= day
        ]

        # Sort in reverse chronological order (latest on top)
        # Default sorting: If time is the same, sort by ID in descending order
        valid_pool.sort(key=lambda x: (x['time'], str(x['id'])), reverse=True)

        # Intercept front page capacity (simulating the range users can see)
        homepage = valid_pool[:capacity]

        # Count protest content
        protest_count = 0
        for c in homepage:
            topic = str(c.get('topic', '')).upper()
            detail = str(c.get('content_detail', '')).upper()

            # Determination criteria
            if 'NO AI' in topic or 'PROTEST' in topic or 'NO AI' in detail:
                protest_count += 1

        # Calculate proportion
        # Denominator: Actual number of front page displays (may be less than capacity in the early stages)
        denominator = len(homepage)
        ratio = protest_count / denominator if denominator > 0 else 0.0
        sim_protest_ratios.append(ratio)

        # -----------------------------------
        # 2. Satisfaction calculation
        # -----------------------------------
        sat_sum = 0
        count = 0
        for p in all_personas:
            if p['type'] == 'Compliant Creator':
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
    Load data from multiple run folders and return a matrix.
    """
    all_runs_ratios = []
    all_runs_sats = []

    print(f"🚀 Starting to aggregate {len(run_dirs)} run results...")

    for i, root_dir in enumerate(run_dirs):
        print(f"   Reading Run #{i + 1}: {root_dir}")
        try:
            # Reuse the previous build_simulation_data_strict_window function
            ratios, sats = build_simulation_data_strict_window(
                root_dir, end_day, capacity, lookback_days
            )
            all_runs_ratios.append(ratios)
            all_runs_sats.append(sats)
        except Exception as e:
            print(f"   ⚠️ Run #{i + 1} failed to load: {e}")

    # Convert to Numpy matrix [N_runs, N_days]
    # Note: Ensure that the day lengths of all runs are consistent; truncate if inconsistent
    min_len = min([len(r) for r in all_runs_ratios])

    ratio_matrix = np.array([r[:min_len] for r in all_runs_ratios])
    sat_matrix = np.array([s[:min_len] for s in all_runs_sats])

    return ratio_matrix, sat_matrix


def plot_confidence_interval(run_directories):
    """
    Plot confidence interval charts for multiple sets of run results
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
        # Load data matrix
        ratio_matrix, sat_matrix = load_multi_run_data(
            run_directories,
            end_day=30,
            capacity=40,
            lookback_days=3
        )

        # Calculate mean curve for metric validation
        sim_mean_curve = np.mean(ratio_matrix, axis=0).tolist()
        sat_mean_curve = np.mean(sat_matrix, axis=0).tolist()

        # Data alignment
        min_len = min(len(ground_truth_ratios), len(sim_mean_curve))
        gt_aligned = ground_truth_ratios[:min_len]
        sim_mean_aligned = sim_mean_curve[:min_len]

        # --- Validation (using the mean curve for scoring) ---
        print("\n" + "=" * 40)
        print(f"🔬 Multi-run aggregated validation report (N={len(run_directories)})")
        print("=" * 40)

        # 1. Trend validation (comparing means)
        trend_res = CaseValidator.validate_trend_correlation(gt_aligned, sim_mean_aligned)
        print(f"📈 [Mean Trend] Pearson: {trend_res['pearson']['value']} | Spearman: {trend_res['spearman']['value']}")

        # 2. Robustness validation (New: calculate the mean of standard deviations)
        avg_std = np.mean(np.std(ratio_matrix, axis=0))
        print(f"🛡️ [Robustness] Average Standard Deviation (Avg STD): {avg_std:.4f} (lower is more stable)")

        # 3. Plotting (with confidence intervals)
        save_path = CaseValidator.plot_trend_with_ci(
            gt_aligned,
            ratio_matrix,  # Pass the matrix
            title=f"ArtStation Simulation Robustness (N={len(run_directories)})"
        )
        print(f"🖼️ Confidence interval chart saved: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


def analysis_single_data(input_file_root):
    """
    Analyze the result of a single simulation run
    :return:
    """
    # 1. Real historical data
    ground_truth_ratios = [
        0, 0, 0, 0, 0, 0, 0,
        0.10, 0.09, 0.28, 0.30, 0.34,
        0.35, 0.33, 0.42, 0.42, 0.31,
        0.27, 0.13, 0.04, 0.04, 0.03,
        0.01, 0.01, 0.01, 0, 0.01, 0, 0, 0
    ]

    # 3. Parameters
    HOMEPAGE_CAPACITY = 40
    CONTENT_EXPIRATION_DAYS = 3

    try:
        sim_ratios, sim_sat = build_simulation_data_strict_window(
            input_file_root,
            end_day=30,
            capacity=HOMEPAGE_CAPACITY,
            lookback_days=CONTENT_EXPIRATION_DAYS
        )

        # Data alignment
        min_len = min(len(ground_truth_ratios), len(sim_ratios))
        gt_aligned = ground_truth_ratios[:min_len]
        sim_aligned = sim_ratios[:min_len]
        sat_aligned = sim_sat[:min_len]

        # --- Output report ---
        print("\n" + "=" * 40)
        print("🔬 Case Validation Report (Timeliness Window Model)")
        print("=" * 40)

        # Here CaseValidator internally prints a detailed report with ratings
        trend_res = CaseValidator.validate_trend_correlation(gt_aligned, sim_aligned)
        mech_res = CaseValidator.validate_mechanism_causality(sat_aligned, sim_aligned)
        peak_res = CaseValidator.validate_peak_alignment(gt_aligned, sim_aligned)

        # --- [Fix Point] The following simple summary print lines need key names updated ---
        print(f"\n📝 Brief Summary:")
        print(f"📈 [Trend Fitting] Pearson: {trend_res['pearson']['value']} | Spearman: {trend_res['spearman']['value']}")
        print(f"🏔️ [Time Series Sync] Lag: {peak_res['lag_days']} days (Sim Peak: Day {peak_res['sim_peak_day']})")
        print(f"⚙️ [Mechanism Validation] Correlation: {mech_res['correlation']}")

        # Plotting
        output_dir = r'experiment\simulation_social_assessment\case_validation\output'
        save_path = CaseValidator.plot_trend_comparison(
            gt_aligned,
            sim_aligned,
            title=f"ArtStation Validation: Front Page Ratio (Top{HOMEPAGE_CAPACITY})",
            output_dir=output_dir
        )
        print(f"Chart saved: {save_path}")

        # Save JSON
        res_data = {
            "metrics": {"trend": trend_res, "peak": peak_res, "mechanism": mech_res},
            "data": {"ground_truth": gt_aligned, "simulation": sim_aligned, "satisfaction": sat_aligned}
        }
        json_path = os.path.join(output_dir, f"validation_report_{datetime.now().strftime('%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(res_data, f, ensure_ascii=False, indent=4, cls=NpEncoder)

    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    # Plot confidence interval
    # 2. Directory list of multiple runs
    # directories = [
    #     r'experiment\Simulated_social_assessment\Case_Verification_and_Ablation\data\Passed\155658\Punishment0_01_LowEducation_ai_threshold_0_8',
    #     r'experiment\Simulated_social_assessment\Case_Verification_and_Ablation\data\t7\case_validation',
    # ]
    # plot_confidence_interval(directories)
    # Analyze the results of a single run
    # 2. Input path
    # input_file = r'experiment\simulation_social_assessment\case_validation\data\enabled_psychological_parameters\case_validation'
    # analysis_single_data(input_file)
    pass