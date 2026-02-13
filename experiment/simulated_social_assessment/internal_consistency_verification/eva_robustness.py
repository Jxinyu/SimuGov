import json
import os
import re
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Any

mpl.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def evaluate_monte_carlo_experiment(
        all_runs_data: List[Dict[str, List[float]]],
        output_dir: str,
        cv_threshold: float = 0.20
):
    """
    Evaluate the robustness and convergence of Monte Carlo simulations.
    By calculating the coefficient of variation (CV) and confidence intervals of multiple runs, determine whether the system results converge stably.
    And output the detailed statistical data required for plotting.
    """

    # 1. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Data reorganization and preprocessing
    metrics = ['safety', 'satisfaction', 'creativity']
    processed_data = {m: [] for m in metrics}

    # Get the minimum number of days to prevent crashes caused by inconsistent lengths across different runs
    min_days = min(len(run['safety']) for run in all_runs_data)

    for run in all_runs_data:
        for m in metrics:
            # Truncate to minimum length
            processed_data[m].append(run[m][:min_days])

    # Convert to numpy matrix: shape = (N_runs, min_days)
    for m in metrics:
        processed_data[m] = np.array(processed_data[m])

    # 3. Statistical calculation function
    def calculate_stats(data_matrix):
        """Calculate the mean, confidence interval, and coefficient of variation of the final state"""
        N = data_matrix.shape[0]
        # Statistics in the time dimension
        mean_series = np.mean(data_matrix, axis=0)
        std_series = np.std(data_matrix, axis=0)
        sem_series = stats.sem(data_matrix, axis=0)  # Standard error

        # 95% Confidence interval
        ci_h = sem_series * stats.t.ppf((1 + 0.95) / 2., N - 1)
        lower_bound = mean_series - ci_h
        upper_bound = mean_series + ci_h

        # Calculate coefficient of variation (CV) - based on the final state (Final Day)
        final_values = data_matrix[:, -1]
        final_mean = np.mean(final_values)
        final_std = np.std(final_values)

        # Prevent division by zero
        if final_mean == 0:
            cv = 0.0
        else:
            cv = final_std / final_mean

        return mean_series, lower_bound, upper_bound, cv

    # Execute calculations
    safe_mean, safe_low, safe_high, safe_cv = calculate_stats(processed_data['safety'])
    sat_mean, sat_low, sat_high, sat_cv = calculate_stats(processed_data['satisfaction'])
    creat_mean, creat_low, creat_high, creat_cv = calculate_stats(processed_data['creativity'])

    # 4. Decision logic
    is_passed = (safe_cv < cv_threshold) and (sat_cv < cv_threshold) and (creat_cv < cv_threshold)

    # 5. Draw Monte Carlo cloud plot (Shadow Plot)
    days = list(range(1, min_days + 1))  # Convert to list for convenient subsequent JSON serialization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Drawing helper function ---
    def plot_cloud(ax, matrix, mean, lower, upper, title, color):
        # Draw each individual trajectory (Ghost Lines)
        for i in range(matrix.shape[0]):
            ax.plot(days, matrix[i, :], color='gray', alpha=0.15, linewidth=1)

        # Draw mean line
        ax.plot(days, mean, color=color, linewidth=2.5, label='Mean')

        # Draw confidence interval (Shadow)
        ax.fill_between(days, lower, upper, color=color, alpha=0.25, label='95% Confidence Interval')

        ax.set_title(title, fontsize=14)
        ax.set_ylabel('KPI Value')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Draw Safety
    plot_cloud(ax1, processed_data['safety'], safe_mean, safe_low, safe_high,
               f'Safety Convergence Analysis (CV={safe_cv:.4f})', 'blue')

    # Draw Satisfaction
    plot_cloud(ax2, processed_data['satisfaction'], sat_mean, sat_low, sat_high,
               f'Satisfaction Convergence Analysis (CV={sat_cv:.4f})', 'green')

    # Draw Creativity
    plot_cloud(ax3, processed_data['creativity'], creat_mean, creat_low, creat_high,
               f'Creativity Convergence Analysis (CV={creat_cv:.4f})', 'yellow')

    ax3.set_xlabel('Simulation Days (Days)')

    # Add threshold line explanation
    status_text = "Robustness validation passed (Convergence)" if is_passed else "Robustness validation warning (Divergence)"
    fig.suptitle(f'Monte Carlo simulation results (N={len(all_runs_data)}): {status_text}',
                 fontsize=16, fontweight='bold', color='green' if is_passed else '#d62728')

    plot_save_path = os.path.join(output_dir, 'monte_carlo_robustness.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Print report
    print("\n" + "=" * 50)
    print(" >>> Monte Carlo Robustness Validation Report <<<")
    print("=" * 50)
    print(f"Number of runs (N): {len(all_runs_data)}")
    print(f"Validation status: {status_text}")
    print(f"CV threshold standard: < {cv_threshold}")
    print("-" * 30)
    print(f"[Safety]")
    print(f"  - Final mean: {safe_mean[-1]:.4f}")
    print(f"  - Coefficient of variation (CV): {safe_cv:.4f}")
    print("-" * 30)
    print(f"[Satisfaction]")
    print(f"  - Final mean: {sat_mean[-1]:.4f}")
    print(f"  - Coefficient of variation (CV): {sat_cv:.4f}")
    print("=" * 50 + "\n")

    # 7. Save result summary (result.json)
    result_json_path = os.path.join(output_dir, 'result.json')
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "is_passed": str(is_passed),
                "safety_cv": float(safe_cv),
                "satisfaction_cv": float(sat_cv),
                "creativity_cv": float(creat_cv),
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    # =================================================================
    # 8. [New] Save plotting source data (monte_carlo_plotting_data.json)
    #    Contains daily mean, confidence interval upper and lower bounds, convenient for subsequent re-drawing
    # =================================================================
    plotting_data = {
        "meta": {
            "n_runs": len(all_runs_data),
            "days_count": len(days),
            # [Fix Point]: Use bool() to cast numpy.bool_ to python bool
            "is_passed": bool(is_passed)
        },
        "axis": {
            "days": days
        },
        "metrics": {
            "safety": {
                # .tolist() has already handled the conversion from numpy array to list
                "mean": safe_mean.tolist(),
                "ci_lower": safe_low.tolist(),
                "ci_upper": safe_high.tolist()
            },
            "satisfaction": {
                "mean": sat_mean.tolist(),
                "ci_lower": sat_low.tolist(),
                "ci_upper": sat_high.tolist()
            },
            "creativity": {
                "mean": creat_mean.tolist(),
                "ci_lower": creat_low.tolist(),
                "ci_upper": creat_high.tolist()
            }
        }
    }

    data_save_path = os.path.join(output_dir, 'monte_carlo_plotting_data.json')
    with open(data_save_path, 'w', encoding='utf-8') as f:
        json.dump(plotting_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Result summary has been saved to: {result_json_path}")
    print(f"✅ Plotting source data has been saved to: {data_save_path}")


def load_robustness_kpi_data(root_path: str) -> List[Dict[str, List[float]]]:
    """
    Load KPI data from multiple runs from the eva_robustness directory.
    Automatically look for output_system_kpi.json of the "last day" under each run folder.
    """
    root = Path(root_path)
    all_runs_data = []

    if not root.exists():
        print(f"❌ Error: Path does not exist - {root}")
        return []

    # 1. Get all run number folders (1, 2, 3...)
    run_dirs = [d for d in root.iterdir() if d.is_dir()]

    # Try to sort by number
    try:
        run_dirs.sort(key=lambda x: int(x.name))
    except ValueError:
        run_dirs.sort(key=lambda x: x.name)

    print(f"📂 Scanned {len(run_dirs)} run directories...")

    for run_dir in run_dirs:
        # 2. Enter run directory, search for policy folders
        policy_dirs = [d for d in run_dir.iterdir() if d.is_dir()]

        if not policy_dirs:
            print(f"⚠️ Warning: {run_dir.name} is empty, skipping.")
            continue

        target_policy_dir = policy_dirs[0]

        # 3. Search for the last day (day_time_X)
        day_folders = []
        for d in target_policy_dir.iterdir():
            if d.is_dir() and d.name.startswith("day_time_"):
                match = re.search(r"day_time_(\d+)", d.name)
                if match:
                    day_num = int(match.group(1))
                    day_folders.append((day_num, d))

        if not day_folders:
            print(f"⚠️ Warning: day_time folder not found in {target_policy_dir.name}, skipping.")
            continue

        # Find the folder with the maximum number of days
        max_day, max_day_folder = max(day_folders, key=lambda x: x[0])

        # 4. Read output_system_kpi.json
        kpi_file = max_day_folder / "output_system_kpi.json"

        if not kpi_file.exists():
            print(f"❌ Error: File does not exist - {kpi_file}")
            continue

        try:
            with open(kpi_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            run_data = {
                "safety": data.get("safety", []),
                "satisfaction": data.get("satisfaction", []),
                "creativity": data.get("creativity", []),
            }

            if run_data["safety"]:
                all_runs_data.append(run_data)
            else:
                print(f"⚠️ Warning: Run {run_dir.name} data is empty.")

        except Exception as e:
            print(f"❌ Error reading JSON ({kpi_file}): {e}")

    print(f"🎉 Successfully loaded {len(all_runs_data)} groups of valid data.")
    return all_runs_data


def eva_robustness(root_path: str, output_dir: str, cv_threshold: float = 0.2) -> None:
    """
    Robustness evaluation entry function
    """
    # Slightly adjust the directory structure to avoid messiness
    output_dir = os.path.join(output_dir, 'robustness_analysis')
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    all_runs_data = load_robustness_kpi_data(root_path)
    if all_runs_data:
        evaluate_monte_carlo_experiment(all_runs_data, output_dir, cv_threshold)
    else:
        print("No valid data loaded, unable to perform evaluation.")