import json
import os
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D  # Used to create custom legends
import logging

log = logging.getLogger(__name__)

mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False

BASE_PROJECT_PATH = Path(r'D:\Assign\topic-code\topic-1')
DATA_ROOT = BASE_PROJECT_PATH / 'method' / 'store' / 'daily_memory_exports'


def find_latest_run_directory(root_path: Path) -> Path | None:
    try:
        latest_date_dir = max([d for d in root_path.iterdir() if d.is_dir() and d.name.split('-')[0].isdigit()],
                              key=lambda d: d.name)
        latest_run_dir = max([d for d in latest_date_dir.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()],
                             key=lambda d: d.name)
        log.info(f"✅ Successfully located the latest run directory: {latest_run_dir}")
        return latest_run_dir
    except (ValueError, FileNotFoundError):
        log.error(f"❌ Error: Cannot find any valid run data directory in '{root_path}'.")
        return None


def _get_strategy_dirs(run_directory: Path, is_simplified: bool) -> list:
    if is_simplified:
        simplified_dir_path = run_directory / 'simplified'
        if simplified_dir_path.exists() and simplified_dir_path.is_dir():
            log.info("--- Mode: Only processing data within the 'simplified' folder ---")
            return [d for d in simplified_dir_path.iterdir() if d.is_dir()]
        else:
            log.warning(f"❌ Warning: 'simplified' folder not found in {run_directory}.")
            return []
    else:
        log.info("--- Mode: Only processing top-level policy data (excluding 'simplified' folder) ---")
        return [d for d in run_directory.iterdir() if d.is_dir() and d.name != 'simplified']


def calculate_stable_score(kpi_list: list, theta_history: list = None, penalty_weight: float = 1.0,
                           jitter_weight: float = 2.0) -> float:
    """
    [New] Calculate comprehensive score considering stability and policy jitter.
    Logic remains completely consistent with evaluate_policy in NSGA-II.
    Score = Mean(KPI) - (Weight * StdDev(KPI)) - (JitterWeight * Mean(|Delta Theta|))
    """
    if not kpi_list:
        return 0.0

    data = np.array(kpi_list)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Base score: mean - fluctuation penalty
    score = mean_val - (penalty_weight * std_val)

    # Extra penalty: policy jitter
    if theta_history and len(theta_history) > 1:
        diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
        jitter = np.mean(diffs)
        score -= (jitter * jitter_weight)

    return float(max(0.0, score))


def collect_timeseries_kpi_data(run_directory: Path, is_simplified: bool) -> list:
    """[Restored] Collect complete KPI time series data for each strategy."""
    results = []
    strategy_dirs_to_process = _get_strategy_dirs(run_directory, is_simplified)
    if not strategy_dirs_to_process:
        log.warning("Warning: No eligible strategy result folders found.")
        return []
    for strategy_dir in strategy_dirs_to_process:
        try:
            day_dirs = sorted([d for d in strategy_dir.iterdir() if d.is_dir() and d.name.startswith('day_time_')],
                              key=lambda d: int(d.name.split('_')[-1]))
            if not day_dirs: continue
            last_day_dir = day_dirs[-1]
            kpi_file_path = last_day_dir / 'output_system_kpi.json'
            policy_file_path = last_day_dir / 'output_policy.json'
            if not kpi_file_path.exists() or not policy_file_path.exists(): continue
            with open(kpi_file_path, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
            with open(policy_file_path, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)

            def get_list_from_value(value):
                return value if isinstance(value, list) else []

            kpi_history = {'safety': get_list_from_value(kpi_data.get('safety')),
                           'creativity': get_list_from_value(kpi_data.get('creativity')),
                           'satisfaction': get_list_from_value(kpi_data.get('satisfaction'))}
            theta_history = get_list_from_value(kpi_data.get('theta'))

            # Add as long as there is data, no longer force-validate that lengths are exactly consistent (though usually they are, but for robustness)
            if kpi_history['safety']:
                results.append({'policy': policy_data, 'kpis': kpi_history, 'thetas': theta_history})
                log.info(f"  - (Time Series) Successfully processed strategy: {strategy_dir.name} (Days: {len(kpi_history['safety'])})")
        except Exception as e:
            log.error(f"  - Error: An error occurred while processing folder '{strategy_dir.name}': {e}")
    results.sort(key=lambda r: r['policy'].get('f_penalty', 0))
    return results


def collect_final_kpi_data(run_directory: Path, is_simplified: bool) -> list:
    """
    [Modified version] Collect KPI data for each strategy and calculate 'stability score' for plotting Pareto front.
    No longer just take the last day, but apply the same formula as NSGA.
    """
    results = []
    strategy_dirs_to_process = _get_strategy_dirs(run_directory, is_simplified)
    if not strategy_dirs_to_process: return []
    for strategy_dir in strategy_dirs_to_process:
        try:
            day_dirs = sorted([d for d in strategy_dir.iterdir() if d.is_dir() and d.name.startswith('day_time_')],
                              key=lambda d: int(d.name.split('_')[-1]))
            if not day_dirs: continue
            last_day_dir = day_dirs[-1]

            kpi_file_path = last_day_dir / 'output_system_kpi.json'
            policy_file_path = last_day_dir / 'output_policy.json'

            if not kpi_file_path.exists() or not policy_file_path.exists(): continue

            with open(kpi_file_path, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
            with open(policy_file_path, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)

            # Get complete sequence
            s_list = kpi_data.get('safety', [])
            c_list = kpi_data.get('creativity', [])
            sat_list = kpi_data.get('satisfaction', [])
            theta_list = kpi_data.get('theta', [])

            if not s_list:
                continue

            # === Core modification: calculate stability score ===
            # Use the same weight parameters as evaluate_policy in NSGA
            w_std = 1.0
            w_jitter = 2.0

            # Calculate score (Mean - Std - Jitter)
            score_safety = calculate_stable_score(s_list, theta_list, w_std, w_jitter)
            score_creativity = calculate_stable_score(c_list, theta_list, w_std, w_jitter)
            # Satisfaction can be given slightly different weights, here temporarily keeping consistent or fine-tuning (referencing NSGA code it is 0.8)
            score_satisfaction = calculate_stable_score(sat_list, theta_list, 0.8, w_jitter)

            final_kpis = {
                'safety': score_safety,
                'creativity': score_creativity,
                'satisfaction': score_satisfaction
            }

            results.append({'policy': policy_data, 'final_kpis': final_kpis})
            log.info(
                f"  - (Pareto-Stability Score) Processed strategy: {strategy_dir.name} | S:{score_safety:.2f} C:{score_creativity:.2f}")

        except Exception as e:
            log.error(f"  - Error: An error occurred while processing folder '{strategy_dir.name}': {e}")
    return results


def format_policy_for_legend(policy: dict) -> str:
    key_map = {'f_penalty': 'Penalty', 'ai_threshold': 'Statutory AI Threshold', 'e_edu': 'Education'}
    parts = [f"{key_map.get(k, k)}={v}" for k, v in policy.items()]
    return ", ".join(parts)


def plot_kpi_timeseries(results: list, output_filename: str):
    """Plot KPI curves over time, supporting cases where different strategies have inconsistent run days."""
    if not results:
        log.error("❌ Error: No time series data available for plotting.")
        return

    kpi_names = ['safety', 'creativity', 'satisfaction']
    kpi_titles = {'safety': 'Comparison of Safety over time',
                  'creativity': 'Comparison of Creativity over time',
                  'satisfaction': 'Comparison of Satisfaction over time'}

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 22), sharex=True)
    fig.suptitle('Comparison of KPI evolution over time under different strategies (with daily θ values)', fontsize=22, weight='bold')

    # 1. Find the maximum number of days in all results to set the X-axis range
    max_days = 0
    for res in results:
        # Use safety length as the number of days for this strategy
        days_count = len(res['kpis']['safety'])
        if days_count > max_days:
            max_days = days_count

    if max_days == 0:
        log.error("❌ Error: No valid day information found in the data.")
        return

    # Global X-axis ticks
    global_days_ticks = np.arange(1, max_days + 1)

    for i, kpi_name in enumerate(kpi_names):
        ax = axes[i]
        for result in results:
            policy = result['policy']
            kpi_values = result['kpis'][kpi_name]
            thetas = result['thetas']
            legend_label = format_policy_for_legend(policy)

            # 2. Generate independent X-axis data for the current line
            current_len = len(kpi_values)
            if current_len == 0:
                continue

            current_days = np.arange(1, current_len + 1)

            # Draw curve
            line, = ax.plot(current_days, kpi_values, marker='o', linestyle='-', markersize=5, label=legend_label)

            # Annotate theta value
            # Note: need to handle edge cases where thetas length might not be completely consistent with kpi_values (though theoretically they should be)
            for day_idx, (day, kpi_val) in enumerate(zip(current_days, kpi_values)):
                if day_idx < len(thetas):
                    theta_val = thetas[day_idx]
                    if theta_val is not None:
                        ax.annotate(f'θ={theta_val:.2f}', (day, kpi_val),
                                    textcoords="offset points", xytext=(0, 10), ha='center',
                                    fontsize=8, color=line.get_color(), alpha=0.8)

        ax.set_title(kpi_titles[kpi_name], fontsize=16)
        ax.set_ylabel('KPI Index', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Strategy Parameter Combination', bbox_to_anchor=(1.02, 1), loc='upper left')

        # Set X-axis ticks to the global maximum number of days
        if i == len(kpi_names) - 1:
            ax.set_xlabel('Days (Day)', fontsize=14)
            ax.set_xticks(global_days_ticks)
    for ax in axes:
        ax.tick_params(axis='x', labelbottom=True)

    # Still only need to set the X-axis title for the bottom chart
    axes[-1].set_xlabel('Days (Day)', fontsize=14)

    plt.subplots_adjust(right=0.7)
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        log.info(f"\n✅ Time series plot successfully saved to: {output_filename}")
    except Exception as e:
        log.error(f"\n❌ Error: Failed to save chart. Error: {e}")


def plot_pareto_front(results: list, output_filename: str):
    """Plot Pareto front chart (coordinates are stability scores)."""
    if not results:
        log.error("❌ Error: No data available for plotting Pareto front.")
        return

    points = np.array([[r['final_kpis']['safety'], r['final_kpis']['creativity']] for r in results])
    satisfaction = np.array([r['final_kpis']['satisfaction'] for r in results])

    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if np.any(np.all(points >= p, axis=1) & np.any(points > p, axis=1)):
            is_pareto[i] = False

    pareto_points = points[is_pareto]
    # Sort by safety to correctly connect lines
    if len(pareto_points) > 0:
        pareto_front_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
    else:
        pareto_front_sorted = np.array([])

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(points[:, 0], points[:, 1], s=100, c=satisfaction, cmap='viridis', edgecolors='#333333',
                          linewidths=0.5, zorder=3)

    if len(pareto_front_sorted) > 0:
        plt.plot(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], 'r-', lw=2, zorder=2,
                 label='Pareto Front (Efficiency Boundary)')

    texts = []
    for result in results:
        policy_text = format_policy_for_legend(result['policy'])
        coords = (result['final_kpis']['safety'], result['final_kpis']['creativity'])
        texts.append(plt.text(coords[0], coords[1], policy_text, fontsize=9, color='navy'))

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # === Modified title and axes to reflect score essence ===
    plt.title('Pareto Front of AI Governance Strategies (Score after stability adjustment)', fontsize=20, weight='bold')
    plt.xlabel('Safety Score (Mean - Std - Jitter)', fontsize=14)
    plt.ylabel('Creativity Score (Mean - Std - Jitter)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Satisfaction Score (Satisfaction Score)', fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Strategy solution (color represents satisfaction)', markerfacecolor='gray', markersize=10,
               markeredgecolor='#333333'),
        Line2D([0], [0], color='red', lw=2, label='Pareto Front')
    ]
    plt.legend(handles=legend_elements, fontsize=12)

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        log.info(f"\n✅ Pareto front plot successfully saved to: {output_filename}")
    except Exception as e:
        log.error(f"\n❌ Error: Failed to save chart. Error: {e}")


def draw_kpi_timeseries_main(plot_simplified: bool, output_dir: Path):
    """[Restored] Main function: execute time series plot drawing process"""
    log.info("--- Starting [KPI Time Series] analysis script ---")
    os.makedirs(output_dir, exist_ok=True)
    latest_run_dir = find_latest_run_directory(DATA_ROOT)
    if latest_run_dir:
        kpi_results = collect_timeseries_kpi_data(latest_run_dir, is_simplified=plot_simplified)
        if kpi_results:
            plot_type = "simplified" if plot_simplified else "completed"
            output_filename = output_dir / f"kpi_timeseries_{plot_type}.png"
            plot_kpi_timeseries(kpi_results, output_filename)
    log.info("--- Script execution finished ---")


def draw_pareto_front_main(plot_simplified: bool, output_dir: Path):
    """Main function: execute Pareto front plot drawing process"""
    log.info("--- Starting [Pareto Front] analysis script ---")
    os.makedirs(output_dir, exist_ok=True)
    latest_run_dir = find_latest_run_directory(DATA_ROOT)
    if latest_run_dir:
        final_kpi_results = collect_final_kpi_data(latest_run_dir, is_simplified=plot_simplified)
        if final_kpi_results:
            plot_type = "simplified" if plot_simplified else "completed"
            output_filename = output_dir / f"pareto_front_{plot_type}.png"
            plot_pareto_front(final_kpi_results, output_filename)
    log.info("--- Script execution finished ---")


def draw_kpi_main(plot_mode='T', use_simplified_data=False,
                  output_dir: Path = Path(r'result_data')):
    """

    :param output_dir: File output directory
    :param plot_mode: Select plotting mode: 'T (time series comparison)' or 'P (Pareto front)'
    :param use_simplified_data: Select data source: True represents 'simplified' folder, False represents top-level folder
    :return:
    """

    if plot_mode == 'T':
        draw_kpi_timeseries_main(plot_simplified=use_simplified_data, output_dir=output_dir)
    elif plot_mode == 'P':
        draw_pareto_front_main(plot_simplified=use_simplified_data, output_dir=output_dir)
    else:
        log.error(f"❌ Error: Invalid PLOT_MODE '{plot_mode}'. Please choose 'timeseries' or 'pareto'.")
    pass


if __name__ == '__main__':
    draw_kpi_main(plot_mode='P', use_simplified_data=True)
    pass