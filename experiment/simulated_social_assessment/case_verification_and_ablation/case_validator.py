import os
from datetime import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from typing import List, Dict, Union

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class CaseValidator:
    """
    ArtStation case validation dedicated evaluation toolbox
    Contains automatic rating and detailed metric explanation.
    """

    @staticmethod
    def _rate_correlation(value: float) -> str:
        """Helper function: rate correlation coefficients"""
        v = abs(value)
        if v >= 0.8: return "Extremely Strong"
        if v >= 0.6: return "Strong"
        if v >= 0.4: return "Moderate"
        return "Weak/No Correlation"

    @staticmethod
    def _rate_mae(value: float) -> str:
        """Helper function: rate MAE error"""
        if value <= 0.1: return "Excellent"
        if value <= 0.2: return "Good"
        if value <= 0.3: return "Fair"
        return "Large Deviation"

    @staticmethod
    def validate_trend_correlation(ground_truth: List[float], simulation: List[float]) -> Dict:
        """
        [Dimension 1: Trend Consistency Validation]

        Parameter role description:
        1. Pearson: Measure [linear] fitness. The higher the value, the more the shape of the simulation curve "overlaps" with the real curve.
        2. Spearman: Measure [rank] consistency. Does not care about absolute values, only cares whether the "up and down trends" are synchronized. More robust for non-linear relationships.
        3. MAE (Mean Absolute Error): Measure [numerical] offset. Lower is better, indicating that simulation values are very close to real values in absolute terms.

        Rating standards (Pearson/Spearman):
        - < 0.4: ❌ Poor (Model Failure)
        - 0.4 - 0.6: ⚠️ Moderate (Barely Passing)
        - 0.6 - 0.8: ✅ Excellent (Recognized as valid in academia)
        - > 0.8: 🌟 SOTA (Extremely high precision fitting)
        """
        # 1. Data alignment
        if len(ground_truth) != len(simulation):
            min_len = min(len(ground_truth), len(simulation))
            ground_truth = ground_truth[:min_len]
            simulation = simulation[:min_len]

        # 2. Calculate metrics
        if np.std(ground_truth) == 0 or np.std(simulation) == 0:
            p_corr, s_corr = 0.0, 0.0
        else:
            p_corr, _ = pearsonr(ground_truth, simulation)
            s_corr, _ = spearmanr(ground_truth, simulation)

        mae = mean_absolute_error(ground_truth, simulation)

        # 3. Get ratings
        p_rating = CaseValidator._rate_correlation(p_corr)
        s_rating = CaseValidator._rate_correlation(s_corr)
        m_rating = CaseValidator._rate_mae(mae)

        # 4. Print detailed report
        print("\n" + "-" * 60)
        print("[Dimension 1] Trend Consistency Validation Report")
        print("-" * 60)
        print(f"1. Pearson (Linear Fitness): {p_corr:.4f}  =>  {p_rating}")
        print(f"   * Explanation: Whether the simulation curve accurately reproduced the undulation shape of the historical curve.")
        print(f"2. Spearman (Trend Synchronicity): {s_corr:.4f}  =>  {s_rating}")
        print(f"   * Explanation: Whether the simulation captured the monotonicity patterns of 'when rising and when falling'.")
        print(f"3. MAE (Numerical Mean Error):    {mae:.4f}    =>  {m_rating}")
        print(f"   * Explanation: Average daily proportion prediction deviation amount (e.g., 0.05 represents an average deviation of 5%).")
        print("-" * 60)

        return {
            "pearson": {"value": round(p_corr, 4), "rating": p_rating},
            "spearman": {"value": round(s_corr, 4), "rating": s_rating},
            "mae": {"value": round(mae, 4), "rating": m_rating}
        }

    @staticmethod
    def validate_mechanism_causality(satisfaction_curve: List[float], protest_curve: List[float]) -> Dict:
        """
        [Dimension 3: Mechanism Validity Validation]

        Parameter role description:
        1. Correlation: Measure the relationship between [satisfaction] and [protest proportion].
        2. Target direction: Must be [negative correlation]. That is, the lower the satisfaction, the more protests.

        Rating standards (correlation coefficient r):
        - > -0.3: ❌ Invalid (No causal relationship, protests might occur randomly)
        - -0.3 ~ -0.5: ⚠️ Weak (Some relationship, but not significant)
        - -0.5 ~ -0.7: ✅ Valid (Significant causal chain exists)
        - < -0.7: 🌟 Strong Robustness (Proves protests are entirely driven by emotions, the emergence mechanism is very solid)
        """
        min_len = min(len(satisfaction_curve), len(protest_curve))
        if min_len < 2:
            corr = 0.0
        else:
            corr, _ = pearsonr(satisfaction_curve[:min_len], protest_curve[:min_len])

        # Rating logic (for negative correlation)
        if corr > -0.3:
            rating = "Invalid/Random"
        elif corr > -0.5:
            rating = "Weak Correlation"
        elif corr > -0.7:
            rating = "Valid"
        else:
            rating = "Strong Robustness"

        # Print detailed report
        print("\n" + "-" * 60)
        print("[Dimension 3] Mechanism Causality Validation Report")
        print("-" * 60)
        print(f"Metric: Satisfaction-Protest negative correlation coefficient")
        print(f"Value: {corr:.4f}")
        print(f"Rating: {rating}")
        print(f"Judgment basis:")
        print(f"   - Only when significant negative correlation (r < -0.5) appears can it prove that 'protests' are an emergent phenomenon driven by 'dissatisfaction'.")
        print(f"   - If close to 0, it indicates protests are randomly injected, and simulation failed.")
        print("-" * 60)

        return {
            "correlation": round(corr, 4),
            "rating": rating,
            "is_valid": corr < -0.5
        }

    @staticmethod
    def plot_trend_comparison(ground_truth: List[float], simulation: List[float],
                              title: str = "Trend Comparison",
                              output_dir: str = "./output") -> str:
        min_len = min(len(ground_truth), len(simulation))
        ground_truth = ground_truth[:min_len]
        simulation = simulation[:min_len]
        days = range(1, min_len + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(days, ground_truth, label='Ground Truth', color='black', linestyle='--', linewidth=2,
                 alpha=0.7)
        plt.plot(days, simulation, label='Simulation', color='#D93025', marker='o', linewidth=2.5, markersize=5)

        plt.axvspan(8, 12, color='orange', alpha=0.1, label='Outbreak Phase')
        plt.axvspan(19, min_len, color='gray', alpha=0.1, label='Compromise Phase')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Day', fontsize=12)
        plt.ylabel('Protest Proportion', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.4, linestyle=':')
        plt.ylim(-0.05, 1.05)

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = f"trend_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    @staticmethod
    def validate_peak_alignment(ground_truth: List[float], simulation: List[float]) -> Dict:
        min_len = min(len(ground_truth), len(simulation))
        gt_peak = int(np.argmax(ground_truth[:min_len])) + 1
        sim_peak = int(np.argmax(simulation[:min_len])) + 1
        return {"gt_peak_day": gt_peak, "sim_peak_day": sim_peak, "lag_days": sim_peak - gt_peak}

    @staticmethod
    def plot_trend_with_ci(ground_truth: List[float],
                           sim_matrix: np.ndarray,
                           title: str = "Simulated Trend with 95% Confidence Interval",
                           output_dir: str = "./output") -> str:
        """
        Plot trend comparison chart with confidence interval.

        Args:
            ground_truth: Ground truth data list [Days]
            sim_matrix: Simulation data matrix [Runs, Days] (e.g., 5 runs, 30 days, shape=(5,30))
        """
        # 1. Calculate statistics
        # Truncation and alignment
        min_len = min(len(ground_truth), sim_matrix.shape[1])
        ground_truth = np.array(ground_truth[:min_len])
        sim_matrix = sim_matrix[:, :min_len]

        days = range(1, min_len + 1)

        # Calculate mean and standard error
        sim_mean = np.mean(sim_matrix, axis=0)
        sim_std = np.std(sim_matrix, axis=0)
        n_runs = sim_matrix.shape[0]

        # 95% Confidence Interval (1.96 * SE)
        # If the number of runs is small (e.g., <10), min/max can be used instead of CI, or 1.0 * std
        ci_bound = 1.96 * sim_std / np.sqrt(n_runs)

        lower_bound = sim_mean - ci_bound
        upper_bound = sim_mean + ci_bound

        # 2. Plotting
        plt.figure(figsize=(12, 6))

        # Plot ground truth data
        plt.plot(days, ground_truth, label='Ground Truth',
                 color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=10)

        # Plot simulation mean line
        plt.plot(days, sim_mean, label=f'Simulation Mean (N={n_runs})',
                 color='#D93025', linewidth=2.5, marker='o', markersize=4, zorder=9)

        # Plot confidence interval shadow
        plt.fill_between(days, lower_bound, upper_bound,
                         color='#D93025', alpha=0.2,
                         label='95% Confidence Interval')

        # Label areas
        plt.axvspan(8, 12, color='orange', alpha=0.1, label='Outbreak Phase')
        plt.axvspan(19, min_len, color='gray', alpha=0.1, label='Compromise Phase')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (Day)', fontsize=12)
        plt.ylabel('Protest Content Proportion (Proportion)', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.4, linestyle=':')
        plt.ylim(-0.05, 1.05)

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = f"trend_ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path