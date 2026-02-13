import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

plt.rcParams['font.sans-serif'] = ['Arial']  # Used to display labels normally
plt.rcParams['axes.unicode_minus'] = False  # Used to display minus signs normally


class MacroVisualizer:
    def __init__(self, json_path):
        self.data = self._load_data(json_path)
        self.output_dir = os.path.dirname(json_path)

    def _load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_social_structure_radar(self):
        """
        Compare 4 macro indicators of the two most extreme strategies (Liberal vs Strict):
        1. Power law goodness of fit (R2) - represents statistical authenticity
        2. Clustering coefficient (Clustering) - represents the degree of cliques
        3. Homophily (Homophily) - represents the echo chamber effect
        4. Gini coefficient (Gini) - represents class solidification
        """
        print("Drawing social structure radar chart...")

        # Traverse all experimental groups and draw a radar chart for each group
        for group_name, result in self.data.items():
            if "comparison" not in result or result["comparison"].get("status") != "success":
                continue

            # 1. Find the names of the two most extreme strategies
            comp = result["comparison"]
            policy_liberal = comp["policy_liberal"]
            policy_strict = comp["policy_strict"]

            # 2. Extract the indicators of these two strategies from single_runs
            metrics_liberal = self._extract_metrics(result["single_runs"], policy_liberal)
            metrics_strict = self._extract_metrics(result["single_runs"], policy_strict)

            if not metrics_liberal or not metrics_strict:
                continue

            # 3. Prepare plotting data
            labels = ['Power Law Fit (R²)', 'Clustering Coefficient', 'Homophily (Echo Chamber)', 'Gini Coefficient (Inequality)']

            # Extract values (note the order)
            values_liberal = [
                metrics_liberal['power_law']['r2'],
                metrics_liberal['clustering'],
                metrics_liberal['homophily'],
                metrics_liberal['gini']
            ]
            values_strict = [
                metrics_strict['power_law']['r2'],
                metrics_strict['clustering'],
                metrics_strict['homophily'],
                metrics_strict['gini']
            ]

            # 4. Plotting
            N = len(labels)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Closed loop

            values_liberal += values_liberal[:1]
            values_strict += values_strict[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            # Draw lines and fill
            ax.plot(angles, values_liberal, linewidth=2, linestyle='--', label='Liberal Strategy (Liberal)', color='green')
            ax.fill(angles, values_liberal, 'green', alpha=0.1)

            ax.plot(angles, values_strict, linewidth=2, linestyle='-', label='Strict Strategy (Strict)', color='red')
            ax.fill(angles, values_strict, 'red', alpha=0.1)

            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=12)

            # Set range (usually these indicators are between 0 and 1)
            ax.set_ylim(0, 1.0)

            plt.title(f"[{group_name}] Social structure pattern comparison under different governance strategies", size=15, y=1.05)
            plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))

            save_path = os.path.join(self.output_dir, f"{group_name}_radar.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  -> Saved: {save_path}")

    def _extract_metrics(self, single_runs, policy_name):
        for run in single_runs:
            if run["policy"] == policy_name:
                return run["metrics"]
        return None

    def plot_kpi_tradeoff(self):
        """
        Visualize the KPI difference of 'Liberal Group - Strict Group'.
        Verify: Safety should increase significantly (high in Strict), creativity should decrease significantly (low in Strict).
        """
        print("Drawing KPI tradeoff chart...")

        groups = []
        diff_safety = []
        diff_creativity = []
        diff_satisfaction = []

        for group_name, result in self.data.items():
            if "comparison" in result and result["comparison"].get("status") == "success":
                metrics = result["comparison"]["metrics"]
                groups.append(group_name)
                # Note: Here we negate to show "the change of Strict relative to Liberal"
                # The original data is (Liberal - Strict)
                # Current logic: Strict - Liberal = -(Liberal - Strict)
                # In this way: the increase in safety is positive, and the decrease in creativity is negative, which is more intuitive
                diff_safety.append(-metrics["diff_safety"])
                diff_creativity.append(-metrics["diff_creativity"])
                diff_satisfaction.append(-metrics["diff_satisfaction"])

        if not groups:
            return

        x = np.arange(len(groups))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(x - width, diff_safety, width, label='Safety Change (Safety)', color='#2ca02c')
        rects2 = ax.bar(x, diff_creativity, width, label='Creativity Change (Creativity)', color='#d62728')
        rects3 = ax.bar(x + width, diff_satisfaction, width, label='Satisfaction Change (Satisfaction)', color='#1f77b4')

        ax.set_ylabel('Change amount of Strict strategy relative to Liberal strategy')
        ax.set_title('Mechanism logic verification: Nonlinear tradeoff brought by strict policy (Trade-off)')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()

        # Add auxiliary lines and descriptions
        plt.text(0, max(diff_safety) * 1.1, "Expectation: Significant positive value (effective deterrence)", ha='center', color='green', fontsize=9)
        plt.text(0, min(diff_creativity) * 1.1, "Expectation: Significant negative value (chilling effect)", ha='center', color='red', fontsize=9)

        save_path = os.path.join(self.output_dir, "mechanism_tradeoff.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  -> Saved: {save_path}")

    def plot_policy_robustness(self):
        """
        Search for [same-name policies] appearing in different experimental groups and compare if their indicators are stable.
        """
        print("Drawing robustness validation chart...")

        # 1. Aggregate data: { "Policy A": [ {metrics1}, {metrics2} ], ... }
        policy_collection = {}

        for group_name, result in self.data.items():
            for run in result["single_runs"]:
                p_name = run["policy"]
                if p_name not in policy_collection:
                    policy_collection[p_name] = []
                # Record source groups and indicators
                policy_collection[p_name].append({
                    "group": group_name,
                    "metrics": run["metrics"]
                })

        # 2. Filter policies with run count >= 2
        duplicates = {k: v for k, v in policy_collection.items() if len(v) > 1}

        if not duplicates:
            print("  ⚠️ No duplicate running policies found, skipping robustness plotting.")
            return

        # 3. Draw chart for each duplicate policy
        for p_name, runs in duplicates.items():
            # Extract indicators to compare
            groups = [r['group'] for r in runs]
            ginis = [r['metrics']['gini'] for r in runs]
            clusterings = [r['metrics']['clustering'] for r in runs]
            homophilies = [r['metrics']['homophily'] for r in runs]

            x = np.arange(len(groups))
            width = 0.2

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width, ginis, width, label='Gini Coefficient', color='purple', alpha=0.7)
            ax.bar(x, clusterings, width, label='Clustering Coefficient', color='orange', alpha=0.7)
            ax.bar(x + width, homophilies, width, label='Homophily', color='cyan', alpha=0.7)

            ax.set_ylabel('Indicator value')
            ax.set_title(f'Strategy consistency verification: {p_name}\n(Result fluctuation under different experimental groups)')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.set_ylim(0, 1.0)
            ax.legend()

            # Calculate coefficient of variation (CV) as a quantitative description of stability
            cv_gini = np.std(ginis) / np.mean(ginis) if np.mean(ginis) != 0 else 0
            plt.figtext(0.5, -0.05, f"Gini coefficient fluctuation rate (CV): {cv_gini:.2%} (Expectation < 10%)", ha="center", fontsize=10,
                        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

            save_path = os.path.join(self.output_dir, f"robustness_{p_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  -> Saved: {save_path}")

    def run_all(self):
        self.plot_social_structure_radar()
        self.plot_kpi_tradeoff()
        self.plot_policy_robustness()


if __name__ == '__main__':
    # Path pointing to automation_summary.json
    # Please modify to the path you actually generated
    SUMMARY_PATH = r'experiment\simulation_social_assessment\macro_behavior_validation\output\17669043098512447\automation_summary.json'

    # Can also search for the latest automatically
    base_output = r'experiment\simulation_social_assessment\macro_behavior_validation\output'
    timestamps = sorted([d for d in os.listdir(base_output) if d.isdigit()], reverse=True)
    if timestamps:
        latest_path = os.path.join(base_output, timestamps[0], "automation_summary.json")
        if os.path.exists(latest_path):
            print(f"Automatically positioned to the latest result: {latest_path}")
            viz = MacroVisualizer(latest_path)
            viz.run_all()
        else:
            print("Summary json file not found.")