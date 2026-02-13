import json
import re
import time
import os
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from ana_utils import *


def power_law_fit_data(file_path, day_time, output_file_path):
    """
    Construct data needed for power law distribution
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_contents.json', 'r', encoding='utf-8') as f:
        contents = json.load(f)

    data_values = []

    for content in contents:
        data_values.append((content['likes'] * 10) + (len(content['comments']) * 20) + content['views'])
    r_squared, alpha = verify_power_law_fit(data_values, output_file_path)
    #  - Alpha: Power law index, social networks are usually between 2.0~3.0. Small-scale social networks are usually between 0.5-1.5
    #  - R_squared: Goodness of fit (0~1), the closer to 1, the more it conforms to the power law.
    print(f"Power exponent alpha: {alpha}, R^2: {r_squared}")
    return alpha, r_squared


def compare_time_series_trends_data(file_path_1, file_path_2, day_time, output_file_path):
    """
    KPI time series trend comparison data construction
    :param file_path:
    :param day_time:
    :param output_file_path:
    :return:
    """
    with open(f'{file_path_1}/day_time_{day_time}/output_system_kpi.json', 'r', encoding='utf-8') as f:
        kpis_1 = json.load(f)
    with open(f'{file_path_2}/day_time_{day_time}/output_system_kpi.json', 'r', encoding='utf-8') as f:
        kpis_2 = json.load(f)

    safety = [kpis_1['safety'], kpis_2['safety']]
    creativity = [kpis_1['creativity'], kpis_2['creativity']]
    satisfaction = [kpis_1['satisfaction'], kpis_2['satisfaction']]

    safety_avg_diff = compare_time_series_trends(safety[0], safety[1], metric_name='Safety',
                                                 output_file_path=output_file_path)
    creativity_avg_diff = compare_time_series_trends(creativity[0], creativity[1], metric_name='Creativity',
                                                     output_file_path=output_file_path)
    satisfaction_avg_diff = compare_time_series_trends(satisfaction[0], satisfaction[1], metric_name='Satisfaction',
                                                       output_file_path=output_file_path)
    print(f"Average difference Safety: {safety_avg_diff}, Creativity: {creativity_avg_diff}, Satisfaction: {satisfaction_avg_diff}")
    return safety_avg_diff, creativity_avg_diff, satisfaction_avg_diff


def calculate_clustering_coefficient_data(file_path, day_time, output_file_path):
    """
    Network authenticity: clustering coefficient data construction
    :param file_path:
    :param day_time:
    :param output_file_path:
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_personas.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)
    edges = []
    for persona in personas:
        for move_agent, v in persona['social_relationships'].items():
            if v < 0:
                continue
            edges.append((persona['agent_id'], move_agent))

    # Calculate average clustering coefficient
    avg_clustering = calculate_clustering_coefficient(edges)
    print(f"Average clustering coefficient: {avg_clustering}")
    return avg_clustering


def calculate_homophily_score_data(file_path, day_time, output_file_path):
    """
    Social dynamics: homophily coefficient data construction
    :param file_path:
    :param day_time:
    :param output_file_path:
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_personas.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)
    edges = []
    node_attributes = {}
    standpoint_map = {0: 'Trust Faction', 1: 'Resistance Faction', 2: 'Neutral Faction'}
    # 2. Construct graph data
    for persona in personas:
        agent_id = persona['agent_id']

        # --- 1: Construct attribute dictionary ---
        max_idx = np.argmax(persona['standpoint'])
        node_attributes[agent_id] = standpoint_map[max_idx]
        for target_agent_id, strength in persona['social_relationships'].items():
            # Only when relationship strength > 0, regard it as a valid connection (peer relationship)
            if strength > 0:
                edges.append((agent_id, target_agent_id))

    # float: Assortativity coefficient (-1.0 ~ 1.0).
    #             - Positive value (>0): Homophily.
    #             - 0: Random connection.
    #             - Negative value (<0): Heterophily.
    score = calculate_homophily_score(edges, node_attributes)
    print(f"Assortativity coefficient: {score}")
    return score


def calculate_gini_coefficient_data(file_path, day_time, output_file_path):
    """
    Gini coefficient data construction
    :param file_path:
    :param day_time:
    :param output_file_path:
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_contents.json', 'r', encoding='utf-8') as f:
        contents = json.load(f)
    with open(f'{file_path}/day_time_{day_time}/output_personas.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)
    agent_score = {}
    for content in contents:
        agent_id = content['author_id']
        if agent_id not in agent_score.keys():
            influence = 0
            for persona in personas:
                if persona['agent_id'] == agent_id:
                    influence = persona['influence']
            agent_score[agent_id] = influence * 100
        agent_score[agent_id] += (content['views'] + content['likes'] * 4 + len(content['comments']) * 6 + content['shares'] * 12)
    wealth_distribution = agent_score.values()
    # Calculate Gini coefficient
    # float: Gini coefficient (0.0 ~ 1.0).
    #             - 0.0: Perfect equality.
    #             - 1.0: Absolute inequality.
    #             - Above 0.4: Usually considered as the inequality warning line.
    gini = calculate_gini_coefficient(wealth_distribution)
    print(f"Gini coefficient: {gini}")
    return gini


class ExperimentAutomator:
    def __init__(self, data_root_path: str, output_root_path: str):
        self.data_root = data_root_path
        self.output_root = output_root_path

        # Ensure output directory exists
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def _parse_policy_params(self, folder_name: str) -> Dict:
        """
        [Fully fixed version] Parse parameters in the policy folder name
        """
        try:
            # 1. Extract penalty term
            penalty_match = re.search(r'惩罚(\d+(?:_\d+)?)', folder_name)
            if penalty_match:
                # Process 0_01 -> 0.01 and remove any possible trailing underscores
                raw_p = penalty_match.group(1).replace('_', '.')
                if raw_p.endswith('.'): raw_p = raw_p[:-1]
                penalty = float(raw_p)
            else:
                penalty = 0.5

            # 2. Extract threshold term
            threshold_match = re.search(r'ai_threshold_(\d+(?:_\d+)?)', folder_name)
            if threshold_match:
                raw_t = threshold_match.group(1).replace('_', '.')
                if raw_t.endswith('.'): raw_t = raw_t[:-1]
                threshold = float(raw_t)
            else:
                threshold = 0.5

            # Calculate strictness score
            strictness_score = penalty + (1.0 - threshold)

            return {
                "folder_name": folder_name,
                "penalty": penalty,
                "threshold": threshold,
                "score": strictness_score
            }
        except Exception as e:
            print(f"⚠️ Parsing folder name failed: {folder_name}, error: {e}")
            return {"folder_name": folder_name, "score": 0}

    def _get_max_runtime_day(self, policy_path: str) -> int:
        """Get the maximum number of day_time_X within the policy folder"""
        max_day = 0
        if not os.path.exists(policy_path) or not os.path.isdir(policy_path):
            return 0

        for item in os.listdir(policy_path):
            if item.startswith("day_time_"):
                try:
                    day_num = int(item.split("_")[-1])
                    if day_num > max_day:
                        max_day = day_num
                except:
                    continue
        return max_day

    def _find_extreme_pair(self, policies: List[Dict]) -> Tuple[Dict, Dict]:
        """Filter out the most liberal and most strict policy pairs"""
        if len(policies) < 2:
            return None, None
        sorted_p = sorted(policies, key=lambda x: x['score'])
        return sorted_p[0], sorted_p[-1]

    def _process_group(self, group_name: str, group_path: str):
        """Process all policies under an experimental group"""
        group_results = {"single_runs": [], "comparison": {}}
        parsed_policies = []

        # Get all folders under the current group
        policy_folders = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]

        for p_folder in policy_folders:
            p_full_path = os.path.join(group_path, p_folder)
            run_day = self._get_max_runtime_day(p_full_path)

            # If run_day is 0, it indicates this layer is not a policy folder (it might be day_time_X itself), skip
            if run_day == 0:
                continue

            print(f"  👉 Analyzing policy: {p_folder} (Max days: {run_day})")

            # Create output sub-directory
            out_dir = os.path.join(self.output_root, group_name, p_folder)
            os.makedirs(out_dir, exist_ok=True)

            p_info = self._parse_policy_params(p_folder)
            p_info.update({'full_path': p_full_path, 'run_day': run_day})
            parsed_policies.append(p_info)

            metrics = {}
            try:
                # 1. Power law
                alpha, r2 = power_law_fit_data(p_full_path, run_day, out_dir + "/power_law.json")
                metrics["power_law"] = {"alpha": float(alpha), "r2": float(r2)}
                # 2. Clustering
                c = calculate_clustering_coefficient_data(p_full_path, run_day, out_dir + "/clustering.json")
                metrics["clustering"] = float(c)
                # 3. Homophily
                h = calculate_homophily_score_data(p_full_path, run_day, out_dir + "/homophily.json")
                metrics["homophily"] = float(h)
                # 4. Gini
                g = calculate_gini_coefficient_data(p_full_path, run_day, out_dir + "/gini.json")
                metrics["gini"] = float(g)

                status = "success"
            except Exception as e:
                print(f"     ❌ Indicator calculation failed: {e}")
                status = "error"
                metrics = str(e)

            group_results["single_runs"].append({
                "policy": p_folder,
                "days": run_day,
                "status": status,
                "metrics": metrics
            })

        # Comparison validation
        liberal, strict = self._find_extreme_pair(parsed_policies)
        if liberal and strict:
            print(f"  ⚖️  Performing extreme comparison: {liberal['folder_name']} VS {strict['folder_name']}")
            try:
                day = min(liberal['run_day'], strict['run_day'])
                comp_out = os.path.join(self.output_root, group_name, f"comparison_report.json")
                d_safe, d_cre, d_sat = compare_time_series_trends_data(liberal['full_path'], strict['full_path'], day,
                                                                       comp_out)
                group_results["comparison"] = {
                    "status": "success",
                    "policy_liberal": liberal['folder_name'],
                    "policy_strict": strict['folder_name'],
                    "metrics": {"diff_safety": float(d_safe), "diff_creativity": float(d_cre),
                                "diff_satisfaction": float(d_sat)}
                }
            except Exception as e:
                group_results["comparison"] = {"status": "error", "msg": str(e)}

        return group_results

    def run(self):
        """Core running logic: automatically detect directory depth"""
        summary_report = {}

        # Get the first-level sub-directories under the root directory
        top_sub_dirs = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]

        # Detection: if there is day_time_X directly under the first-level sub-directory, it indicates a 2-layer structure
        is_2_layer = any(self._get_max_runtime_day(os.path.join(self.data_root, d)) > 0 for d in top_sub_dirs)

        if is_2_layer:
            print("🚀 Detected 2-layer structure (Root directory -> Policy directory)")
            summary_report["Direct_Base"] = self._process_group("Direct_Base", self.data_root)
        else:
            print("🚀 Detected 3-layer structure (Root directory -> Experimental group -> Policy directory)")
            for group in top_sub_dirs:
                print(f"\n📂 Processing experimental group: {group}")
                group_path = os.path.join(self.data_root, group)
                summary_report[group] = self._process_group(group, group_path)

        # Save summary report
        final_summary_path = os.path.join(self.output_root, "automation_summary.json")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        print(f"\n✅ All automated tasks completed! Summary report at: {final_summary_path}")


def full_auto_main():
    # Configure your root directory
    DATA_ROOT = r'experiment\Simulated social assessment\Macro behavioral verification\data\case_policy'

    # Output directory
    OUTPUT_ROOT = fr'experiment\Simulated social assessment\Macro behavioral verification\output\{str(time.time()).split(".")[0]}'

    runner = ExperimentAutomator(DATA_ROOT, OUTPUT_ROOT)
    runner.run()


if __name__ == '__main__':
    full_auto_main()