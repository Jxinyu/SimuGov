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
    构建幂律分布需要的数据
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_contents.json', 'r', encoding='utf-8') as f:
        contents = json.load(f)

    data_values = []

    for content in contents:
        data_values.append((content['likes'] * 10) + (len(content['comments']) * 20) + content['views'])
    r_squared, alpha = verify_power_law_fit(data_values, output_file_path)
    #  - Alpha: 幂律指数，社交网络通常在 2.0~3.0 之间。小规模社交网络通常在0.5-1.5
    #  - R_squared: 拟合优度 (0~1)，越接近 1 表示越符合幂律。
    print(f"幂指数 alpha: {alpha}, R^2: {r_squared}")
    return alpha, r_squared


def compare_time_series_trends_data(file_path_1, file_path_2, day_time, output_file_path):
    """
    KPI 时序趋势对比  数据构建
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
    print(f"平均差 Safety: {safety_avg_diff}, Creativity: {creativity_avg_diff}, Satisfaction: {satisfaction_avg_diff}")
    return safety_avg_diff, creativity_avg_diff, satisfaction_avg_diff


def calculate_clustering_coefficient_data(file_path, day_time, output_file_path):
    """
    网络真实性：聚类系数  数据构建
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

    # 计算 平均聚类系数
    avg_clustering = calculate_clustering_coefficient(edges)
    print(f"平均聚类系数: {avg_clustering}")
    return avg_clustering


def calculate_homophily_score_data(file_path, day_time, output_file_path):
    """
    社会动力学：同质性系数  数据构建
    :param file_path:
    :param day_time:
    :param output_file_path:
    :return:
    """
    with open(f'{file_path}/day_time_{day_time}/output_personas.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)
    edges = []
    node_attributes = {}
    standpoint_map = {0: '信任派', 1: '反抗派', 2: '中立派'}
    # 2. 构建图数据
    for persona in personas:
        agent_id = persona['agent_id']

        # --- 1：构建属性字典 ---
        max_idx = np.argmax(persona['standpoint'])
        node_attributes[agent_id] = standpoint_map[max_idx]
        for target_agent_id, strength in persona['social_relationships'].items():
            # 只有当关系强度 > 0 时，才视为有效连接（同伴关系）
            if strength > 0:
                edges.append((agent_id, target_agent_id))

    # float: 同配系数 (-1.0 ~ 1.0)。
    #             - 正值 (>0): 同类相吸 (Homophily)。
    #             - 0: 随机连接。
    #             - 负值 (<0): 异类相吸 (Heterophily)。
    score = calculate_homophily_score(edges, node_attributes)
    print(f"同配系数: {score}")
    return score


def calculate_gini_coefficient_data(file_path, day_time, output_file_path):
    """
    基尼系数  数据构建
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
    # 计算 Gini 系数
    # float: 基尼系数 (0.0 ~ 1.0)。
    #             - 0.0: 完全平等。
    #             - 1.0: 绝对不平等。
    #             - 0.4以上: 通常被认为是不平等警戒线。
    gini = calculate_gini_coefficient(wealth_distribution)
    print(f"基尼系数: {gini}")
    return gini


class ExperimentAutomator:
    def __init__(self, data_root_path: str, output_root_path: str):
        self.data_root = data_root_path
        self.output_root = output_root_path

        # 确保输出目录存在
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def _parse_policy_params(self, folder_name: str) -> Dict:
        """
        [完全修复版] 解析政策文件夹名称中的参数
        """
        try:
            # 1. 提取惩罚项
            penalty_match = re.search(r'惩罚(\d+(?:_\d+)?)', folder_name)
            if penalty_match:
                # 处理 0_01 -> 0.01 并移除末尾可能存在的下划线
                raw_p = penalty_match.group(1).replace('_', '.')
                if raw_p.endswith('.'): raw_p = raw_p[:-1]
                penalty = float(raw_p)
            else:
                penalty = 0.5

            # 2. 提取阈值项
            threshold_match = re.search(r'ai_threshold_(\d+(?:_\d+)?)', folder_name)
            if threshold_match:
                raw_t = threshold_match.group(1).replace('_', '.')
                if raw_t.endswith('.'): raw_t = raw_t[:-1]
                threshold = float(raw_t)
            else:
                threshold = 0.5

            # 计算严管程度得分
            strictness_score = penalty + (1.0 - threshold)

            return {
                "folder_name": folder_name,
                "penalty": penalty,
                "threshold": threshold,
                "score": strictness_score
            }
        except Exception as e:
            print(f"⚠️ 解析文件夹名失败: {folder_name}, 错误: {e}")
            return {"folder_name": folder_name, "score": 0}

    def _get_max_runtime_day(self, policy_path: str) -> int:
        """获取该策略文件夹内 day_time_X 的最大数字"""
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
        """筛选出最放任和最严管的策略对"""
        if len(policies) < 2:
            return None, None
        sorted_p = sorted(policies, key=lambda x: x['score'])
        return sorted_p[0], sorted_p[-1]

    def _process_group(self, group_name: str, group_path: str):
        """处理一个实验组下的所有政策"""
        group_results = {"single_runs": [], "comparison": {}}
        parsed_policies = []

        # 获取当前组下的所有文件夹
        policy_folders = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]

        for p_folder in policy_folders:
            p_full_path = os.path.join(group_path, p_folder)
            run_day = self._get_max_runtime_day(p_full_path)

            # 如果 run_day 为 0，说明这一层不是政策文件夹（可能是 day_time_X 本身），跳过
            if run_day == 0:
                continue

            print(f"  👉 正在分析政策: {p_folder} (最大天数: {run_day})")

            # 创建输出子目录
            out_dir = os.path.join(self.output_root, group_name, p_folder)
            os.makedirs(out_dir, exist_ok=True)

            p_info = self._parse_policy_params(p_folder)
            p_info.update({'full_path': p_full_path, 'run_day': run_day})
            parsed_policies.append(p_info)

            metrics = {}
            try:
                # 1. 幂律
                alpha, r2 = power_law_fit_data(p_full_path, run_day, out_dir + "/power_law.json")
                metrics["power_law"] = {"alpha": float(alpha), "r2": float(r2)}
                # 2. 聚类
                c = calculate_clustering_coefficient_data(p_full_path, run_day, out_dir + "/clustering.json")
                metrics["clustering"] = float(c)
                # 3. 同质性
                h = calculate_homophily_score_data(p_full_path, run_day, out_dir + "/homophily.json")
                metrics["homophily"] = float(h)
                # 4. 基尼
                g = calculate_gini_coefficient_data(p_full_path, run_day, out_dir + "/gini.json")
                metrics["gini"] = float(g)

                status = "success"
            except Exception as e:
                print(f"     ❌ 指标计算失败: {e}")
                status = "error"
                metrics = str(e)

            group_results["single_runs"].append({
                "policy": p_folder,
                "days": run_day,
                "status": status,
                "metrics": metrics
            })

        # 对比验证
        liberal, strict = self._find_extreme_pair(parsed_policies)
        if liberal and strict:
            print(f"  ⚖️  进行极端对比: {liberal['folder_name']} VS {strict['folder_name']}")
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
        """核心运行逻辑：自动探测目录深度"""
        summary_report = {}

        # 获取根目录下的一级子目录
        top_sub_dirs = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]

        # 探测：如果一级子目录下直接有 day_time_X，说明是 2 层结构
        is_2_layer = any(self._get_max_runtime_day(os.path.join(self.data_root, d)) > 0 for d in top_sub_dirs)

        if is_2_layer:
            print("🚀 检测到 2 层结构 (根目录 -> 政策目录)")
            summary_report["Direct_Base"] = self._process_group("Direct_Base", self.data_root)
        else:
            print("🚀 检测到 3 层结构 (根目录 -> 实验组 -> 政策目录)")
            for group in top_sub_dirs:
                print(f"\n📂 正在处理实验组: {group}")
                group_path = os.path.join(self.data_root, group)
                summary_report[group] = self._process_group(group, group_path)

        # 保存汇总报告
        final_summary_path = os.path.join(self.output_root, "automation_summary.json")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 自动化任务全部完成！汇总报告见: {final_summary_path}")


def full_auto_main():
    # 配置你的根目录
    DATA_ROOT = r'experiment\仿真社会评估\宏观行为验证\data\case_policy'

    # 输出目录
    OUTPUT_ROOT = fr'experiment\仿真社会评估\宏观行为验证\output\{str(time.time()).split(".")[0]}'

    runner = ExperimentAutomator(DATA_ROOT, OUTPUT_ROOT)
    runner.run()


if __name__ == '__main__':
    full_auto_main()