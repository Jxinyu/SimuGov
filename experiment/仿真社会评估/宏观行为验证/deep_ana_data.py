import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# --- 配置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class MacroVisualizer:
    def __init__(self, json_path):
        self.data = self._load_data(json_path)
        self.output_dir = os.path.dirname(json_path)

    def _load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # =========================================================
    # 1. 不同政策对比：社会结构雷达图 (Radar Chart)
    # =========================================================
    def plot_social_structure_radar(self):
        """
        对比最极端的两个策略（放任 vs 严管）的 4 个宏观指标：
        1. 幂律拟合度 (R2) - 代表统计真实性
        2. 聚类系数 (Clustering) - 代表小圈子程度
        3. 同质性 (Homophily) - 代表回声室效应
        4. 基尼系数 (Gini) - 代表阶层固化
        """
        print("正在绘制社会结构雷达图...")

        # 遍历所有实验组，为每一组画一个雷达图
        for group_name, result in self.data.items():
            if "comparison" not in result or result["comparison"].get("status") != "success":
                continue

            # 1. 找出最极端的两个策略的名称
            comp = result["comparison"]
            policy_liberal = comp["policy_liberal"]
            policy_strict = comp["policy_strict"]

            # 2. 从 single_runs 中提取这两个策略的指标
            metrics_liberal = self._extract_metrics(result["single_runs"], policy_liberal)
            metrics_strict = self._extract_metrics(result["single_runs"], policy_strict)

            if not metrics_liberal or not metrics_strict:
                continue

            # 3. 准备绘图数据
            labels = ['幂律拟合(R²)', '聚类系数', '同质性(回声室)', '基尼系数(不平等)']

            # 提取数值 (注意顺序)
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

            # 4. 绘图
            N = len(labels)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # 闭合圆环

            values_liberal += values_liberal[:1]
            values_strict += values_strict[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            # 画线和填充
            ax.plot(angles, values_liberal, linewidth=2, linestyle='--', label='放任策略 (Liberal)', color='green')
            ax.fill(angles, values_liberal, 'green', alpha=0.1)

            ax.plot(angles, values_strict, linewidth=2, linestyle='-', label='严管策略 (Strict)', color='red')
            ax.fill(angles, values_strict, 'red', alpha=0.1)

            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=12)

            # 设置范围 (通常这些指标都在 0-1 之间)
            ax.set_ylim(0, 1.0)

            plt.title(f"【{group_name}】不同治理策略下的社会结构形态对比", size=15, y=1.05)
            plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))

            save_path = os.path.join(self.output_dir, f"{group_name}_radar.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  -> 已保存: {save_path}")

    def _extract_metrics(self, single_runs, policy_name):
        for run in single_runs:
            if run["policy"] == policy_name:
                return run["metrics"]
        return None

    # =========================================================
    # 2. 机制响应逻辑：KPI 剪刀差图 (Diverging Bar Chart)
    # =========================================================
    def plot_kpi_tradeoff(self):
        """
        可视化 '放任组 - 严管组' 的 KPI 差值。
        验证：安全性应大幅提升(严管高)，创造力应大幅下降(严管低)。
        """
        print("正在绘制 KPI 权衡图...")

        groups = []
        diff_safety = []
        diff_creativity = []
        diff_satisfaction = []

        for group_name, result in self.data.items():
            if "comparison" in result and result["comparison"].get("status") == "success":
                metrics = result["comparison"]["metrics"]
                groups.append(group_name)
                # 注意：这里我们取反，展示 "严管相对于放任的变化"
                # 原数据是 (Liberal - Strict)
                # 现在的逻辑：Strict - Liberal = -(Liberal - Strict)
                # 这样：安全性提升是正的，创造力下降是负的，更符合直觉
                diff_safety.append(-metrics["diff_safety"])
                diff_creativity.append(-metrics["diff_creativity"])
                diff_satisfaction.append(-metrics["diff_satisfaction"])

        if not groups:
            return

        x = np.arange(len(groups))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(x - width, diff_safety, width, label='安全性变化 (Safety)', color='#2ca02c')
        rects2 = ax.bar(x, diff_creativity, width, label='创造力变化 (Creativity)', color='#d62728')
        rects3 = ax.bar(x + width, diff_satisfaction, width, label='满意度变化 (Satisfaction)', color='#1f77b4')

        ax.set_ylabel('严管策略相对于放任策略的变化量')
        ax.set_title('机制逻辑验证：严管政策带来的非线性权衡 (Trade-off)')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()

        # 添加辅助线和说明
        plt.text(0, max(diff_safety) * 1.1, "预期：显著正值 (威慑有效)", ha='center', color='green', fontsize=9)
        plt.text(0, min(diff_creativity) * 1.1, "预期：显著负值 (寒蝉效应)", ha='center', color='red', fontsize=9)

        save_path = os.path.join(self.output_dir, "mechanism_tradeoff.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  -> 已保存: {save_path}")

    # =========================================================
    # 3. 相同政策对比：鲁棒性验证 (Robustness Check)
    # =========================================================
    def plot_policy_robustness(self):
        """
        寻找在不同实验组中出现的【同名政策】，对比其指标是否稳定。
        """
        print("正在绘制鲁棒性验证图...")

        # 1. 聚合数据： { "政策A": [ {metrics1}, {metrics2} ], ... }
        policy_collection = {}

        for group_name, result in self.data.items():
            for run in result["single_runs"]:
                p_name = run["policy"]
                if p_name not in policy_collection:
                    policy_collection[p_name] = []
                # 记录来源组和指标
                policy_collection[p_name].append({
                    "group": group_name,
                    "metrics": run["metrics"]
                })

        # 2. 筛选出运行次数 >= 2 的政策
        duplicates = {k: v for k, v in policy_collection.items() if len(v) > 1}

        if not duplicates:
            print("  ⚠️ 未发现重复运行的政策，跳过鲁棒性绘图。")
            return

        # 3. 为每个重复政策画图
        for p_name, runs in duplicates.items():
            # 提取要对比的指标
            groups = [r['group'] for r in runs]
            ginis = [r['metrics']['gini'] for r in runs]
            clusterings = [r['metrics']['clustering'] for r in runs]
            homophilies = [r['metrics']['homophily'] for r in runs]

            x = np.arange(len(groups))
            width = 0.2

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width, ginis, width, label='基尼系数', color='purple', alpha=0.7)
            ax.bar(x, clusterings, width, label='聚类系数', color='orange', alpha=0.7)
            ax.bar(x + width, homophilies, width, label='同质性', color='cyan', alpha=0.7)

            ax.set_ylabel('指标数值')
            ax.set_title(f'策略一致性验证：{p_name}\n(不同实验组下的结果波动)')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.set_ylim(0, 1.0)
            ax.legend()

            # 计算变异系数 (CV) 作为稳定性的定量描述
            cv_gini = np.std(ginis) / np.mean(ginis) if np.mean(ginis) != 0 else 0
            plt.figtext(0.5, -0.05, f"基尼系数波动率(CV): {cv_gini:.2%} (预期 < 10%)", ha="center", fontsize=10,
                        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

            save_path = os.path.join(self.output_dir, f"robustness_{p_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  -> 已保存: {save_path}")

    def run_all(self):
        self.plot_social_structure_radar()
        self.plot_kpi_tradeoff()
        self.plot_policy_robustness()


if __name__ == '__main__':
    # 指向 automation_summary.json 的路径
    # 请修改为你实际生成的路径
    SUMMARY_PATH = r'experiment\仿真社会评估\宏观行为验证\output\17669043098512447\automation_summary.json'

    # 也可以自动搜索最新的
    base_output = r'experiment\仿真社会评估\宏观行为验证\output'
    timestamps = sorted([d for d in os.listdir(base_output) if d.isdigit()], reverse=True)
    if timestamps:
        latest_path = os.path.join(base_output, timestamps[0], "automation_summary.json")
        if os.path.exists(latest_path):
            print(f"自动定位到最新结果: {latest_path}")
            viz = MacroVisualizer(latest_path)
            viz.run_all()
        else:
            print("未找到 summary json 文件。")
