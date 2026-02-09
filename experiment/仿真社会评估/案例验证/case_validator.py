import os
from datetime import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from typing import List, Dict, Union

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CaseValidator:
    """
    ArtStation 案例验证专用评估工具箱
    包含自动评级和详细指标解释。
    """

    @staticmethod
    def _rate_correlation(value: float) -> str:
        """辅助函数：对相关系数进行评级"""
        v = abs(value)
        if v >= 0.8: return "极强"
        if v >= 0.6: return "强"
        if v >= 0.4: return "中"
        return "弱/无相关"

    @staticmethod
    def _rate_mae(value: float) -> str:
        """辅助函数：对MAE误差进行评级"""
        if value <= 0.1: return "极优"
        if value <= 0.2: return "良好"
        if value <= 0.3: return "一般"
        return "偏差大"

    @staticmethod
    def validate_trend_correlation(ground_truth: List[float], simulation: List[float]) -> Dict:
        """
        【维度一：趋势一致性验证】

        参数作用说明：
        1. Pearson (皮尔逊): 衡量【线性】拟合度。数值越高，说明仿真曲线与真实曲线的形状越“重合”。
        2. Spearman (斯皮尔曼): 衡量【秩/排名】一致性。不关心绝对值，只关心“涨跌趋势”是否同步。对非线性关系更鲁棒。
        3. MAE (平均绝对误差): 衡量【数值】偏移量。越低越好，表示仿真值与真实值在绝对数量上很接近。

        评级标准 (Pearson/Spearman):
        - < 0.4: ❌ 差 (模型失效)
        - 0.4 - 0.6: ⚠️ 中 (勉强及格)
        - 0.6 - 0.8: ✅ 优 (学术界公认有效)
        - > 0.8: 🌟 SOTA (极高精度拟合)
        """
        # 1. 数据对齐
        if len(ground_truth) != len(simulation):
            min_len = min(len(ground_truth), len(simulation))
            ground_truth = ground_truth[:min_len]
            simulation = simulation[:min_len]

        # 2. 计算指标
        if np.std(ground_truth) == 0 or np.std(simulation) == 0:
            p_corr, s_corr = 0.0, 0.0
        else:
            p_corr, _ = pearsonr(ground_truth, simulation)
            s_corr, _ = spearmanr(ground_truth, simulation)

        mae = mean_absolute_error(ground_truth, simulation)

        # 3. 获取评级
        p_rating = CaseValidator._rate_correlation(p_corr)
        s_rating = CaseValidator._rate_correlation(s_corr)
        m_rating = CaseValidator._rate_mae(mae)

        # 4. 打印详细报告
        print("\n" + "-" * 60)
        print("[维度一] 趋势一致性验证报告")
        print("-" * 60)
        print(f"1. Pearson (线性拟合度): {p_corr:.4f}  =>  {p_rating}")
        print(f"   * 解释: 仿真曲线是否精确复现了历史曲线的起伏形态。")
        print(f"2. Spearman (趋势同步性): {s_corr:.4f}  =>  {s_rating}")
        print(f"   * 解释: 仿真是否捕捉到了“何时涨、何时跌”的单调性规律。")
        print(f"3. MAE (数值平均误差):    {mae:.4f}    =>  {m_rating}")
        print(f"   * 解释: 平均每天的占比预测偏差量 (例如 0.05 代表平均偏差 5%)。")
        print("-" * 60)

        return {
            "pearson": {"value": round(p_corr, 4), "rating": p_rating},
            "spearman": {"value": round(s_corr, 4), "rating": s_rating},
            "mae": {"value": round(mae, 4), "rating": m_rating}
        }

    @staticmethod
    def validate_mechanism_causality(satisfaction_curve: List[float], protest_curve: List[float]) -> Dict:
        """
        【维度三：机制有效性验证】

        参数作用说明：
        1. Correlation (相关性): 衡量【满意度】与【抗议占比】之间的关系。
        2. 目标方向: 必须是【负相关】(Negative Correlation)。即满意度越低，抗议越多。

        评级标准 (相关系数 r):
        - > -0.3: ❌ 无效 (无因果关系，抗议可能是随机发生的)
        - -0.3 ~ -0.5: ⚠️ 弱 (有一定关系，但不显著)
        - -0.5 ~ -0.7: ✅ 有效 (存在显著的因果链条)
        - < -0.7: 🌟 强鲁棒性 (证明抗议完全由情绪驱动，涌现机制非常坚实)
        """
        min_len = min(len(satisfaction_curve), len(protest_curve))
        if min_len < 2:
            corr = 0.0
        else:
            corr, _ = pearsonr(satisfaction_curve[:min_len], protest_curve[:min_len])

        # 评级逻辑 (针对负相关)
        if corr > -0.3:
            rating = "无效/随机"
        elif corr > -0.5:
            rating = "弱相关"
        elif corr > -0.7:
            rating = "有效"
        else:
            rating = "强鲁棒性"

        # 打印详细报告
        print("\n" + "-" * 60)
        print("[维度三] 机制因果性验证报告")
        print("-" * 60)
        print(f"指标: 满意度-抗议负相关系数")
        print(f"数值: {corr:.4f}")
        print(f"评级: {rating}")
        print(f"判定依据:")
        print(f"   - 只有出现显著负相关 (r < -0.5)，才能证明“抗议”是由“不满”驱动的涌现现象。")
        print(f"   - 若接近 0，说明抗议是随机注入的，仿真失败。")
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
        plt.plot(days, ground_truth, label='Ground Truth (真实历史)', color='black', linestyle='--', linewidth=2,
                 alpha=0.7)
        plt.plot(days, simulation, label='Simulation (仿真)', color='#D93025', marker='o', linewidth=2.5, markersize=5)

        plt.axvspan(8, 12, color='orange', alpha=0.1, label='爆发期')
        plt.axvspan(19, min_len, color='gray', alpha=0.1, label='妥协期')

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
        绘制带置信区间的趋势对比图。

        Args:
            ground_truth: 真实数据列表 [Days]
            sim_matrix: 仿真数据矩阵 [Runs, Days] (例如 5次运行，30天，shape=(5,30))
        """
        # 1. 计算统计量
        # 截断对齐
        min_len = min(len(ground_truth), sim_matrix.shape[1])
        ground_truth = np.array(ground_truth[:min_len])
        sim_matrix = sim_matrix[:, :min_len]

        days = range(1, min_len + 1)

        # 计算均值和标准误差
        sim_mean = np.mean(sim_matrix, axis=0)
        sim_std = np.std(sim_matrix, axis=0)
        n_runs = sim_matrix.shape[0]

        # 95% 置信区间 (1.96 * SE)
        # 如果运行次数少(如<10)，可以用 min/max 代替 CI，或者用 1.0 * std
        ci_bound = 1.96 * sim_std / np.sqrt(n_runs)

        lower_bound = sim_mean - ci_bound
        upper_bound = sim_mean + ci_bound

        # 2. 绘图
        plt.figure(figsize=(12, 6))

        # 绘制真实数据
        plt.plot(days, ground_truth, label='Ground Truth (真实历史)',
                 color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=10)

        # 绘制仿真均值线
        plt.plot(days, sim_mean, label=f'Simulation Mean (N={n_runs})',
                 color='#D93025', linewidth=2.5, marker='o', markersize=4, zorder=9)

        # 绘制置信区间阴影
        plt.fill_between(days, lower_bound, upper_bound,
                         color='#D93025', alpha=0.2,
                         label='95% Confidence Interval')

        # 标注区域
        plt.axvspan(8, 12, color='orange', alpha=0.1, label='爆发期')
        plt.axvspan(19, min_len, color='gray', alpha=0.1, label='妥协期')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间 (Day)', fontsize=12)
        plt.ylabel('抗议内容占比 (Proportion)', fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.4, linestyle=':')
        plt.ylim(-0.05, 1.05)

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = f"trend_ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
