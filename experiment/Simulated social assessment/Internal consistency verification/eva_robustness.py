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

# 设置绘图风格和中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def evaluate_monte_carlo_experiment(
        all_runs_data: List[Dict[str, List[float]]],
        output_dir: str,
        cv_threshold: float = 0.20
):
    """
    评估蒙特卡洛模拟的鲁棒性与收敛性。
    通过计算多次运行的变异系数 (CV) 和置信区间，判断系统结果是否稳定收敛。
    并输出绘图所需的详细统计数据。
    """

    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 数据重组与预处理
    metrics = ['safety', 'satisfaction', 'creativity']
    processed_data = {m: [] for m in metrics}

    # 获取最小天数，防止不同运行长度不一致导致的崩溃
    min_days = min(len(run['safety']) for run in all_runs_data)

    for run in all_runs_data:
        for m in metrics:
            # 截取到最小长度
            processed_data[m].append(run[m][:min_days])

    # 转为 numpy 矩阵: shape = (N_runs, min_days)
    for m in metrics:
        processed_data[m] = np.array(processed_data[m])

    # 3. 统计计算函数
    def calculate_stats(data_matrix):
        """计算均值、置信区间、最终状态的变异系数"""
        N = data_matrix.shape[0]
        # 时间维度上的统计
        mean_series = np.mean(data_matrix, axis=0)
        std_series = np.std(data_matrix, axis=0)
        sem_series = stats.sem(data_matrix, axis=0)  # 标准误差

        # 95% 置信区间
        ci_h = sem_series * stats.t.ppf((1 + 0.95) / 2., N - 1)
        lower_bound = mean_series - ci_h
        upper_bound = mean_series + ci_h

        # 计算变异系数 (CV) - 基于最终状态 (Final Day)
        final_values = data_matrix[:, -1]
        final_mean = np.mean(final_values)
        final_std = np.std(final_values)

        # 防止除以0
        if final_mean == 0:
            cv = 0.0
        else:
            cv = final_std / final_mean

        return mean_series, lower_bound, upper_bound, cv

    # 执行计算
    safe_mean, safe_low, safe_high, safe_cv = calculate_stats(processed_data['safety'])
    sat_mean, sat_low, sat_high, sat_cv = calculate_stats(processed_data['satisfaction'])
    creat_mean, creat_low, creat_high, creat_cv = calculate_stats(processed_data['creativity'])

    # 4. 判定逻辑
    is_passed = (safe_cv < cv_threshold) and (sat_cv < cv_threshold) and (creat_cv < cv_threshold)

    # 5. 绘制蒙特卡洛云图 (Shadow Plot)
    days = list(range(1, min_days + 1))  # 转为list方便后续JSON序列化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- 绘图辅助函数 ---
    def plot_cloud(ax, matrix, mean, lower, upper, title, color):
        # 绘制每一条单独的轨迹 (Ghost Lines)
        for i in range(matrix.shape[0]):
            ax.plot(days, matrix[i, :], color='gray', alpha=0.15, linewidth=1)

        # 绘制均值线
        ax.plot(days, mean, color=color, linewidth=2.5, label='均值 (Mean)')

        # 绘制置信区间 (Shadow)
        ax.fill_between(days, lower, upper, color=color, alpha=0.25, label='95% 置信区间')

        ax.set_title(title, fontsize=14)
        ax.set_ylabel('KPI 值')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 绘制 Safety
    plot_cloud(ax1, processed_data['safety'], safe_mean, safe_low, safe_high,
               f'Safety 收敛性分析 (CV={safe_cv:.4f})', 'blue')

    # 绘制 Satisfaction
    plot_cloud(ax2, processed_data['satisfaction'], sat_mean, sat_low, sat_high,
               f'Satisfaction 收敛性分析 (CV={sat_cv:.4f})', 'green')

    # 绘制 Creativity
    plot_cloud(ax3, processed_data['creativity'], creat_mean, creat_low, creat_high,
               f'Creativity 收敛性分析 (CV={creat_cv:.4f})', 'yellow')

    ax3.set_xlabel('仿真天数 (Days)')

    # 添加阈值线说明
    status_text = "鲁棒性验证通过 (收敛)" if is_passed else "鲁棒性验证警告 (发散)"
    fig.suptitle(f'蒙特卡洛模拟结果 (N={len(all_runs_data)}): {status_text}',
                 fontsize=16, fontweight='bold', color='green' if is_passed else '#d62728')

    plot_save_path = os.path.join(output_dir, 'monte_carlo_robustness.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 打印报告
    print("\n" + "=" * 50)
    print(" >>> 蒙特卡洛鲁棒性验证报告 (Monte Carlo Report) <<<")
    print("=" * 50)
    print(f"运行次数 (N): {len(all_runs_data)}")
    print(f"验证状态: {status_text}")
    print(f"CV 阈值标准: < {cv_threshold}")
    print("-" * 30)
    print(f"[安全性 Safety]")
    print(f"  - 最终均值: {safe_mean[-1]:.4f}")
    print(f"  - 变异系数 (CV): {safe_cv:.4f}")
    print("-" * 30)
    print(f"[满意度 Satisfaction]")
    print(f"  - 最终均值: {sat_mean[-1]:.4f}")
    print(f"  - 变异系数 (CV): {sat_cv:.4f}")
    print("=" * 50 + "\n")

    # 7. 保存结果摘要 (result.json)
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
    # 8. [新增] 保存绘图源数据 (monte_carlo_plotting_data.json)
    #    包含每一天的均值、置信区间上下界，方便后续重绘
    # =================================================================
    plotting_data = {
        "meta": {
            "n_runs": len(all_runs_data),
            "days_count": len(days),
            # 【修复点】：使用 bool() 将 numpy.bool_ 强转为 python bool
            "is_passed": bool(is_passed)
        },
        "axis": {
            "days": days
        },
        "metrics": {
            "safety": {
                # .tolist() 已经处理了 numpy array 到 list 的转换
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

    print(f"✅ 结果摘要已保存至: {result_json_path}")
    print(f"✅ 绘图源数据已保存至: {data_save_path}")


def load_robustness_kpi_data(root_path: str) -> List[Dict[str, List[float]]]:
    """
    从 eva_robustness 目录加载多次运行的 KPI 数据。
    自动寻找每个运行文件夹下“最后一天”的 output_system_kpi.json。
    """
    root = Path(root_path)
    all_runs_data = []

    if not root.exists():
        print(f"❌ 错误：路径不存在 - {root}")
        return []

    # 1. 获取所有运行编号文件夹 (1, 2, 3...)
    run_dirs = [d for d in root.iterdir() if d.is_dir()]

    # 尝试按数字排序
    try:
        run_dirs.sort(key=lambda x: int(x.name))
    except ValueError:
        run_dirs.sort(key=lambda x: x.name)

    print(f"📂 扫描到 {len(run_dirs)} 个运行目录...")

    for run_dir in run_dirs:
        # 2. 进入运行目录，寻找策略文件夹
        policy_dirs = [d for d in run_dir.iterdir() if d.is_dir()]

        if not policy_dirs:
            print(f"⚠️ 警告：{run_dir.name} 下为空，跳过。")
            continue

        target_policy_dir = policy_dirs[0]

        # 3. 寻找最后一天 (day_time_X)
        day_folders = []
        for d in target_policy_dir.iterdir():
            if d.is_dir() and d.name.startswith("day_time_"):
                match = re.search(r"day_time_(\d+)", d.name)
                if match:
                    day_num = int(match.group(1))
                    day_folders.append((day_num, d))

        if not day_folders:
            print(f"⚠️ 警告：在 {target_policy_dir.name} 中未找到 day_time 文件夹，跳过。")
            continue

        # 找到天数最大的文件夹
        max_day, max_day_folder = max(day_folders, key=lambda x: x[0])

        # 4. 读取 output_system_kpi.json
        kpi_file = max_day_folder / "output_system_kpi.json"

        if not kpi_file.exists():
            print(f"❌ 错误：文件不存在 - {kpi_file}")
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
                print(f"⚠️ 警告：Run {run_dir.name} 数据为空。")

        except Exception as e:
            print(f"❌ 读取 JSON 出错 ({kpi_file}): {e}")

    print(f"🎉 成功加载 {len(all_runs_data)} 组有效数据。")
    return all_runs_data


def eva_robustness(root_path: str, output_dir: str, cv_threshold: float = 0.2) -> None:
    """
    鲁棒性评估入口函数
    """
    # 稍微调整一下目录结构，避免太乱
    output_dir = os.path.join(output_dir, 'robustness_analysis')
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    all_runs_data = load_robustness_kpi_data(root_path)
    if all_runs_data:
        evaluate_monte_carlo_experiment(all_runs_data, output_dir, cv_threshold)
    else:
        print("未加载到有效数据，无法进行评估。")