import json
import os
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D  # 用于创建自定义图例
import logging

log = logging.getLogger(__name__)

# --- 1. 配置区域 ---

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

BASE_PROJECT_PATH = Path(r'D:\Assign\topic-code\topic-1')
DATA_ROOT = BASE_PROJECT_PATH / 'method' / 'store' / 'daily_memory_exports'


# --- 2. 辅助函数 ---

def find_latest_run_directory(root_path: Path) -> Path | None:
    try:
        latest_date_dir = max([d for d in root_path.iterdir() if d.is_dir() and d.name.split('-')[0].isdigit()],
                              key=lambda d: d.name)
        latest_run_dir = max([d for d in latest_date_dir.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()],
                             key=lambda d: d.name)
        log.info(f"✅ 成功定位到最新的运行目录: {latest_run_dir}")
        return latest_run_dir
    except (ValueError, FileNotFoundError):
        log.error(f"❌ 错误: 在 '{root_path}' 中找不到任何有效的运行数据目录。")
        return None


def _get_strategy_dirs(run_directory: Path, is_simplified: bool) -> list:
    if is_simplified:
        simplified_dir_path = run_directory / '简化'
        if simplified_dir_path.exists() and simplified_dir_path.is_dir():
            log.info("--- 模式: 仅处理 '简化' 文件夹内的数据 ---")
            return [d for d in simplified_dir_path.iterdir() if d.is_dir()]
        else:
            log.warning(f"❌ 警告: 在 {run_directory} 中未找到 '简化' 文件夹。")
            return []
    else:
        log.info("--- 模式: 仅处理顶层策略数据 (排除 '简化' 文件夹) ---")
        return [d for d in run_directory.iterdir() if d.is_dir() and d.name != '简化']


def calculate_stable_score(kpi_list: list, theta_history: list = None, penalty_weight: float = 1.0,
                           jitter_weight: float = 2.0) -> float:
    """
    【新增】计算考虑了稳定性与政策抖动的综合得分。
    逻辑与 NSGA-II 中的 evaluate_policy 保持完全一致。
    Score = Mean(KPI) - (Weight * StdDev(KPI)) - (JitterWeight * Mean(|Delta Theta|))
    """
    if not kpi_list:
        return 0.0

    data = np.array(kpi_list)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # 基础分：均值 - 波动惩罚
    score = mean_val - (penalty_weight * std_val)

    # 额外惩罚：政策抖动
    if theta_history and len(theta_history) > 1:
        diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
        jitter = np.mean(diffs)
        score -= (jitter * jitter_weight)

    return float(max(0.0, score))


def collect_timeseries_kpi_data(run_directory: Path, is_simplified: bool) -> list:
    """【已恢复】收集每个策略的完整KPI时间序列数据。"""
    results = []
    strategy_dirs_to_process = _get_strategy_dirs(run_directory, is_simplified)
    if not strategy_dirs_to_process:
        log.warning("警告: 未找到任何符合条件的策略结果文件夹。")
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

            # 只要有数据就添加，不再强制校验长度完全一致（虽然通常是一致的，但为了鲁棒性）
            if kpi_history['safety']:
                results.append({'policy': policy_data, 'kpis': kpi_history, 'thetas': theta_history})
                log.info(f"  - (时间序列) 成功处理策略: {strategy_dir.name} (天数: {len(kpi_history['safety'])})")
        except Exception as e:
            log.error(f"  - 错误: 处理文件夹 '{strategy_dir.name}' 时发生错误: {e}")
    results.sort(key=lambda r: r['policy'].get('f_penalty', 0))
    return results


def collect_final_kpi_data(run_directory: Path, is_simplified: bool) -> list:
    """
    【修改版】收集每个策略的KPI数据，并计算‘稳定性得分’用于绘制帕累托前沿。
    不再只取最后一天，而是应用与 NSGA 相同的公式。
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

            # 获取完整序列
            s_list = kpi_data.get('safety', [])
            c_list = kpi_data.get('creativity', [])
            sat_list = kpi_data.get('satisfaction', [])
            theta_list = kpi_data.get('theta', [])

            if not s_list:
                continue

            # === 核心修改：计算稳定性得分 ===
            # 使用与 NSGA 中 evaluate_policy 相同的权重参数
            w_std = 1.0
            w_jitter = 2.0

            # 计算得分 (Mean - Std - Jitter)
            score_safety = calculate_stable_score(s_list, theta_list, w_std, w_jitter)
            score_creativity = calculate_stable_score(c_list, theta_list, w_std, w_jitter)
            # 满意度可以给稍微不同的权重，这里暂且保持一致或者微调 (参考NSGA代码是0.8)
            score_satisfaction = calculate_stable_score(sat_list, theta_list, 0.8, w_jitter)

            final_kpis = {
                'safety': score_safety,
                'creativity': score_creativity,
                'satisfaction': score_satisfaction
            }

            results.append({'policy': policy_data, 'final_kpis': final_kpis})
            log.info(
                f"  - (帕累托-稳定分) 处理策略: {strategy_dir.name} | S:{score_safety:.2f} C:{score_creativity:.2f}")

        except Exception as e:
            log.error(f"  - 错误: 处理文件夹 '{strategy_dir.name}' 时发生错误: {e}")
    return results


def format_policy_for_legend(policy: dict) -> str:
    key_map = {'f_penalty': '惩罚', 'ai_threshold': '法定ai阈值', 'e_edu': '教育'}
    parts = [f"{key_map.get(k, k)}={v}" for k, v in policy.items()]
    return ", ".join(parts)


def plot_kpi_timeseries(results: list, output_filename: str):
    """绘制KPI随时间变化的曲线图，支持不同策略运行天数不一致的情况。"""
    if not results:
        log.error("❌ 错误: 没有可供绘制的时间序列数据。")
        return

    kpi_names = ['safety', 'creativity', 'satisfaction']
    kpi_titles = {'safety': '安全性 (Safety) 随时间变化对比',
                  'creativity': '创造力 (Creativity) 随时间变化对比',
                  'satisfaction': '满意度 (Satisfaction) 随时间变化对比'}

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 22), sharex=True)
    fig.suptitle('不同策略下KPI随时间演变对比 (附每日θ值)', fontsize=22, weight='bold')

    # 1. 找出所有结果中最大的天数，用于设置 X 轴范围
    max_days = 0
    for res in results:
        # 取 safety 的长度作为该策略的天数
        days_count = len(res['kpis']['safety'])
        if days_count > max_days:
            max_days = days_count

    if max_days == 0:
        log.error("❌ 错误: 数据中似乎没有有效的天数信息。")
        return

    # 全局 X 轴刻度
    global_days_ticks = np.arange(1, max_days + 1)

    for i, kpi_name in enumerate(kpi_names):
        ax = axes[i]
        for result in results:
            policy = result['policy']
            kpi_values = result['kpis'][kpi_name]
            thetas = result['thetas']
            legend_label = format_policy_for_legend(policy)

            # 2. 为当前这条线生成独立的 X 轴数据
            current_len = len(kpi_values)
            if current_len == 0:
                continue

            current_days = np.arange(1, current_len + 1)

            # 绘制曲线
            line, = ax.plot(current_days, kpi_values, marker='o', linestyle='-', markersize=5, label=legend_label)

            # 标注 theta 值
            # 注意：要处理 thetas 长度可能跟 kpi_values 不完全一致的边界情况（虽然理论上应该一致）
            for day_idx, (day, kpi_val) in enumerate(zip(current_days, kpi_values)):
                if day_idx < len(thetas):
                    theta_val = thetas[day_idx]
                    if theta_val is not None:
                        ax.annotate(f'θ={theta_val:.2f}', (day, kpi_val),
                                    textcoords="offset points", xytext=(0, 10), ha='center',
                                    fontsize=8, color=line.get_color(), alpha=0.8)

        ax.set_title(kpi_titles[kpi_name], fontsize=16)
        ax.set_ylabel('KPI 指数', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='策略参数组合', bbox_to_anchor=(1.02, 1), loc='upper left')

        # 设置 X 轴刻度为全局最大天数
        if i == len(kpi_names) - 1:
            ax.set_xlabel('天数 (Day)', fontsize=14)
            ax.set_xticks(global_days_ticks)
    for ax in axes:
        ax.tick_params(axis='x', labelbottom=True)

        # 仍然只需要为最下面的图设置 X 轴标题
    axes[-1].set_xlabel('天数 (Day)', fontsize=14)

    plt.subplots_adjust(right=0.7)
    plt.subplots_adjust(right=0.7)
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        log.info(f"\n✅ 时间序列图已成功保存到: {output_filename}")
    except Exception as e:
        log.error(f"\n❌ 错误: 保存图表失败。错误: {e}")
    # plt.show()


def plot_pareto_front(results: list, output_filename: str):
    """绘制帕累托前沿图 (坐标为稳定性得分)。"""
    if not results:
        log.error("❌ 错误: 没有可供绘制帕累托前沿的数据。")
        return

    points = np.array([[r['final_kpis']['safety'], r['final_kpis']['creativity']] for r in results])
    satisfaction = np.array([r['final_kpis']['satisfaction'] for r in results])

    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if np.any(np.all(points >= p, axis=1) & np.any(points > p, axis=1)):
            is_pareto[i] = False

    pareto_points = points[is_pareto]
    # 根据 safety 排序以正确连线
    if len(pareto_points) > 0:
        pareto_front_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
    else:
        pareto_front_sorted = np.array([])

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(points[:, 0], points[:, 1], s=100, c=satisfaction, cmap='viridis', edgecolors='#333333',
                          linewidths=0.5, zorder=3)

    if len(pareto_front_sorted) > 0:
        plt.plot(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], 'r-', lw=2, zorder=2,
                 label='帕累托前沿 (效率边界)')

    texts = []
    for result in results:
        policy_text = format_policy_for_legend(result['policy'])
        coords = (result['final_kpis']['safety'], result['final_kpis']['creativity'])
        texts.append(plt.text(coords[0], coords[1], policy_text, fontsize=9, color='navy'))

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # === 修改了标题和坐标轴，反映得分本质 ===
    plt.title('AI治理策略的帕累托前沿 (稳定性调整后得分)', fontsize=20, weight='bold')
    plt.xlabel('安全性得分 (Mean - Std - Jitter)', fontsize=14)
    plt.ylabel('创造力得分 (Mean - Std - Jitter)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label('满意度得分 (Satisfaction Score)', fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='策略解 (颜色代表满意度)', markerfacecolor='gray', markersize=10,
               markeredgecolor='#333333'),
        Line2D([0], [0], color='red', lw=2, label='帕累托前沿')
    ]
    plt.legend(handles=legend_elements, fontsize=12)

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        log.info(f"\n✅ 帕累托前沿图已成功保存到: {output_filename}")
    except Exception as e:
        log.error(f"\n❌ 错误: 保存图表失败。错误: {e}")
    # plt.show()


# --- 3. 主执行函数 ---

def draw_kpi_timeseries_main(plot_simplified: bool, output_dir: Path):
    """【已恢复】主函数：执行时间序列图的绘制流程"""
    log.info("--- 开始运行【KPI时间序列】分析脚本 ---")
    os.makedirs(output_dir, exist_ok=True)
    latest_run_dir = find_latest_run_directory(DATA_ROOT)
    if latest_run_dir:
        kpi_results = collect_timeseries_kpi_data(latest_run_dir, is_simplified=plot_simplified)
        if kpi_results:
            plot_type = "simplified" if plot_simplified else "completed"
            output_filename = output_dir / f"kpi_timeseries_{plot_type}.png"
            plot_kpi_timeseries(kpi_results, output_filename)
    log.info("--- 脚本运行结束 ---")


def draw_pareto_front_main(plot_simplified: bool, output_dir: Path):
    """主函数：执行帕累托前沿图的绘制流程"""
    log.info("--- 开始运行【帕累托前沿】分析脚本 ---")
    os.makedirs(output_dir, exist_ok=True)
    latest_run_dir = find_latest_run_directory(DATA_ROOT)
    if latest_run_dir:
        final_kpi_results = collect_final_kpi_data(latest_run_dir, is_simplified=plot_simplified)
        if final_kpi_results:
            plot_type = "simplified" if plot_simplified else "completed"
            output_filename = output_dir / f"pareto_front_{plot_type}.png"
            plot_pareto_front(final_kpi_results, output_filename)
    log.info("--- 脚本运行结束 ---")


def draw_kpi_main(plot_mode='T', use_simplified_data=False,
                  output_dir: Path = Path(r'result_data')):
    """

    :param output_dir: 文件输出目录
    :param plot_mode: 择绘图模式: 'T(时序对比)' 或 'P（帕累托前沿）'
    :param use_simplified_data: 选择数据源: True 代表 '简化' 文件夹, False 代表顶层文件夹
    :return:
    """

    if plot_mode == 'T':
        draw_kpi_timeseries_main(plot_simplified=use_simplified_data, output_dir=output_dir)
    elif plot_mode == 'P':
        draw_pareto_front_main(plot_simplified=use_simplified_data, output_dir=output_dir)
    else:
        log.error(f"❌ 错误: 无效的 PLOT_MODE '{plot_mode}'. 请选择 'timeseries' 或 'pareto'.")
    pass


if __name__ == '__main__':
    # --- 控制开关 ---
    draw_kpi_main(plot_mode='P', use_simplified_data=True)
    pass

