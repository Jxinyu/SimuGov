import json
from pathlib import Path

import numpy as np


def calculate_theta_jitter(theta_history: list) -> float:
    """计算 Theta (政策) 的抖动程度。"""
    if not theta_history or len(theta_history) < 2:
        return 0.0
    diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
    return float(np.mean(diffs))


def calculate_stable_score(kpi_list: list, theta_history: list = None, penalty_weight: float = 1.0,
                           jitter_weight: float = 2.0) -> float:
    """
    计算考虑了稳定性与政策抖动的综合得分。
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
    if theta_history:
        jitter = calculate_theta_jitter(theta_history)
        score -= (jitter * jitter_weight)

    return float(max(0.0, score))


def calculate_best_policy_by_path(path: str):
    """
    根据传入的目录计算最佳政策。
    逻辑：遍历目录下所有策略，计算三项KPI稳健分的平均值，选出最高者。
    """
    base_path = Path(path)
    if not base_path.exists():
        print(f"❌ 路径不存在: {path}")
        return None

    best_policy_data = {
        "overall_robust_score": -1.0,
        "policy_id": "",
        "params": {},
        "kpi_robust_detail": {}
    }

    # 1. 遍历所有政策文件夹
    for policy_folder in base_path.iterdir():
        # 排除“简化”文件夹及非目录文件
        if not policy_folder.is_dir() or policy_folder.name == "简化":
            continue

        # 2. 定位最新的 day_time 文件夹 (例如 day_time_15)
        day_dirs = list(policy_folder.glob("day_time_*"))
        if not day_dirs:
            continue

        # 按照文件夹后缀的数字进行排序
        latest_day_dir = max(day_dirs, key=lambda d: int(d.name.split('_')[-1]))

        # 3. 读取数据文件
        kpi_path = latest_day_dir / "output_system_kpi.json"
        policy_path = latest_day_dir / "output_policy.json"

        if not kpi_path.exists() or not policy_path.exists():
            continue

        try:
            with open(kpi_path, 'r', encoding='utf-8') as f:
                kpi_json = json.load(f)
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_params = json.load(f)

            # 4. 提取序列数据
            # 这里的 key 需要对应你 json 中的实际字段名 (一般是 safety, creativity, satisfaction)
            theta_hist = kpi_json.get('theta', [])

            # 分别计算三个维度的稳健得分
            s_robust = calculate_stable_score(kpi_json.get('safety', []), theta_hist)
            c_robust = calculate_stable_score(kpi_json.get('creativity', []), theta_hist)
            sa_robust = calculate_stable_score(kpi_json.get('satisfaction', []), theta_hist)

            # 计算综合稳健分 (三者平均)
            avg_robust_score = (s_robust + c_robust + sa_robust) / 3

            # 5. 更新全局最优
            if avg_robust_score > best_policy_data["overall_robust_score"]:
                best_policy_data.update({
                    "overall_robust_score": avg_robust_score,
                    "policy_id": policy_folder.name,
                    "params": policy_params,
                    "kpi_robust_detail": {
                        "safety_robust": s_robust,
                        "creativity_robust": c_robust,
                        "satisfaction_robust": sa_robust,
                        "theta_jitter": calculate_theta_jitter(theta_hist)
                    }
                })

        except Exception as e:
            print(f"⚠️ 处理文件夹 {policy_folder.name} 时出错: {e}")
            continue

    if best_policy_data["overall_robust_score"] == -1.0:
        return None

    return best_policy_data


# --- 使用示例 ---
if __name__ == "__main__":
    test_path = r'D:\A-课题\小论文内容\code\SimuGov\experiment\multi_granularity_method_evaluation\closed_loop_effectiveness_experiment\Verification_passed\Adaptive experiment passed\2\data\低逆反\elite'
    result = calculate_best_policy_by_path(test_path)

    if result:
        print("\n" + "=" * 30 + " 低逆反最优解 " + "=" * 30)
        print(f"🥇 最佳策略ID: {result['policy_id']}")
        print(f"📊 综合稳健得分: {result['overall_robust_score']:.4f}")
        print(f"🛠️ 政策参数: {result['params']}")
        print(f"📈 维度详情: {result['kpi_robust_detail']}")
        print("=" * 75)

    # test_path = r'method\store\daily_memory_exports\2026-01-19\高逆反'
    # result = calculate_best_policy_by_path(test_path)
    #
    # if result:
    #     print("\n" + "=" * 30 + " 高逆反最优解 " + "=" * 30)
    #     print(f"🥇 最佳策略ID: {result['policy_id']}")
    #     print(f"📊 综合稳健得分: {result['overall_robust_score']:.4f}")
    #     print(f"🛠️ 政策参数: {result['params']}")
    #     print(f"📈 维度详情: {result['kpi_robust_detail']}")
    #     print("=" * 75)
    #
    # test_path = r'experiment\多粒度方法评估\闭环有效性实验\验证通过\自适应实验通过\2\data\低入高\运行数据'
    # result = calculate_best_policy_by_path(test_path)
    #
    # if result:
    #     print("\n" + "=" * 30 + " 低入高结果 " + "=" * 30)
    #     print(f"🥇 最佳策略ID: {result['policy_id']}")
    #     print(f"📊 综合稳健得分: {result['overall_robust_score']:.4f}")
    #     print(f"🛠️ 政策参数: {result['params']}")
    #     print(f"📈 维度详情: {result['kpi_robust_detail']}")
    #     print("=" * 75)



















