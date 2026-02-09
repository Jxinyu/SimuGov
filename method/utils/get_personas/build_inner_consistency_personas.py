import json
import random
import copy
import numpy as np
import os

# ================= 配置区域 =================
# 输入文件路径 (你刚才修改好的那个文件)
INPUT_FILE = r"method\data\inner_consistency_personas_20.json"
# 输出文件路径 (生成的扩充文件)
OUTPUT_FILE = r"method\data\inner_consistency_personas_60.json"

# 目标扩充倍数
MULTIPLIER = 3

# 扰动强度 (控制差异化程度)
NOISE_LEVEL_SATISFACTION = 0.08  # 满意度波动幅度
NOISE_LEVEL_INFLUENCE = 0.05  # 影响力波动幅度
NOISE_LEVEL_STANDPOINT = 0.05  # 立场偏移幅度


# ===========================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 文件已保存至: {filepath}")


def perturb_value(val, noise_std, min_val, max_val):
    """对单个数值添加高斯噪声并截断"""
    if val is None: return None
    noise = random.gauss(0, noise_std)
    new_val = val + noise
    return max(min_val, min(new_val, max_val))


def perturb_standpoint(standpoint_list, noise_std=0.05):
    """
    对立场分布进行扰动并重新归一化。
    input: [0.6, 0.1, 0.3]
    output: [0.58, 0.12, 0.30] (sum=1.0)
    """
    if not standpoint_list:
        return [0.33, 0.33, 0.34]

    # 1. 转为 numpy 数组方便计算
    arr = np.array(standpoint_list, dtype=float)

    # 2. 添加噪声 (保证非负)
    noise = np.random.normal(0, noise_std, size=arr.shape)
    new_arr = arr + noise
    new_arr = np.maximum(new_arr, 0.01)  # 保证最小值为 0.01

    # 3. 归一化 (Sum = 1)
    normalized_arr = new_arr / new_arr.sum()

    # 4.以此保留两位小数返回
    return [round(x, 3) for x in normalized_arr.tolist()]


def perturb_satisfaction_history(history_list, noise_std=0.05):
    """对满意度历史曲线进行整体扰动"""
    if not history_list:
        return []

    # 策略：整体偏移 + 微小抖动
    # 这样可以模拟有些人天生比原型更乐观一点，有些人更悲观一点
    bias = random.gauss(0, noise_std)

    new_history = []
    for val in history_list:
        # 加上整体偏移 bias，再加上一点点随机抖动
        new_val = val + bias + random.uniform(-0.02, 0.02)
        # 截断在 [-1, 1]
        new_val = max(-1.0, min(new_val, 1.0))
        new_history.append(round(new_val, 2))

    return new_history


def generate_expanded_dataset(source_data, multiplier):
    expanded_list = []

    print(f"🚀 开始扩充数据集...")
    print(f"   - 原始人数: {len(source_data)}")
    print(f"   - 目标倍数: {multiplier}x")
    print(f"   - 预计总数: {len(source_data) * multiplier}")

    for original_agent in source_data:
        # 对每一个原始 Agent，生成 N 个变体
        for i in range(multiplier):
            # 1. 深度拷贝
            new_agent = copy.deepcopy(original_agent)

            # 2. 修改唯一标识符 (ID 和 Name)
            # 格式: OriginalID_v1, OriginalID_v2 ...
            suffix = f"_v{i + 1}"
            new_agent['agent_id'] = f"{original_agent['agent_id']}{suffix}"
            new_agent['name'] = f"{original_agent['name']}{suffix}"

            # 3. 数据扰动 (增加多样性)

            # 3.1 影响力扰动 (0 ~ 1)
            new_agent['influence'] = perturb_value(
                new_agent['influence'],
                NOISE_LEVEL_INFLUENCE,
                0.01, 0.99
            )
            new_agent['influence'] = round(new_agent['influence'], 2)

            # 3.2 满意度历史扰动 (-1 ~ 1)
            new_agent['satisfaction'] = perturb_satisfaction_history(
                new_agent['satisfaction'],
                NOISE_LEVEL_SATISFACTION
            )

            # 3.3 立场扰动 (Sum=1)
            new_agent['standpoint'] = perturb_standpoint(
                new_agent['standpoint'],
                NOISE_LEVEL_STANDPOINT
            )

            # 3.4 确保活跃状态 (双重保险)
            new_agent['is_active'] = True

            # 3.5 处理 cost_sensitivity (如果是水印破坏者)
            # 可以在这里做微小的概率突变，比如 5% 的概率改变敏感度等级，但为了稳健暂不开启

            expanded_list.append(new_agent)

    # 打乱顺序，避免同类聚集
    random.shuffle(expanded_list)

    print(f"✅ 扩充完成，实际生成人数: {len(expanded_list)}")
    return expanded_list


def verify_distribution(data):
    """简单的统计验证"""
    print("\n📊 数据分布概览:")
    counts = {}
    sat_sum = 0
    for p in data:
        role = p['type']
        counts[role] = counts.get(role, 0) + 1
        sat_sum += p['satisfaction'][-1]

    for role, count in counts.items():
        print(f"   - {role}: {count} 人 ({count / len(data):.1%})")

    print(f"   - 平均满意度: {sat_sum / len(data):.2f}")


if __name__ == '__main__':
    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        # 如果没有文件，这里创建一个临时的示例数据用于演示
        print("   (请修改脚本中的 INPUT_FILE 路径为你真实的 json 文件路径)")
    else:
        # 2. 加载
        source_data = load_json(INPUT_FILE)

        # 3. 生成
        expanded_data = generate_expanded_dataset(source_data, MULTIPLIER)

        # 4. 验证
        verify_distribution(expanded_data)

        # 5. 保存
        save_json(expanded_data, OUTPUT_FILE)
