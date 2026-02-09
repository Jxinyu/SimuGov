import numpy as np
import random
import pprint


def latin_hypercube_sampling(n_samples: int):
    """
    针对AI水印政策参数实现拉丁超立方抽样 (LHS)。
    修正版：正确处理数值范围分布。
    """
    # 获取配置中的粒度
    granularity = 0.01
    print(f"--- 开始 {n_samples} 个样本的拉丁超立方体采样 (粒度: {granularity}) ---")

    # 1. 定义参数空间配置
    # 注意：不再预先生成 101 个离散值，而是使用 bounds 来动态计算
    params_config = {
        'f_penalty': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'ai_threshold': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'e_edu': {'type': 'discrete', 'values': ['低', '中', '高']}
    }

    # 用于存储每个参数生成的样本列表
    samples_per_param = {}

    # 2. 独立为每个参数生成分层样本
    for param_name, config in params_config.items():

        # --- 情况 A: 真正的离散类别 (如 '低', '中', '高') ---
        if config['type'] == 'discrete':
            categories = config['values']
            k = len(categories)

            # 计算每个类别出现的次数
            base_count = n_samples // k
            remainder = n_samples % k

            counts = [base_count] * k
            # 随机分配余数给不同的类别，防止总是偏向前几个
            remainder_indices = random.sample(range(k), remainder)
            for idx in remainder_indices:
                counts[idx] += 1

            param_samples = np.repeat(categories, counts)
            np.random.shuffle(param_samples)
            samples_per_param[param_name] = param_samples

        # --- 情况 B: 连续数值 (如 0.00 - 1.00) ---
        elif config['type'] == 'continuous':
            low, high = config['bounds']

            # LHS 核心：将区间 [low, high] 分成 n_samples 个等宽的小区间
            intervals = np.linspace(low, high, n_samples + 1)

            param_samples = []
            for i in range(n_samples):
                # 在每个小区间内随机取一个点
                val = random.uniform(intervals[i], intervals[i + 1])
                # 应用粒度控制 (Rounding)
                val = round(val / granularity) * granularity
                # 再次截断以防万一
                val = max(low, min(high, val))
                param_samples.append(round(val, 2))

            # 必须打乱顺序！否则所有参数都是从小到大排列，相关性为 1
            random.shuffle(param_samples)
            samples_per_param[param_name] = param_samples

    # 3. 组合最终策略
    final_policy_samples = []
    for i in range(n_samples):
        policy = {
            'f_penalty': float(samples_per_param['f_penalty'][i]),
            'ai_threshold': float(samples_per_param['ai_threshold'][i]),
            'e_edu': str(samples_per_param['e_edu'][i])
        }
        final_policy_samples.append(policy)

    print("--- LHS 完成。所有参数样本均已合并。 ---")
    return final_policy_samples


if __name__ == '__main__':
    # 测试代码
    POPULATION_SIZE = 10
    population = latin_hypercube_sampling(n_samples=POPULATION_SIZE)
    pprint.pprint(population)
