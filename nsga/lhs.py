import numpy as np
import random
import pprint


def latin_hypercube_sampling(n_samples: int):
    """
    针对AI水印政策参数实现拉丁超立方抽样 (LHS)。
    修正版：正确处理数值范围分布。
    """
              
    granularity = 0.01
    print(f"--- 开始 {n_samples} 个样本的拉丁超立方体采样 (粒度: {granularity}) ---")

                 
                                          
    params_config = {
        'f_penalty': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'ai_threshold': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'e_edu': {'type': 'discrete', 'values': ['低', '中', '高']}
    }

                     
    samples_per_param = {}

                      
    for param_name, config in params_config.items():

                                                 
        if config['type'] == 'discrete':
            categories = config['values']
            k = len(categories)

                         
            base_count = n_samples // k
            remainder = n_samples % k

            counts = [base_count] * k
                                    
            remainder_indices = random.sample(range(k), remainder)
            for idx in remainder_indices:
                counts[idx] += 1

            param_samples = np.repeat(categories, counts)
            np.random.shuffle(param_samples)
            samples_per_param[param_name] = param_samples

                                            
        elif config['type'] == 'continuous':
            low, high = config['bounds']

                                                         
            intervals = np.linspace(low, high, n_samples + 1)

            param_samples = []
            for i in range(n_samples):
                               
                val = random.uniform(intervals[i], intervals[i + 1])
                                   
                val = round(val / granularity) * granularity
                          
                val = max(low, min(high, val))
                param_samples.append(round(val, 2))

                                          
            random.shuffle(param_samples)
            samples_per_param[param_name] = param_samples

               
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
          
    POPULATION_SIZE = 10
    population = latin_hypercube_sampling(n_samples=POPULATION_SIZE)
    pprint.pprint(population)
