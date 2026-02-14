import numpy as np
import random
import pprint


def latin_hypercube_sampling(n_samples: int):
    """
    Implement Latin Hypercube Sampling (LHS) for AI watermarking policy parameters.
    Revised version: correctly handle numerical range distribution.
    """
    granularity = 0.01
    print(f"--- Starting Latin Hypercube Sampling for {n_samples} samples (Granularity: {granularity}) ---")

    params_config = {
        'f_penalty': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'ai_threshold': {'type': 'continuous', 'bounds': (0.00, 1.00)},
        'e_edu': {'type': 'discrete', 'values': ['Low', 'Medium', 'High']}
    }

    # Used to store the sample list generated for each parameter
    samples_per_param = {}

    # 2. Independently generate stratified samples for each parameter
    for param_name, config in params_config.items():

        if config['type'] == 'discrete':
            categories = config['values']
            k = len(categories)

            # Calculate the number of occurrences for each category
            base_count = n_samples // k
            remainder = n_samples % k

            counts = [base_count] * k
            # Randomly assign remainders to different categories to prevent always biasing towards the first few
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

    # 3. Combine final policies
    final_policy_samples = []
    for i in range(n_samples):
        policy = {
            'f_penalty': float(samples_per_param['f_penalty'][i]),
            'ai_threshold': float(samples_per_param['ai_threshold'][i]),
            'e_edu': str(samples_per_param['e_edu'][i])
        }
        final_policy_samples.append(policy)

    print("--- LHS complete. All parameter samples have been merged. ---")
    return final_policy_samples


if __name__ == '__main__':
    # Test code
    POPULATION_SIZE = 10
    population = latin_hypercube_sampling(n_samples=POPULATION_SIZE)
    pprint.pprint(population)
