import json
import random
import copy
import numpy as np
import os

INPUT_FILE = r"method\data\inner_consistency_personas_20.json"
OUTPUT_FILE = r"method\data\inner_consistency_personas_60.json"

MULTIPLIER = 3

NOISE_LEVEL_SATISFACTION = 0.08  # Satisfaction fluctuation range
NOISE_LEVEL_INFLUENCE = 0.05  # Influence fluctuation range
NOISE_LEVEL_STANDPOINT = 0.05  # Standpoint shift range


# ===========================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ File saved to: {filepath}")


def perturb_value(val, noise_std, min_val, max_val):
    """Add Gaussian noise to a single value and truncate"""
    if val is None: return None
    noise = random.gauss(0, noise_std)
    new_val = val + noise
    return max(min_val, min(new_val, max_val))


def perturb_standpoint(standpoint_list, noise_std=0.05):
    """
    Perturb the standpoint distribution and re-normalize.
    input: [0.6, 0.1, 0.3]
    output: [0.58, 0.12, 0.30] (sum=1.0)
    """
    if not standpoint_list:
        return [0.33, 0.33, 0.34]

    # 1. Convert to numpy array for easier calculation
    arr = np.array(standpoint_list, dtype=float)

    # 2. Add noise (ensure non-negative)
    noise = np.random.normal(0, noise_std, size=arr.shape)
    new_arr = arr + noise
    new_arr = np.maximum(new_arr, 0.01)  # Ensure minimum value is 0.01

    # 3. Normalization (Sum = 1)
    normalized_arr = new_arr / new_arr.sum()

    # 4. Return formatted with three decimal places
    return [round(x, 3) for x in normalized_arr.tolist()]


def perturb_satisfaction_history(history_list, noise_std=0.05):
    """Apply overall perturbation to the satisfaction history curve"""
    if not history_list:
        return []

    bias = random.gauss(0, noise_std)

    new_history = []
    for val in history_list:
        # Add overall bias plus a bit of random jitter
        new_val = val + bias + random.uniform(-0.02, 0.02)
        # Truncate between [-1, 1]
        new_val = max(-1.0, min(new_val, 1.0))
        new_history.append(round(new_val, 2))

    return new_history


def generate_expanded_dataset(source_data, multiplier):
    expanded_list = []

    print(f"🚀 Starting dataset expansion...")
    print(f"   - Original population: {len(source_data)}")
    print(f"   - Target multiplier: {multiplier}x")
    print(f"   - Estimated total: {len(source_data) * multiplier}")

    for original_agent in source_data:
        # For each original Agent, generate N variants
        for i in range(multiplier):
            # 1. Deep copy
            new_agent = copy.deepcopy(original_agent)

            # 2. Modify unique identifiers (ID and Name)
            suffix = f"_v{i + 1}"
            new_agent['agent_id'] = f"{original_agent['agent_id']}{suffix}"
            new_agent['name'] = f"{original_agent['name']}{suffix}"

            # 3. Data perturbation (to increase diversity)

            # 3.1 Influence perturbation (0 ~ 1)
            new_agent['influence'] = perturb_value(
                new_agent['influence'],
                NOISE_LEVEL_INFLUENCE,
                0.01, 0.99
            )
            new_agent['influence'] = round(new_agent['influence'], 2)

            # 3.2 Satisfaction history perturbation (-1 ~ 1)
            new_agent['satisfaction'] = perturb_satisfaction_history(
                new_agent['satisfaction'],
                NOISE_LEVEL_SATISFACTION
            )

            # 3.3 Standpoint perturbation (Sum=1)
            new_agent['standpoint'] = perturb_standpoint(
                new_agent['standpoint'],
                NOISE_LEVEL_STANDPOINT
            )

            # 3.4 Ensure active status (double assurance)
            new_agent['is_active'] = True

            # 3.5 Handle cost_sensitivity (if it is a Watermark Breaker)

            expanded_list.append(new_agent)

    # Shuffle to avoid clusters of the same type
    random.shuffle(expanded_list)

    print(f"✅ Expansion complete, actual generated population: {len(expanded_list)}")
    return expanded_list


def verify_distribution(data):
    """Simple statistical verification"""
    print("\n📊 Data Distribution Overview:")
    counts = {}
    sat_sum = 0
    for p in data:
        role = p['type']
        counts[role] = counts.get(role, 0) + 1
        sat_sum += p['satisfaction'][-1]

    for role, count in counts.items():
        print(f"   - {role}: {count} persons ({count / len(data):.1%})")

    print(f"   - Average satisfaction: {sat_sum / len(data):.2f}")


if __name__ == '__main__':
    # 1. Check input file
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        # If the file is missing, create a temporary sample for demonstration
        print("   (Please modify the INPUT_FILE path in the script to your actual JSON file path)")
    else:
        # 2. Load
        source_data = load_json(INPUT_FILE)

        # 3. Generate
        expanded_data = generate_expanded_dataset(source_data, MULTIPLIER)

        # 4. Verify
        verify_distribution(expanded_data)

        # 5. Save
        save_json(expanded_data, OUTPUT_FILE)
