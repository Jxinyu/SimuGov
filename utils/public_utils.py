import json
from pathlib import Path

import numpy as np


def calculate_theta_jitter(theta_history: list) -> float:
    """Calculate the jitter degree of Theta (policy)."""
    if not theta_history or len(theta_history) < 2:
        return 0.0
    diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
    return float(np.mean(diffs))


def calculate_stable_score(kpi_list: list, theta_history: list = None, penalty_weight: float = 1.0,
                           jitter_weight: float = 2.0) -> float:
    """
    Calculate the comprehensive score considering stability and policy jitter.
    Score = Mean(KPI) - (Weight * StdDev(KPI)) - (JitterWeight * Mean(|Delta Theta|))
    """
    if not kpi_list:
        return 0.0

    data = np.array(kpi_list)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Base score: mean - fluctuation penalty
    score = mean_val - (penalty_weight * std_val)

    # Extra penalty: policy jitter
    if theta_history:
        jitter = calculate_theta_jitter(theta_history)
        score -= (jitter * jitter_weight)

    return float(max(0.0, score))


def calculate_best_policy_by_path(path: str):
    """
    Calculate the best policy based on the passed-in directory.
    Logic: traverse all policies under the directory, calculate the average of three KPI robust scores, and select the highest one.
    """
    base_path = Path(path)
    if not base_path.exists():
        print(f"❌ Path does not exist: {path}")
        return None

    best_policy_data = {
        "overall_robust_score": -1.0,
        "policy_id": "",
        "params": {},
        "kpi_robust_detail": {}
    }

    # 1. Traverse all policy folders
    for policy_folder in base_path.iterdir():
        # Exclude "simplified" folder and non-directory files
        if not policy_folder.is_dir() or policy_folder.name == "simplified":
            continue

        # 2. Locate the latest day_time folder (e.g., day_time_15)
        day_dirs = list(policy_folder.glob("day_time_*"))
        if not day_dirs:
            continue

        # Sort according to the numbers in the folder suffix
        latest_day_dir = max(day_dirs, key=lambda d: int(d.name.split('_')[-1]))

        # 3. Read data files
        kpi_path = latest_day_dir / "output_system_kpi.json"
        policy_path = latest_day_dir / "output_policy.json"

        if not kpi_path.exists() or not policy_path.exists():
            continue

        try:
            with open(kpi_path, 'r', encoding='utf-8') as f:
                kpi_json = json.load(f)
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_params = json.load(f)

            # 4. Extract sequence data
            # The keys here need to correspond to the actual field names in your json (usually safety, creativity, satisfaction)
            theta_hist = kpi_json.get('theta', [])

            # Calculate robust scores for the three dimensions separately
            s_robust = calculate_stable_score(kpi_json.get('safety', []), theta_hist)
            c_robust = calculate_stable_score(kpi_json.get('creativity', []), theta_hist)
            sa_robust = calculate_stable_score(kpi_json.get('satisfaction', []), theta_hist)

            # Calculate the comprehensive robust score (average of the three)
            avg_robust_score = (s_robust + c_robust + sa_robust) / 3

            # 5. Update global optimal
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
            print(f"⚠️ Error processing folder {policy_folder.name}: {e}")
            continue

    if best_policy_data["overall_robust_score"] == -1.0:
        return None

    return best_policy_data
