import json
from pathlib import Path

import numpy as np


def calculate_theta_jitter(theta_history: list) -> float:
    if not theta_history or len(theta_history) < 2:
        return 0.0
    diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
    return float(np.mean(diffs))


def calculate_stable_score(
    kpi_list: list,
    theta_history: list | None = None,
    penalty_weight: float = 1.0,
    jitter_weight: float = 2.0,
) -> float:
    if not kpi_list:
        return 0.0

    data = np.array(kpi_list, dtype=float)
    score = float(np.mean(data) - penalty_weight * np.std(data))
    if theta_history:
        score -= calculate_theta_jitter(theta_history) * jitter_weight
    return float(max(0.0, score))


def calculate_best_policy_by_path(path: str | Path):
    base_path = Path(path)
    if not base_path.exists():
        return None

    best_policy_data = {
        "overall_robust_score": -1.0,
        "policy_id": "",
        "params": {},
        "kpi_robust_detail": {},
    }

    for policy_folder in base_path.iterdir():
        if not policy_folder.is_dir():
            continue

        day_dirs = [
            d for d in policy_folder.glob("day_time_*")
            if d.name.split("_")[-1].isdigit()
        ]
        if not day_dirs:
            continue

        latest_day_dir = max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))
        kpi_path = latest_day_dir / "output_system_kpi.json"
        policy_path = latest_day_dir / "output_policy.json"
        if not kpi_path.exists() or not policy_path.exists():
            continue

        with kpi_path.open("r", encoding="utf-8") as f:
            kpi_json = json.load(f)
        with policy_path.open("r", encoding="utf-8") as f:
            policy_params = json.load(f)

        theta_hist = kpi_json.get("theta", [])
        safety_score = calculate_stable_score(kpi_json.get("safety", []), theta_hist)
        creativity_score = calculate_stable_score(kpi_json.get("creativity", []), theta_hist)
        satisfaction_score = calculate_stable_score(kpi_json.get("satisfaction", []), theta_hist)
        avg_score = (safety_score + creativity_score + satisfaction_score) / 3

        if avg_score > best_policy_data["overall_robust_score"]:
            best_policy_data.update(
                {
                    "overall_robust_score": avg_score,
                    "policy_id": policy_folder.name,
                    "params": policy_params,
                    "kpi_robust_detail": {
                        "safety_robust": safety_score,
                        "creativity_robust": creativity_score,
                        "satisfaction_robust": satisfaction_score,
                        "theta_jitter": calculate_theta_jitter(theta_hist),
                    },
                }
            )

    if best_policy_data["overall_robust_score"] == -1.0:
        return None
    return best_policy_data


if __name__ == "__main__":
    sample_path = Path("method/store/daily_memory_exports/sample_run")
    if sample_path.exists():
        print(calculate_best_policy_by_path(sample_path))
