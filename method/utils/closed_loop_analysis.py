import json
from pathlib import Path

import numpy as np
import pandas as pd


def extract_latest_data(folder: Path):
    """Extract data from the last day of a single policy folder"""
    day_dirs = sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith('day_time_')],
                      key=lambda x: int(x.name.split('_')[-1]))
    if not day_dirs: return None
    last_day = day_dirs[-1]
    try:
        with open(last_day / "output_system_kpi.json", 'r', encoding='utf-8') as f:
            kpis = json.load(f)
        with open(last_day / "output_policy.json", 'r', encoding='utf-8') as f:
            policy = json.load(f)
        return {"policy": policy, "kpis": kpis, "id": folder.name}
    except:
        return None


def analyze_objective_conflict(data_path: Path):
    """
    Analyze the correlation between Safety, Creativity, and Satisfaction under different policies.
    """
    records = []
    simple_dir = Path(data_path) / "简化"

    for policy_folder in simple_dir.iterdir():
        if not policy_folder.is_dir(): continue
        res = extract_latest_data(policy_folder)
        if res:
            records.append({
                "safety": res['kpis']['safety'][-1],
                "creativity": res['kpis']['creativity'][-1],
                "satisfaction": res['kpis']['satisfaction'][-1],
                "f_penalty": res['policy']['f_penalty'],
                "ai_threshold": res['policy']['ai_threshold']
            })

    df = pd.DataFrame(records)
    corr = df[['safety', 'creativity', 'satisfaction']].corr()

    print("📊 Indicator Correlation Matrix:")
    print(corr)

    if corr.loc['safety', 'creativity'] > 0:
        print("⚠️ Warning: Positive correlation found! Increasing regulation actually improved creativity, leading to frontier collapse.")
    return df


def analyze_parameter_sensitivity(df):
    """
    Analyze the contribution of policy parameters (F, Threshold) to the results.
    """
    from sklearn.ensemble import RandomForestRegressor

    X = df[['f_penalty', 'ai_threshold']]
    for target in ['safety', 'creativity']:
        model = RandomForestRegressor().fit(X, df[target])
        importances = dict(zip(X.columns, model.feature_importances_))
        print(f"🎯 Influence weight on {target}: {importances}")


def diagnostic_micro_behavior(policy_folder: Path):
    """
    Compare the true state of agents under high-pressure vs. relaxed policies.
    """
    last_day = sorted(policy_folder.glob("day_time_*"))[-1]

    with open(last_day / "output_personas.json", 'r', encoding='utf-8') as f:
        personas = json.load(f)
    with open(last_day / "output_contents.json", 'r', encoding='utf-8') as f:
        contents = json.load(f)

    active_creators = [p for p in personas if p['type'] == '合规创作者' and p['is_active']]
    post_wishes = [p['post_wish'] for p in active_creators]

    actual_posts = len([c for c in contents if c['author_id'] in [p['agent_id'] for p in active_creators]])

    return {
        "active_rate": len(active_creators) / 10,
        "wish_rate": np.mean(post_wishes) if post_wishes else 0,
        "content_density": actual_posts / len(active_creators) if active_creators else 0
    }


if __name__ == '__main__':
    path = Path(r"method\store\daily_memory_exports\2026-01-16\164123")
    d = analyze_objective_conflict(path)
    print(d)
    analyze_parameter_sensitivity(d)
    # print(diagnostic_micro_behavior(path))
    pass
