import json
from pathlib import Path

import numpy as np
import pandas as pd


def extract_latest_data(folder: Path):
    """从单个策略文件夹提取最后一天的数据"""
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
    分析不同政策下，Safety, Creativity, Satisfaction 之间的相关性。
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

    print("📊 指标相关性矩阵:")
    print(corr)

                                                           
    if corr.loc['safety', 'creativity'] > 0:
        print("⚠️ 警告：发现正相关！增加监管居然提升了创造力，这导致了前沿坍缩。")
    return df


def analyze_parameter_sensitivity(df):
    """
    分析政策参数(F, Threshold)对结果的贡献度。
    """
    from sklearn.ensemble import RandomForestRegressor

    X = df[['f_penalty', 'ai_threshold']]
    for target in ['safety', 'creativity']:
        model = RandomForestRegressor().fit(X, df[target])
        importances = dict(zip(X.columns, model.feature_importances_))
        print(f"🎯 对 {target} 的影响力权重: {importances}")


def diagnostic_micro_behavior(policy_folder: Path):
    """
    对比高压政策和宽松政策下，智能体的真实状态。
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
    sample_path = Path("method/store/daily_memory_exports/sample_run")
    if sample_path.exists():
        data = analyze_objective_conflict(sample_path)
        print(data)
        analyze_parameter_sensitivity(data)

























