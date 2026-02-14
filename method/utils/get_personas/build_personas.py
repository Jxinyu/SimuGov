import json
import random
import os
from typing import List, Dict, Literal


def _normalize_attribute(value: str) -> str:
    if not isinstance(value, str):
        return "中"  # Default fallback

    # If it is already a standard label, return directly
    if value in ["高", "中", "低"]:
        return value

    # These keywords are matched against raw data and are kept as keys
    mapping_rules = [
        ("天生反骨", "高"),
        ("独立思考", "中"),
        ("秩序拥护者", "低"),

        ("固执己见", "高"),
        ("有立场", "中"),
        ("绝对理性", "低"),

        ("玻璃心", "高"),
        ("务实派", "中"),
        ("乐天派", "低"),

        ("精打细算", "高"),
        ("追求性价比", "中"),
        ("不惜代价", "低")
    ]

    for keyword, tag in mapping_rules:
        if keyword in value:
            return tag

    return "中"


def _stratified_sample(
        pool: List[dict],
        total_count: int,
        attribute_name: str,
        ratios: Dict[str, float] = None
) -> List[dict]:
    """
    General stratified sampling function.
    """
    if not pool:
        print(f"⚠️ Warning: Candidate pool is empty, sampling cannot be performed.")
        return []

    # 1. Default ratios
    if ratios is None:
        ratios = {'高': 0.33, '中': 0.33, '低': 0.34}

    # 2. Bucketing - Cleaning logic added here
    buckets = {'高': [], '中': [], '低': []}

    for agent in pool:
        raw_val = agent.get(attribute_name, "中")

        clean_val = _normalize_attribute(raw_val)

        if clean_val in buckets:
            buckets[clean_val].append(agent)
        else:
            buckets['中'].append(agent)

    print(f"   [Pool Distribution] {attribute_name}: 高({len(buckets['高'])}) 中({len(buckets['中'])}) 低({len(buckets['低'])})")

    # 3. Calculate quotas
    target_counts = {}
    current_sum = 0

    for key, ratio in ratios.items():
        count = int(total_count * ratio)
        target_counts[key] = count
        current_sum += count

    # Fill the remainder gap
    remainder = total_count - current_sum
    if remainder > 0:
        sorted_keys = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
        for i in range(remainder):
            key = sorted_keys[i % len(sorted_keys)]
            target_counts[key] += 1

    # 4. Execute sampling
    selected_agents = []

    for key, target in target_counts.items():
        candidates = buckets[key]
        actual_available = len(candidates)

        if actual_available >= target:
            selected = random.sample(candidates, target)
            selected_agents.extend(selected)
        else:
            print(f"   ⚠️ Warning: Insufficient samples for '{attribute_name}={key}' (Required {target}, only {actual_available}). All available taken.")
            selected_agents.extend(candidates)

    # 5. Fallback supplement
    shortage = total_count - len(selected_agents)
    if shortage > 0:
        print(f"   🔄 Triggering fallback mechanism: Supplementing {shortage} samples...")
        selected_ids = {a['agent_id'] for a in selected_agents}
        remaining_candidates = [a for a in pool if a['agent_id'] not in selected_ids]

        if len(remaining_candidates) >= shortage:
            fallback_selection = random.sample(remaining_candidates, shortage)
            selected_agents.extend(fallback_selection)
        else:
            selected_agents.extend(remaining_candidates)

    return selected_agents


def select_compliance_creators(pool_path: str, count: int,
                               distribution_mode: Literal['uniform', 'sensitive', 'robust'] = 'uniform') -> List[dict]:
    """Select Compliance Creators (fp_sensitivity)"""
    print(f"\n🎯 [Compliance Creator] Loading... Target: {count} persons, Mode: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"Error: File does not exist {pool_path}")
        return []
    with open(pool_path, 'r', encoding='utf-8') as f:
        pool = json.load(f)

    ratio_presets = {
        'uniform': {'高': 0.33, '中': 0.33, '低': 0.34},
        'sensitive': {'高': 0.6, '中': 0.3, '低': 0.1},
        'robust': {'高': 0.1, '中': 0.3, '低': 0.6},
    }
    return _stratified_sample(pool, count, 'fp_sensitivity',
                              ratio_presets.get(distribution_mode, ratio_presets['uniform']))


def select_watermark_breakers(pool_path: str, count: int,
                              distribution_mode: Literal['pyramid', 'hardcore', 'opportunist'] = 'pyramid') -> List[
    dict]:
    """Select Watermark Breakers (cost_sensitivity)"""
    print(f"\n☠️ [Watermark Breaker] Loading... Target: {count} persons, Mode: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"Error: File does not exist {pool_path}")
        return []
    with open(pool_path, 'r', encoding='utf-8') as f:
        pool = json.load(f)

    ratio_presets = {
        'pyramid': {'高': 0.6, '中': 0.3, '低': 0.1},
        'hardcore': {'高': 0.1, '中': 0.3, '低': 0.6},
        'opportunist': {'高': 0.8, '中': 0.2, '低': 0.0}
    }
    return _stratified_sample(pool, count, 'cost_sensitivity',
                              ratio_presets.get(distribution_mode, ratio_presets['pyramid']))


def select_public(pool_path: str, count: int, distribution_mode: Literal['normal', 'rebel', 'conformist'] = 'normal') -> \
        List[dict]:
    """Select Public Agents (beta)"""
    print(f"\n📢 [Public] Loading... Target: {count} persons, Mode: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"Error: File does not exist {pool_path}")
        return []
    with open(pool_path, 'r', encoding='utf-8') as f:
        pool = json.load(f)

    ratio_presets = {
        'normal': {'高': 0.2, '中': 0.6, '低': 0.2},
        'rebel': {'高': 0.6, '中': 0.3, '低': 0.1},
        'conformist': {'高': 0.1, '中': 0.3, '低': 0.6}
    }
    return _stratified_sample(pool, count, 'beta', ratio_presets.get(distribution_mode, ratio_presets['normal']))


def generate_simulation_population(pool_dir: str, output_path: str, config: Dict):
    print("=" * 60)
    print("🚀 Starting simulation population construction (based on stratified quota sampling)...")
    print("=" * 60)

    creators = select_compliance_creators(
        os.path.join(pool_dir, "pool_compliance.json"),
        config['compliance_count'], config.get('compliance_mode', 'uniform')
    )

    breakers = select_watermark_breakers(
        os.path.join(pool_dir, "pool_breakers.json"),
        config['breaker_count'], config.get('breaker_mode', 'pyramid')
    )

    public = select_public(
        os.path.join(pool_dir, "pool_public.json"),
        config['public_count'], config.get('public_mode', 'normal')
    )

    all_agents = creators + breakers + public

    print("-" * 30)
    print(f"✅ Final Population Statistics: Total {len(all_agents)} persons")
    print(f"   - Compliance Creators: {len(creators)}")
    print(f"   - Watermark Breakers: {len(breakers)}")
    print(f"   - Public: {len(public)}")
    print("-" * 30)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_agents, f, ensure_ascii=False, indent=4)

    print(f"💾 Results saved to: {output_path}")
    return output_path


def set_low_beta_fp_sensitivity_personas(personas):
    """
    Set Low Rebellion Psychology (beta)
    """
    beta_dict = {
        '高': '【Innate Rebel】You extremely dislike being "managed" or "disciplined". If you feel the platform\'s hand of moderation extends too far (even for safety), your first reaction is physiological disgust and escape rather than compliance.',
        '中': '【Independent Thinker】You neither follow authority blindly nor rebel for the sake of rebellion. You critically examine every rule: you obey reasonable ones, and watch unreasonable or stupid ones with cold indifference, deducting points in your heart.',
        '低': '【Order Upholder】You are a mild-mannered citizen. You tend to trust the platform and authority, believing strict regulation is a necessary means to maintain community order. You might even dislike those who always complain about rules, viewing them as troublemakers.'
    }

    fp_sensitivity_dict = {
        '高': '【Fragile Heart/Highly Sensitive】You have extremely high self-esteem. Even a tiny misunderstanding or accidental hurt is magnified in your heart as an insult to your professional ability and a betrayal by the platform, triggering intense anger.',
        '中': '【Pragmatist/Has Boundaries】You are a rational person. Due to technical immaturity, you will tolerate occasional errors, but if errors become the norm, your patience will quickly run out.',
        '低': '【Optimist/Thick Skin】You have a very open and inclusive mindset. You believe that in the AI era, algorithmic misjudgment is a necessary cost of technical development. As long as it is not malicious targeting, you usually laugh it off without strong negative emotions.'
    }

    gamma_dict = {
        '高': '【Opinionated】You are very stubborn and a heavy user of information cocoons. Once you form a fixed impression of the platform (good or bad), even if there is contrary evidence later, you tend to ignore it and continue reinforcing your original view.',
        '中': '【Principled but Rational】You have preferences, but you are not blind. If strong facts are presented (e.g., seeing bad experiences for many consecutive days), you will slowly correct your views, though the process is a bit slow.',
        '低': '【Absolute Rationalist】You are a cold observer. You have almost no preconceived biases and only look at the facts at hand. Your attitude fluctuates rapidly with daily actual experiences and you do not get stuck in a fixed mindset.'
    }

    for person in personas:
        person['beta'] = beta_dict['低']
        person['gamma'] = gamma_dict['中']
        person['fp_sensitivity'] = fp_sensitivity_dict['低']
        person['standpoint'] = [0.8, 0.1, 0.1]

    with open(fr'method\data\low_beta_personas_{len(personas)}.json', 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(personas)} low beta sensitivity personas to file.")


def set_high_beta_fp_sensitivity_personas(personas):
    """
    Set High Rebellion Psychology (beta)
    """
    beta_dict = {
        '高': '【Innate Rebel】You extremely dislike being "managed" or "disciplined". If you feel the platform\'s hand of moderation extends too far (even for safety), your first reaction is physiological disgust and escape rather than compliance.',
        '中': '【Independent Thinker】You neither follow authority blindly nor rebel for the sake of rebellion. You critically examine every rule: you obey reasonable ones, and watch unreasonable or stupid ones with cold indifference, deducting points in your heart.',
        '低': '【Order Upholder】You are a mild-mannered citizen. You tend to trust the platform and authority, believing strict regulation is a necessary means to maintain community order. You might even dislike those who always complain about rules, viewing them as troublemakers.'
    }

    fp_sensitivity_dict = {
        '高': '【Fragile Heart/Highly Sensitive】You have extremely high self-esteem. Even a tiny misunderstanding or accidental hurt is magnified in your heart as an insult to your professional ability and a betrayal by the platform, triggering intense anger.',
        '中': '【Pragmatist/Has Boundaries】You are a rational person. Due to technical immaturity, you will tolerate occasional errors, but if errors become the norm, your patience will quickly run out.',
        '低': '【Optimist/Thick Skin】You have a very open and inclusive mindset. You believe that in the AI era, algorithmic misjudgment is a necessary cost of technical development. As long as it is not malicious targeting, you usually laugh it off without strong negative emotions.'
    }

    gamma_dict = {
        '高': '【Opinionated】You are very stubborn and a heavy user of information cocoons. Once you form a fixed impression of the platform (good or bad), even if there is contrary evidence later, you tend to ignore it and continue reinforcing your original view.',
        '中': '【Principled but Rational】You have preferences, but you are not blind. If strong facts are presented (e.g., seeing bad experiences for many consecutive days), you will slowly correct your views, though the process is a bit slow.',
        '低': '【Absolute Rationalist】You are a cold observer. You have almost no preconceived biases and only look at the facts at hand. Your attitude fluctuates rapidly with daily actual experiences and you do not get stuck in a fixed mindset.'
    }

    for person in personas:
        person['beta'] = beta_dict['高']
        person['gamma'] = gamma_dict['中']
        person['fp_sensitivity'] = fp_sensitivity_dict['高']
        person['standpoint'] = [0.1, 0.8, 0.1]

    with open(fr'method\data\high_beta_personas_{len(personas)}.json', 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(personas)} high beta sensitivity personas to file.")


def set_compare_personas_main(file_path):
    """
    Set control group personas.
    """
    with open(file_path,  'r', encoding='utf-8') as f:
        personas = json.load(f)
    set_low_beta_fp_sensitivity_personas(personas)
    set_high_beta_fp_sensitivity_personas(personas)
    print("Control group personas saved to file.")


def build_personas_main():
    BASE_POOL_DIR = r'method\data\pools'
    OUTPUT_FILE = r'method\data\trend_consistency_30.json'

    simulation_config = {
        'compliance_count': 10,
        'compliance_mode': 'uniform',
        'breaker_count': 5,
        'breaker_mode': 'uniform',
        'public_count': 15,
        'public_mode': 'uniform'
    }

    generate_simulation_population(BASE_POOL_DIR, OUTPUT_FILE, simulation_config)


if __name__ == '__main__':
    build_personas_main()
    # set_compare_personas_main(r'method\data\uniform_personas_30.json')
