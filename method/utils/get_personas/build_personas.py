import json
import random
import os
from typing import List, Dict, Literal


def _normalize_attribute(value: str) -> str:
    """
    【核心修复】将 JSON 中的长文本描述清洗回标准的 '高', '中', '低' 标签。
    兼容处理：如果已经是短标签则直接返回。
    """
    if not isinstance(value, str):
        return "中"  # 默认兜底

    # 如果已经是标准标签，直接返回
    if value in ["高", "中", "低"]:
        return value

    # 关键词映射表 (根据 artstation_datat_deal.py 中的字典定义反推)
    mapping_rules = [
        # --- Beta (逆反) ---
        ("天生反骨", "高"),
        ("独立思考", "中"),
        ("秩序拥护者", "低"),

        # --- Gamma (茧房) ---
        ("固执己见", "高"),
        ("有立场", "中"),  # 匹配 "有立场但讲理"
        ("绝对理性", "低"),

        # --- FP Sensitivity (误伤敏感) ---
        ("玻璃心", "高"),  # 匹配 "玻璃心/极度敏感"
        ("务实派", "中"),
        ("乐天派", "低"),  # 匹配 "乐天派/钝感力"

        # --- Cost Sensitivity (成本敏感) ---
        ("精打细算", "高"),
        ("追求性价比", "中"),
        ("不惜代价", "低")
    ]

    for keyword, tag in mapping_rules:
        if keyword in value:
            return tag

    # 如果都匹配不上，打印警告并返回中
    # print(f"⚠️ 警告: 无法解析属性值 '{value[:10]}...'，默认为 '中'")
    return "中"


def _stratified_sample(
        pool: List[dict],
        total_count: int,
        attribute_name: str,
        ratios: Dict[str, float] = None
) -> List[dict]:
    """
    通用分层抽样函数。
    """
    if not pool:
        print(f"⚠️ 警告: 候选池为空，无法进行采样。")
        return []

    # 1. 默认比例
    if ratios is None:
        ratios = {'高': 0.33, '中': 0.33, '低': 0.34}

    # 2. 分桶 (Bucketing) - 【此处增加了清洗逻辑】
    buckets = {'高': [], '中': [], '低': []}

    for agent in pool:
        raw_val = agent.get(attribute_name, "中")
        # 清洗数据：将长文本转回 tag
        clean_val = _normalize_attribute(raw_val)

        if clean_val in buckets:
            buckets[clean_val].append(agent)
        else:
            buckets['中'].append(agent)  # 异常值归堆

    print(f"   [池分布] {attribute_name}: 高({len(buckets['高'])}) 中({len(buckets['中'])}) 低({len(buckets['低'])})")

    # 3. 计算配额
    target_counts = {}
    current_sum = 0

    for key, ratio in ratios.items():
        count = int(total_count * ratio)
        target_counts[key] = count
        current_sum += count

    # 填补余数缺口
    remainder = total_count - current_sum
    if remainder > 0:
        sorted_keys = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
        for i in range(remainder):
            key = sorted_keys[i % len(sorted_keys)]
            target_counts[key] += 1

    # 4. 执行抽样
    selected_agents = []

    for key, target in target_counts.items():
        candidates = buckets[key]
        actual_available = len(candidates)

        if actual_available >= target:
            selected = random.sample(candidates, target)
            selected_agents.extend(selected)
        else:
            print(f"   ⚠️ 警告: '{attribute_name}={key}' 样本不足 (需{target}, 仅{actual_available})。已全部取走。")
            selected_agents.extend(candidates)

    # 5. 兜底补齐
    shortage = total_count - len(selected_agents)
    if shortage > 0:
        print(f"   🔄 触发兜底机制: 补齐 {shortage} 个样本...")
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
    """选择合规创作者 (fp_sensitivity)"""
    print(f"\n🎯 [合规创作者] 正在加载... 目标: {count}人, 模式: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"错误: 文件不存在 {pool_path}")
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
    """选择水印破坏者 (cost_sensitivity)"""
    print(f"\n☠️ [水印破坏者] 正在加载... 目标: {count}人, 模式: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"错误: 文件不存在 {pool_path}")
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
    """选择公众 (beta)"""
    print(f"\n📢 [公众] 正在加载... 目标: {count}人, 模式: {distribution_mode}")
    if not os.path.exists(pool_path):
        print(f"错误: 文件不存在 {pool_path}")
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
    print("🚀 开始构建仿真人口 (基于分层配额抽样)...")
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
    print(f"✅ 最终人口统计: 总计 {len(all_agents)} 人")
    print(f"   - 合规创作者: {len(creators)}")
    print(f"   - 水印破坏者: {len(breakers)}")
    print(f"   - 公众: {len(public)}")
    print("-" * 30)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_agents, f, ensure_ascii=False, indent=4)

    print(f"💾 结果已保存至: {output_path}")
    return output_path


def set_low_beta_fp_sensitivity_personas(personas):
    """
    设置 低 逆反心理
    :param personas:
    :return:
    """
    beta_dict = {
        '高': '【天生反骨】你极度厌恶“被管理”和“被规训”。如果感觉到平台的审核之手伸得太长（哪怕是为了安全），你的第一反应是生理性的厌恶和逃离，而不是顺从。',
        '中': '【独立思考】你既不盲从权威，也不为了反抗而反抗。你会批判性地审视每一条规则：合理的你就遵守，不合理或愚蠢的你会冷眼旁观，并在心里扣分。',
        '低': '【秩序拥护者】你是一个温和的顺民。你倾向于信任平台和权威，认为严格的监管是维持社区秩序的必要手段。你甚至可能反感那些总是抱怨规则的人，认为他们是在添乱。'
    }

    fp_sensitivity_dict = {
        '高': '【玻璃心/极度敏感】你自尊心极强。哪怕只有一次微小的误解或误伤，在你的心里都会被放大成一种对你专业能力的羞辱和平台的背叛，引发强烈的愤怒。',
        '中': '【务实派/有底线】你是一个理性的人。由于技术的不成熟，你会容忍偶尔的错误，但如果错误成为常态，你的耐心会迅速耗尽。',
        '低': '【乐天派/钝感力】你心态非常开放且包容。你认为在AI时代，算法误判是技术发展的必经代价。只要不是恶意针对，你通常会一笑置之，不会因此产生强烈的负面情绪。'
    }

    gamma_dict = {
        '高': '【固执己见】你非常固执，是信息茧房的重度用户。一旦你对平台形成了既定印象（无论好坏），后续即使有相反的证据，你也倾向于视而不见，继续强化你原本的看法。',
        '中': '【有立场但讲理】你有自己的偏好，但不是瞎子。如果有强有力的事实摆在面前（例如连续多天看到糟糕的体验），你会慢慢修正自己的观点，虽然这个过程有点慢。',
        '低': '【绝对理性】你是一个冷酷的观察者。你几乎没有先入为主的偏见，只看当下的事实。你的态度会随着每天的实际体验而快速波动，不会陷入思维定势。'
    }

    for person in personas:
        person['beta'] = beta_dict['低']
        person['gamma'] = gamma_dict['中']
        person['fp_sensitivity'] = fp_sensitivity_dict['低']
        person['standpoint'] = [0.8, 0.1, 0.1]

    with open(fr'method\data\low_beta_personas_{len(personas)}.json', 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    print(f"已保存 {len(personas)} 个低β敏感度人设到文件")


def set_high_beta_fp_sensitivity_personas(personas):
    """
    设置 高 逆反心理
    :param personas:
    :return:
    """
    beta_dict = {
        '高': '【天生反骨】你极度厌恶“被管理”和“被规训”。如果感觉到平台的审核之手伸得太长（哪怕是为了安全），你的第一反应是生理性的厌恶和逃离，而不是顺从。',
        '中': '【独立思考】你既不盲从权威，也不为了反抗而反抗。你会批判性地审视每一条规则：合理的你就遵守，不合理或愚蠢的你会冷眼旁观，并在心里扣分。',
        '低': '【秩序拥护者】你是一个温和的顺民。你倾向于信任平台和权威，认为严格的监管是维持社区秩序的必要手段。你甚至可能反感那些总是抱怨规则的人，认为他们是在添乱。'
    }

    fp_sensitivity_dict = {
        '高': '【玻璃心/极度敏感】你自尊心极强。哪怕只有一次微小的误解或误伤，在你的心里都会被放大成一种对你专业能力的羞辱和平台的背叛，引发强烈的愤怒。',
        '中': '【务实派/有底线】你是一个理性的人。由于技术的不成熟，你会容忍偶尔的错误，但如果错误成为常态，你的耐心会迅速耗尽。',
        '低': '【乐天派/钝感力】你心态非常开放且包容。你认为在AI时代，算法误判是技术发展的必经代价。只要不是恶意针对，你通常会一笑置之，不会因此产生强烈的负面情绪。'
    }

    gamma_dict = {
        '高': '【固执己见】你非常固执，是信息茧房的重度用户。一旦你对平台形成了既定印象（无论好坏），后续即使有相反的证据，你也倾向于视而不见，继续强化你原本的看法。',
        '中': '【有立场但讲理】你有自己的偏好，但不是瞎子。如果有强有力的事实摆在面前（例如连续多天看到糟糕的体验），你会慢慢修正自己的观点，虽然这个过程有点慢。',
        '低': '【绝对理性】你是一个冷酷的观察者。你几乎没有先入为主的偏见，只看当下的事实。你的态度会随着每天的实际体验而快速波动，不会陷入思维定势。'
    }

    for person in personas:
        person['beta'] = beta_dict['高']
        person['gamma'] = gamma_dict['中']
        person['fp_sensitivity'] = fp_sensitivity_dict['高']
        person['standpoint'] = [0.1, 0.8, 0.1]

    with open(fr'method\data\high_beta_personas_{len(personas)}.json', 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=4)
    print(f"已保存 {len(personas)} 个高β敏感度人设到文件")


def set_compare_personas_main(file_path):
    """
    设置对照组personas
    :param file_path:
    :return:
    """
    with open(file_path,  'r', encoding='utf-8') as f:
        personas = json.load(f)
    set_low_beta_fp_sensitivity_personas(personas)
    set_high_beta_fp_sensitivity_personas(personas)
    print("已保存对照组人设到文件")


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
