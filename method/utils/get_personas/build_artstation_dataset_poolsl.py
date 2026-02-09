import asyncio
import json
import re
import os
import random
import numpy as np
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Dict
from method.utils.get_llm import get_async_llm

# 控制并发数
sem = asyncio.Semaphore(30)


class OutputFormat(BaseModel):
    reasoning: str = Field(description="【深度思维链】推演过程。")
    name: str = Field(description="给自己起一个独一无二的名字")
    type: Literal["合规创作者", "水印破坏者", "公众"] = Field(description="身份类型")
    standpoint: List[float] = Field(description="[信任派概率, 反抗派概率, 中立派概率]")
    description: str = Field(description="第一人称自述")
    beta: Literal["高", "中", "低"] = Field(description="逆反心理系数")
    gamma: Literal["高", "中", "低"] = Field(description="信息茧房/固执程度")
    fp_sensitivity: Literal["高", "中", "低"] = Field(description="误伤敏感度")
    cost_sensitivity: Literal["高", "中", "低"] = Field(description="行动成本敏感度")
    beliefs: List[str] = Field(description="核心信念列表")

    @field_validator('type', mode='before')
    @classmethod
    def fix_type_enum(cls, v):
        if v in ["合规创作者", "水印破坏者", "公众"]: return v
        v_str = str(v)
        if "中间" in v_str or "摇摆" in v_str or "公众" in v_str: return "公众"
        if "合规" in v_str or "技术" in v_str: return "合规创作者"
        if "破坏" in v_str or "反抗" in v_str or "水印" in v_str: return "水印破坏者"
        return "公众"


class Persona(BaseModel):
    agent_id: str = Field(..., description="The agent ID")
    name: str = Field(..., description="The agent name")
    type: Literal['合规创作者', '水印破坏者', '公众']
    description: str = Field(..., description="The agent description")
    standpoint: List[float] = Field(..., description="人物立场")
    beta: str = Field(..., description="逆反心理参数")
    gamma: str = Field(..., description="确认偏误系数")
    fp_sensitivity: Optional[str] = Field(..., description="误伤敏感度")
    cost_sensitivity: Optional[str] = Field(..., description="成本敏感度")
    influence: float = Field(..., description="智能体影响力")
    satisfaction: List[float] = Field(..., description="满意度历史")
    post_wish: Optional[bool] = Field(..., description="发布意愿")
    is_active: bool = Field(..., description="是否活跃")
    beliefs: List[str] = Field(..., description="信念")
    social_relationships: Dict[str, float] = Field(..., description="社交关系")


# ==========================================
# 3. 辅助计算函数 (还原你原有的逻辑)
# ==========================================

def clean_number(value):
    if pd.isna(value) or str(value).strip() == '<null>': return 0
    value = str(value).lower().replace(',', '')
    if 'k' in value:
        return float(re.sub(r'[^0-9.]', '', value)) * 1000
    return float(re.sub(r'[^0-9.]', '', value))


def analyze_user_dna(row):
    """
    分析用户的技术栈，返回 (2D分数, 技术分数)
    """
    traits_2_d = {
        'software': ['procreate', 'clip studio', 'sai', 'painter', 'artrage', 'krita', 'manga studio',
                     'illustrator draw', 'sketchclub', 'firealpaca', 'tvpaint', 'pixelmator', 'artflow'],
        'tags': ['digital illustration', '2d', 'sketch', 'painting', 'drawing', 'character design', 'anime', 'manga',
                 'fantasy', 'solitaire', 'fanart', 'girl', 'portrait', 'handpainted', 'watercolor', 'ink'],
        'title': ['sketch', 'daily', 'study', 'doodle', 'practice', 'girl', 'boy', 'princess', 'dragon', 'happiness',
                  'love', 'dream', 'feeling', 'mood', 'color']
    }
    traits_tech = {
        'software': ['unreal', 'unity', 'blender', 'maya', '3ds max', 'zbrush', 'substance', 'houdini', 'marmoset',
                     'python', 'nuke', 'fusion 360', 'speedtree', 'marvelous designer', 'xgen', 'mari', 'quixel',
                     'megascans', 'rizomuv', 'world machine', 'gaea'],
        'tags': ['3d', 'pbr', 'environment', 'prop', 'asset', 'game art', 'real-time', 'scifi', 'vehicle', 'weapon',
                 'hard surface', 'groom', 'photogrammetry', 'ue4', 'ue5', 'material', 'texture', 'modular',
                 'level art'],
        'title': ['prop', 'asset', 'scene', 'environment', 'wip', 'ue4', 'ue5', 'unity', 'render', 'modeling', 'sculpt',
                  'tool', 'generator', 'game ready', 'material', 'shader', '作业', '练习', '高精', '道具', '临摹']
    }

    # 兼容处理：如果是Series直接get，如果是dict也是get
    software_txt = str(row.get('Software Used', '')).lower()
    tags_txt = str(row.get('Tags Used', '')).lower()
    title_txt = str(row.get('Artwork Title', '')).lower()
    if software_txt == 'nan': software_txt = ''

    score_2d = sum(3 for k in traits_2_d['software'] if k in software_txt) + sum(
        2 for k in traits_2_d['tags'] if k in tags_txt) + sum(1 for k in traits_2_d['title'] if k in title_txt)
    score_tech = sum(3 for k in traits_tech['software'] if k in software_txt) + sum(
        2 for k in traits_tech['tags'] if k in tags_txt) + sum(1 for k in traits_tech['title'] if k in title_txt)

    if 'groom' in tags_txt or 'xgen' in software_txt: score_tech += 5
    if re.search(r'[\u4e00-\u9fa5]', title_txt):
        if '作业' in title_txt or '道具' in title_txt or '练习' in title_txt: score_tech += 4
    return score_2d, score_tech


def inject_noise_to_standpoint(base_standpoint: list, noise_level=0.05) -> list:
    arr = np.array(base_standpoint)
    noise = np.random.normal(0, noise_level, size=arr.shape)
    new_arr = np.maximum(arr + noise, 0.01)
    final_arr = new_arr / new_arr.sum()
    return [round(x, 2) for x in final_arr.tolist()]


def generate_realistic_history(agent_type: str, days: int = 7) -> list:
    if agent_type == "水印破坏者":
        mu, sigma = -0.4, 0.2
    elif agent_type == "合规创作者":
        mu, sigma = 0.6, 0.1
    else:
        mu, sigma = 0.2, 0.15
    history = []
    current_val = max(-1.0, min(1.0, np.random.normal(mu, sigma)))
    history.append(round(current_val, 2))
    alpha = 0.3
    for _ in range(days - 1):
        noise = np.random.normal(0, 0.1)
        current_val = (1 - alpha) * current_val + alpha * mu + noise
        history.append(round(max(-1.0, min(1.0, current_val)), 2))
    return history


def calculate_influence_scientific(agents_list, alpha=3.0):
    print(f"正在为 {len(agents_list)} 个 Agent 计算相对影响力...")
    raw_scores = []
    for agent in agents_list:
        raw_metrics = agent['influence']
        v = clean_number(raw_metrics[0])
        l = clean_number(raw_metrics[1])
        c = clean_number(raw_metrics[2])
        score = v * 0.1 + l * 1.0 + c * 3.0
        raw_scores.append(score)
    s_scores = pd.Series(raw_scores)
    uniform_rank = s_scores.rank(pct=True, method='min')
    power_law_influence = np.power(uniform_rank, alpha)
    final_scores = power_law_influence * 0.99 + 0.01
    for i, agent in enumerate(agents_list):
        agent['influence'] = round(float(final_scores[i]), 2)
    return agents_list


async def get_result(role, company, software, tags, views, likes, title, comments, forced_type, forced_trait_kv):
    # 构建多样性提示词
    diversity_prompt = ""
    if forced_trait_kv:
        key, val = forced_trait_kv

        # 键名中文化，辅助理解
        key_cn_map = {
            "beta": "逆反心理(beta)",
            "fp_sensitivity": "误伤敏感度(fp_sensitivity)",
            "cost_sensitivity": "成本敏感度(cost_sensitivity)",
            "gamma": "信息茧房(gamma)"
        }
        key_cn = key_cn_map.get(key, key)

        diversity_prompt = f"""
       # 2. 关键指令：打破刻板印象，体现人性广度 (Diversity & Spectrum)
       **请注意：任何群体都不是一成不变的，人性具有极大的广度。**
       - “公众”不一定都是温顺的，也包含激进的反叛者。
       - “合规创作者”不一定都是理性的，也包含极度敏感的玻璃心。
       - “破坏者”不一定都是死士，也包含精打细算的投机者。

       为了保证社会仿真的真实性和完备性，系统**指定**该样本必须处于正态分布的特定位置：
       👉 **【{key_cn}】必须为："{val}"**

       **你的任务：**
       请在 `reasoning` 中，结合他的背景（{role}），合理化这一特征。
       - 如果这看起来反直觉（例如“高逆反的公众”），请解释为什么（例如：“他虽然只是个普通学生，但深受赛博朋克反抗精神影响...”）。
       - 最终输出的 `{key}` 字段必须严格等于 "{val}"。
       """

    prompt_str = f"""
       # Role (角色设定)
       你是一位计算社会学家。你正在构建 "2022年 ArtStation 社区" 的虚拟用户画像。

       # Task (任务)
       基于用户的历史元数据，生成他在AI爆发背景下的心理画像。

       # 1. Identity Constraint (身份约束)
       **系统已通过算法将该用户归类为: 【{{forced_type}}】**
       你生成的 `type` 字段必须严格等于 "{{forced_type}}"。
       请基于此身份重新解释他的背景数据。

       {diversity_prompt}

       # Input Data (输入数据)
       - **Role**: {{role}}
       - **Software**: {{software}}
       - **Tags**: {{tags}}
       - **Metrics**: Views {{views}}, Likes {{likes}}

       # Output Requirements
       1. **reasoning**: 必须解释为何符合【{{forced_type}}】，并重点解释为何具有指定的【性格特征】。
       2. **type**: 必须是 "{{forced_type}}"。

       {{output_format_instruction}}
       """

    output_format = JsonOutputParser(pydantic_object=OutputFormat)
    prompt = ChatPromptTemplate.from_template(template=prompt_str, partial_variables={
        "output_format_instruction": output_format.get_format_instructions()})

    current_llm = get_async_llm(model="qwen-max", temperature=0.5)

    agent = prompt | current_llm | output_format

    async with sem:
        try:
            response = await agent.ainvoke({
                "role": role, "company": company, "software": software, "tags": tags,
                "views": views, "likes": likes, "title": title, "comments": comments,
                "forced_type": forced_type
            })
        except Exception as e:
            print(f"LLM 生成出错: {e}, 跳过该条目")
            return None

    response['standpoint'] = inject_noise_to_standpoint(response['standpoint'], 0.1)

    beta_dict = {
        '高': '【天生反骨】你极度厌恶“被管理”和“被规训”。如果感觉到平台的审核之手伸得太长（哪怕是为了安全），你的第一反应是生理性的厌恶和逃离，而不是顺从。',
        '中': '【独立思考】你既不盲从权威，也不为了反抗而反抗。你会批判性地审视每一条规则：合理的你就遵守，不合理或愚蠢的你会冷眼旁观，并在心里扣分。',
        '低': '【秩序拥护者】你是一个温和的顺民。你倾向于信任平台和权威，认为严格的监管是维持社区秩序的必要手段。你甚至可能反感那些总是抱怨规则的人，认为他们是在添乱。'
    }
    gamma_dict = {
        '高': '【固执己见】你非常固执，是信息茧房的重度用户。一旦你对平台形成了既定印象（无论好坏），后续即使有相反的证据，你也倾向于视而不见，继续强化你原本的看法。',
        '中': '【有立场但讲理】你有自己的偏好，但不是瞎子。如果有强有力的事实摆在面前（例如连续多天看到糟糕的体验），你会慢慢修正自己的观点，虽然这个过程有点慢。',
        '低': '【绝对理性】你是一个冷酷的观察者。你几乎没有先入为主的偏见，只看当下的事实。你的态度会随着每天的实际体验而快速波动，不会陷入思维定势。'
    }
    fp_sensitivity_dict = {
        '高': '【玻璃心/极度敏感】你自尊心极强。哪怕只有一次微小的误解或误伤，在你的心里都会被放大成一种对你专业能力的羞辱和平台的背叛，引发强烈的愤怒。',
        '中': '【务实派/有底线】你是一个理性的人。由于技术的不成熟，你会容忍偶尔的错误，但如果错误成为常态，你的耐心会迅速耗尽。',
        '低': '【乐天派/钝感力】你心态非常开放且包容。你认为在AI时代，算法误判是技术发展的必经代价。只要不是恶意针对，你通常会一笑置之，不会因此产生强烈的负面情绪。'
    }
    cost_sensitivity_dict = {
        '高': '【精打细算】你极其看重投入产出比。你倾向于选择免费或低成本的攻击方案，即使成功率不是最高。如果攻击成本过高，你会果断放弃。',
        '中': '【追求性价比】你是一个务实的攻击者。你会在攻击成本（时间/金钱）和预期成功率之间寻找平衡点，不会盲目投入也不会一毛不拔。',
        '低': '【不惜代价】为了达成“规避检测”的终极目标，你愿意投入昂贵的计算资源或学习最复杂的技术。对你来说，为了赢，可以忽略一切成本。',
    }

    final_type = forced_type
    res = {
        "agent_id": str(response['name']).replace(' ', '_'),
        "name": str(response['name']).replace(' ', '_'),
        "type": final_type,
        "description": response['description'],
        "standpoint": response['standpoint'],
        "beta": beta_dict.get(response['beta'], beta_dict['中']),
        "gamma": gamma_dict.get(response['gamma'], gamma_dict['中']),
        "fp_sensitivity": fp_sensitivity_dict.get(response['fp_sensitivity'], fp_sensitivity_dict['中']),
        "cost_sensitivity": cost_sensitivity_dict.get(response['cost_sensitivity'], cost_sensitivity_dict['中']),
        "influence": [views, likes, comments],
        "satisfaction": generate_realistic_history(final_type),
        "post_wish": True,
        "is_active": True if final_type != "公众" else False,
        "beliefs": response['beliefs'],
        "social_relationships": {}
    }
    return res


# ==========================================
# 5. 分池与构建逻辑 (修复 KeyError 的核心)
# ==========================================

def prepare_candidates(csv_path):
    """
    读取CSV，计算分数，并将所有用户分配到三个潜在的池子中。
    """
    print("1. 读取并清洗 CSV...")
    df = pd.read_csv(csv_path)

    # 打印一下列名，确保我们没有拼错
    print(f"CSV 列名: {df.columns.tolist()}")

    df = df[2:]  # 还原你原本的逻辑：跳过前两行

    # 必要的列检查，防止 dropna 报错
    req_cols = ["Software Used", "Tags Used", "Artwork Title"]
    # 如果列存在才 dropna，增加鲁棒性
    valid_cols = [c for c in req_cols if c in df.columns]
    if valid_cols:
        df = df.dropna(how="any", subset=valid_cols)

    print("2. 计算倾向分...")
    scores = df.apply(analyze_user_dna, axis=1)
    df['score_2d'] = [s[0] for s in scores]
    df['score_tech'] = [s[1] for s in scores]

    # 计算原始影响力 (处理可能的列名不一致)
    views_col = '№ Views' if '№ Views' in df.columns else df.columns[3]
    likes_col = '№ Likes' if '№ Likes' in df.columns else df.columns[5]

    df['raw_influence'] = df[views_col].apply(clean_number) + df[likes_col].apply(clean_number) * 5

    # 转换为字典列表
    all_users = df.to_dict('records')

    # 分桶
    breaker_pool_candidates = []
    creator_pool_candidates = []
    public_pool_candidates = []

    for user in all_users:
        # 分数逻辑
        if user['score_2d'] > user['score_tech'] + 1:
            breaker_pool_candidates.append(user)
        elif user['score_tech'] >= user['score_2d']:
            creator_pool_candidates.append(user)

        # 公众池是兜底，包含所有人
        public_pool_candidates.append(user)

    # 排序
    breaker_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)
    creator_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)
    public_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)

    print(
        f"数据准备完成。候选池规模: Breaker({len(breaker_pool_candidates)}), Creator({len(creator_pool_candidates)}), Public({len(public_pool_candidates)})")

    return breaker_pool_candidates, creator_pool_candidates, public_pool_candidates


def generate_trait_distribution(total: int) -> List[str]:
    """生成强制均匀分布列表: 1:1:1"""
    if total <= 0: return []
    part = total // 3
    result = ['高'] * part + ['中'] * part + ['低'] * part
    while len(result) < total:
        result.append(random.choice(['高', '中', '低']))
    random.shuffle(result)
    return result


async def build_agent_pools(csv_path, output_dir, pool_sizes: Dict[str, int]):
    """
    构建三个互不重叠的智能体池。
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    breakers_cand, creators_cand, public_cand = prepare_candidates(csv_path)

    selected_tasks = []
    used_indices = set()

    # 生成 1:1:1 的分布列表
    traits_breaker = generate_trait_distribution(pool_sizes.get('breaker', 0))
    traits_creator = generate_trait_distribution(pool_sizes.get('creator', 0))
    traits_public = generate_trait_distribution(pool_sizes.get('public', 0))

    def add_batch(candidates, role, count, trait_key, trait_values):
        added = 0
        for i, user in enumerate(candidates):
            if added >= count: break
            uid = str(user.get('User Name', f"row_{i}_{random.randint(0, 999)}"))
            if uid not in used_indices:
                val = trait_values[added]
                selected_tasks.append({
                    "user": user,
                    "role": role,
                    "forced_trait": (trait_key, val)  # 传入要注入的属性键值对
                })
                used_indices.add(uid)
                added += 1
        return added

    print("3. 分配名单与多样性注入...")
    b_count = add_batch(breakers_cand, "水印破坏者", pool_sizes.get('breaker', 0), "cost_sensitivity", traits_breaker)
    c_count = add_batch(creators_cand, "合规创作者", pool_sizes.get('creator', 0), "fp_sensitivity", traits_creator)
    p_count = add_batch(public_cand, "公众", pool_sizes.get('public', 0), "beta", traits_public)

    print(f"   - 计划生成: 破坏者{b_count}, 合规者{c_count}, 公众{p_count}")

    print(f"4. 并发生成 {len(selected_tasks)} 个 Persona...")
    tasks = []
    for item in selected_tasks:
        user_row = item['user']
        company_val = user_row.get('Company') or user_row.get('Company Work at') or "Freelance"

        tasks.append(get_result(
            role=user_row.get('Role', 'Artist'),
            company=company_val,
            software=user_row.get('Software Used', ''),
            tags=user_row.get('Tags Used', ''),
            views=user_row.get('№ Views', 0),
            likes=user_row.get('№ Likes', 0),
            title=user_row.get('Artwork Title', 'Untitled'),
            comments=user_row.get('№ Comments', 0),
            forced_type=item['role'],
            forced_trait_kv=item['forced_trait']  # 传入属性
        ))

    raw_results = await asyncio.gather(*tasks)
    valid_results = [r for r in raw_results if r is not None]

    print("5. 结果ID去重与归类...")
    global_id_set = set()
    final_pools = {"水印破坏者": [], "合规创作者": [], "公众": []}

    for p in valid_results:
        original_id = p['agent_id']
        unique_id = original_id
        counter = 1
        while unique_id in global_id_set:
            unique_id = f"{original_id}_{counter}"
            p['name'] = f"{p['name']}_{counter}"
            counter += 1
        p['agent_id'] = unique_id
        global_id_set.add(unique_id)
        final_pools[p['type']].append(p)

    for pool_name, agents in final_pools.items():
        if not agents: continue
        processed = calculate_influence_scientific(agents)
        fname = {"水印破坏者": "pool_breakers.json", "合规创作者": "pool_compliance.json",
                 "公众": "pool_public.json"}.get(pool_name)
        save_path = os.path.join(output_dir, fname)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)

        # 统计分布
        target_key = "cost_sensitivity" if "破坏" in pool_name else (
            "fp_sensitivity" if "合规" in pool_name else "beta")
        dist = {'高': 0, '中': 0, '低': 0}
        for a in processed:
            val = a.get(target_key, '中')
            dist[val] = dist.get(val, 0) + 1

        print(f"✅ 已保存 [{pool_name}] ({len(processed)}人) -> {fname}")
        print(f"   📊 属性 [{target_key}] 分布: {dist}")


def build_agent_pools_demo(breaker_num, creator_num, public_num):
    """
    构建智能体池子，入口函数
    :param breaker_num: 水印破坏者数量
    :param creator_num: 创作者数量
    :param public_num: 公众数量
    :return:
    """
    # 定义需要的池子大小 (建议大一点备用)
    REQUESTED_SIZES = {
        'breaker': breaker_num,
        'creator': creator_num,
        'public': public_num
    }

    CSV_FILE = r'method\data\artstation_main_data.csv'
    OUTPUT_FOLDER = r'method\data\pools'

    asyncio.run(build_agent_pools(CSV_FILE, OUTPUT_FOLDER, REQUESTED_SIZES))

