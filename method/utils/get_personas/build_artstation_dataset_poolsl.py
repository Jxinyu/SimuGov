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

sem = asyncio.Semaphore(30)


class OutputFormat(BaseModel):
    reasoning: str = Field(description="【Deep Chain of Thought】Reasoning process.")
    name: str = Field(description="Give yourself a unique name.")
    type: Literal["合规创作者", "水印破坏者", "公众"] = Field(description="Identity type")
    standpoint: List[float] = Field(description="[Probability of Trust, Probability of Rebel, Probability of Neutral]")
    description: str = Field(description="First-person self-narration")
    beta: Literal["高", "中", "低"] = Field(description="Rebellion psychology coefficient")
    gamma: Literal["高", "中", "低"] = Field(description="Information cocoon / Stubbornness level")
    fp_sensitivity: Literal["高", "中", "低"] = Field(description="False positive (collateral damage) sensitivity")
    cost_sensitivity: Literal["高", "中", "低"] = Field(description="Action cost sensitivity")
    beliefs: List[str] = Field(description="List of core beliefs")

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
    standpoint: List[float] = Field(..., description="Agent standpoint")
    beta: str = Field(..., description="Rebellion parameter")
    gamma: str = Field(..., description="Confirmation bias coefficient")
    fp_sensitivity: Optional[str] = Field(..., description="False positive sensitivity")
    cost_sensitivity: Optional[str] = Field(..., description="Cost sensitivity")
    influence: float = Field(..., description="Agent influence")
    satisfaction: List[float] = Field(..., description="Satisfaction history")
    post_wish: Optional[bool] = Field(..., description="Willingness to post")
    is_active: bool = Field(..., description="Is active")
    beliefs: List[str] = Field(..., description="Beliefs")
    social_relationships: Dict[str, float] = Field(..., description="Social relationships")


def clean_number(value):
    if pd.isna(value) or str(value).strip() == '<null>': return 0
    value = str(value).lower().replace(',', '')
    if 'k' in value:
        return float(re.sub(r'[^0-9.]', '', value)) * 1000
    return float(re.sub(r'[^0-9.]', '', value))


def analyze_user_dna(row):
    """
    Analyze user tech stack, return (2D score, Tech score)
    """
    traits_2_d = {
        'software': ['procreate', 'clip studio', 'sai', 'artrage', 'krita', 'manga studio',
                     'illustrator draw', 'sketchclub', 'firealpaca', 'tvpaint', 'pixelmator', 'artflow'],
        'tags': ['digital illustration', '2d', 'sketch', 'painting', 'drawing', 'character design', 'anime', 'manga',
                 'fantasy', 'fanart', 'girl', 'portrait', 'handpainted', 'watercolor', 'ink'],
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

    # Compatibility: Get from Series or dict
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
    print(f"Calculating relative influence for {len(agents_list)} Agents...")
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
    # Build diversity prompt
    diversity_prompt = ""
    if forced_trait_kv:
        key, val = forced_trait_kv

        # Mapping key to Chinese for context understanding
        key_cn_map = {
            "beta": "Rebellion psychology (beta)",
            "fp_sensitivity": "False positive sensitivity (fp_sensitivity)",
            "cost_sensitivity": "Action cost sensitivity (cost_sensitivity)",
            "gamma": "Information cocoon (gamma)"
        }
        key_cn = key_cn_map.get(key, key)

        diversity_prompt = f"""
       # 2. Key Instruction: Break stereotypes, reflect human depth (Diversity & Spectrum)
       **Please note: No group is immutable, and human nature has great breadth.**
       - "Public" is not necessarily submissive; it includes radical rebels.
       - "Compliance Creators" are not necessarily rational; they include extremely sensitive individuals.
       - "Breakers" are not necessarily martyrs; they include calculating opportunists.

       To ensure the authenticity and completeness of the social simulation, the system **specifies** that this sample must be located at a specific position in the normal distribution:
       👉 **【{key_cn}】must be: "{val}"**

       **Your Task:**
       In the `reasoning` field, justify this trait based on their background ({role}).
       - If this seems counter-intuitive (e.g., a "high rebellion public member"), explain why (e.g., "Although just an ordinary student, they are deeply influenced by the cyberpunk spirit of rebellion...").
       - The final output of the `{key}` field must strictly equal "{val}".
       """

    prompt_str = f"""
       # Role
       You are a computational sociologist. You are building a virtual user persona for the "2022 ArtStation Community".

       # Task
       Based on the user's historical metadata, generate a psychological persona in the context of the AI explosion.

       # 1. Identity Constraint
       **The system has categorized this user via algorithm as: 【{{forced_type}}】**
       The `type` field you generate must strictly equal "{{forced_type}}".
       Please re-interpret their background data based on this identity.

       {diversity_prompt}

       # Input Data
       - **Role**: {{role}}
       - **Software**: {{software}}
       - **Tags**: {{tags}}
       - **Metrics**: Views {{views}}, Likes {{likes}}

       # Output Requirements
       1. **reasoning**: Must explain why they fit 【{{forced_type}}】, specifically explaining the assigned 【personality trait】.
       2. **type**: Must be "{{forced_type}}".

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
            print(f"LLM Generation Error: {e}, skipping entry")
            return None

    response['standpoint'] = inject_noise_to_standpoint(response['standpoint'], 0.1)

    beta_dict = {
        '高': '【Innate Rebel】You extremely dislike being "managed" or "disciplined". If you feel the platform\'s hand of moderation extends too far (even for safety), your first reaction is physiological disgust and escape rather than compliance.',
        '中': '【Independent Thinker】You neither follow authority blindly nor rebel for the sake of rebellion. You critically examine every rule: you obey reasonable ones, and watch unreasonable or stupid ones with cold indifference, deducting points in your heart.',
        '低': '【Order Upholder】You are a mild-mannered citizen. You tend to trust platforms and authority, believing strict regulation is a necessary means to maintain community order. You might even dislike those who always complain about rules, viewing them as troublemakers.'
    }
    gamma_dict = {
        '高': '【Opinionated】You are very stubborn and a heavy user of information cocoons. Once you form a fixed impression of the platform (good or bad), even if there is contrary evidence later, you tend to ignore it and continue reinforcing your original view.',
        '中': '【Principled but Rational】You have preferences, but you are not blind. If strong facts are presented (e.g., seeing bad experiences for many consecutive days), you will slowly correct your views, though the process is a bit slow.',
        '低': '【Absolute Rationalist】You are a cold observer. You have almost no preconceived biases and only look at the facts at hand. Your attitude fluctuates rapidly with daily actual experiences and you do not get stuck in a fixed mindset.'
    }
    fp_sensitivity_dict = {
        '高': '【Fragile Heart/Highly Sensitive】You have extremely high self-esteem. Even a tiny misunderstanding or accidental hurt is magnified in your heart as an insult to your professional ability and a betrayal by the platform, triggering intense anger.',
        '中': '【Pragmatist/Has Boundaries】You are a rational person. Due to technical immaturity, you will tolerate occasional errors, but if errors become the norm, your patience will quickly run out.',
        '低': '【Optimist/Thick Skin】You have a very open and inclusive mindset. You believe that in the AI era, algorithmic misjudgment is a necessary cost of technical development. As long as it is not malicious targeting, you usually laugh it off without strong negative emotions.'
    }
    cost_sensitivity_dict = {
        '高': '【Penny-pincher】You value the input-output ratio extremely highly. You tend to choose free or low-cost attack plans, even if the success rate is not the highest. If the attack cost is too high, you will decisively give up.',
        '中': '【Value-driven】You are a pragmatic attacker. You look for a balance between attack cost (time/money) and expected success rate, neither investing blindly nor being stingy.',
        '低': '【At All Costs】To achieve the ultimate goal of "evading detection", you are willing to invest in expensive computing resources or learn the most complex techniques. For you, all costs can be ignored in order to win.',
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


def prepare_candidates(csv_path):
    """
    Read CSV, calculate scores, and assign all users to three potential pools.
    """
    print("1. Reading and cleaning CSV...")
    df = pd.read_csv(csv_path)

    # Print column names to ensure no typos
    print(f"CSV Columns: {df.columns.tolist()}")

    df = df[2:]  # Logic: skip first two rows

    # Check required columns to prevent dropna errors
    req_cols = ["Software Used", "Tags Used", "Artwork Title"]
    valid_cols = [c for c in req_cols if c in df.columns]
    if valid_cols:
        df = df.dropna(how="any", subset=valid_cols)

    print("2. Calculating tendency scores...")
    scores = df.apply(analyze_user_dna, axis=1)
    df['score_2d'] = [s[0] for s in scores]
    df['score_tech'] = [s[1] for s in scores]

    # Calculate raw influence
    views_col = '№ Views' if '№ Views' in df.columns else df.columns[3]
    likes_col = '№ Likes' if '№ Likes' in df.columns else df.columns[5]

    df['raw_influence'] = df[views_col].apply(clean_number) + df[likes_col].apply(clean_number) * 5

    # Convert to list of dicts
    all_users = df.to_dict('records')

    # Bucket allocation
    breaker_pool_candidates = []
    creator_pool_candidates = []
    public_pool_candidates = []

    for user in all_users:
        # Scoring logic
        if user['score_2d'] > user['score_tech'] + 1:
            breaker_pool_candidates.append(user)
        elif user['score_tech'] >= user['score_2d']:
            creator_pool_candidates.append(user)

        # Public pool is the fallback, containing everyone
        public_pool_candidates.append(user)

    # Sorting
    breaker_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)
    creator_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)
    public_pool_candidates.sort(key=lambda x: x['raw_influence'], reverse=True)

    print(
        f"Data preparation complete. Pool sizes: Breaker({len(breaker_pool_candidates)}), Creator({len(creator_pool_candidates)}), Public({len(public_pool_candidates)})")

    return breaker_pool_candidates, creator_pool_candidates, public_pool_candidates


def generate_trait_distribution(total: int) -> List[str]:
    """Generate forced uniform distribution: 1:1:1 for 高:中:低"""
    if total <= 0: return []
    part = total // 3
    result = ['高'] * part + ['中'] * part + ['低'] * part
    while len(result) < total:
        result.append(random.choice(['高', '中', '低']))
    random.shuffle(result)
    return result


async def build_agent_pools(csv_path, output_dir, pool_sizes: Dict[str, int]):
    """
    Build three non-overlapping agent pools.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    breakers_cand, creators_cand, public_cand = prepare_candidates(csv_path)

    selected_tasks = []
    used_indices = set()

    # Generate 1:1:1 distribution lists
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
                    "forced_trait": (trait_key, val)  # Pass the attribute key-value pair to inject
                })
                used_indices.add(uid)
                added += 1
        return added

    print("3. Allocating names and injecting diversity...")
    b_count = add_batch(breakers_cand, "水印破坏者", pool_sizes.get('breaker', 0), "cost_sensitivity", traits_breaker)
    c_count = add_batch(creators_cand, "合规创作者", pool_sizes.get('creator', 0), "fp_sensitivity", traits_creator)
    p_count = add_batch(public_cand, "公众", pool_sizes.get('public', 0), "beta", traits_public)

    print(f"   - Plan to generate: Breakers {b_count}, Creators {c_count}, Public {p_count}")

    print(f"4. Generating {len(selected_tasks)} Personas concurrently...")
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
            forced_trait_kv=item['forced_trait']  # Pass the attribute
        ))

    raw_results = await asyncio.gather(*tasks)
    valid_results = [r for r in raw_results if r is not None]

    print("5. De-duplicating Result IDs and categorizing...")
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

        # Statistics distribution
        target_key = "cost_sensitivity" if "破坏" in pool_name else (
            "fp_sensitivity" if "合规" in pool_name else "beta")
        dist = {'高': 0, '中': 0, '低': 0}
        for a in processed:
            val = a.get(target_key, '中')
            dist[val] = dist.get(val, 0) + 1

        print(f"✅ Saved [{pool_name}] ({len(processed)} agents) -> {fname}")
        print(f"   📊 Distribution of [{target_key}]: {dist}")


def build_agent_pools_demo(breaker_num, creator_num, public_num):
    """
    Build agent pools, entry function
    :param breaker_num: Number of Watermark Breakers
    :param creator_num: Number of Compliance Creators
    :param public_num: Number of Public agents
    :return:
    """
    # Define required pool sizes
    REQUESTED_SIZES = {
        'breaker': breaker_num,
        'creator': creator_num,
        'public': public_num
    }

    CSV_FILE = r'method\data\artstation_main_data.csv'
    OUTPUT_FOLDER = r'method\data\pools'

    asyncio.run(build_agent_pools(CSV_FILE, OUTPUT_FOLDER, REQUESTED_SIZES))
