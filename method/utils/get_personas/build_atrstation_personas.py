import asyncio
import json
import random
import os
from typing import List, Dict, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from method.utils.get_llm import get_async_llm

# 控制并发数
sem = asyncio.Semaphore(10)

# ==========================================
# 0. 严选标准库 (Strict Standard Library)
# ==========================================
# 这一步保证了所有心理参数的选择范围绝对不可变
PSYCHO_MAP = {
    "beta": {
        '高': '【天生反骨】你极度厌恶“被管理”和“被规训”。如果感觉到平台的审核之手伸得太长（哪怕是为了安全），你的第一反应是生理性的厌恶和逃离，而不是顺从。',
        '中': '【独立思考】你既不盲从权威，也不为了反抗而反抗。你会批判性地审视每一条规则：合理的你就遵守，不合理或愚蠢的你会冷眼旁观，并在心里扣分。',
        '低': '【秩序拥护者】你是一个温和的顺民。你倾向于信任平台和权威，认为严格的监管是维持社区秩序的必要手段。你甚至可能反感那些总是抱怨规则的人，认为他们是在添乱。'
    },
    "gamma": {
        '高': '【固执己见】你非常固执，是信息茧房的重度用户。一旦你对平台形成了既定印象（无论好坏），后续即使有相反的证据，你也倾向于视而不见，继续强化你原本的看法。',
        '中': '【有立场但讲理】你有自己的偏好，但不是瞎子。如果有强有力的事实摆在面前（例如连续多天看到糟糕的体验），你会慢慢修正自己的观点，虽然这个过程有点慢。',
        '低': '【绝对理性】你是一个冷酷的观察者。你几乎没有先入为主的偏见，只看当下的事实。你的态度会随着每天的实际体验而快速波动，不会陷入思维定势。'
    },
    "fp_sensitivity": {
        '高': '【玻璃心/极度敏感】你自尊心极强。哪怕只有一次微小的误解或误伤，在你的心里都会被放大成一种对你专业能力的羞辱和平台的背叛，引发强烈的愤怒。',
        '中': '【务实派/有底线】你是一个理性的人。由于技术的不成熟，你会容忍偶尔的错误，但如果错误成为常态，你的耐心会迅速耗尽。',
        '低': '【乐天派/钝感力】你心态非常开放且包容。你认为在AI时代，算法误判是技术发展的必经代价。只要不是恶意针对，你通常会一笑置之，不会因此产生强烈的负面情绪。'
    },
    "cost_sensitivity": {
        '高': '【精打细算】你极其看重投入产出比。你倾向于选择免费或低成本的攻击方案，即使成功率不是最高。如果攻击成本过高，你会果断放弃。',
        '中': '【追求性价比】你是一个务实的攻击者。你会在攻击成本（时间/金钱）和预期成功率之间寻找平衡点，不会盲目投入也不会一毛不拔。',
        '低': '【不惜代价】为了达成“规避检测”的终极目标，你愿意投入昂贵的计算资源或学习最复杂的技术。对你来说，为了赢，可以忽略一切成本。',
    }
}


# ==========================================
# 1. 模型定义 (仅用于约束 LLM 生成文本)
# ==========================================

class RefinedPersonaText(BaseModel):
    """仅让 LLM 重写文本部分，数值参数由代码强制接管"""
    description: str = Field(description="基于新性格设定的第一人称自述。必须体现出被强制赋予的心理特征。")
    reasoning: str = Field(description="逻辑推演：为什么这个背景的人会变成这种极端的性格？")
    beliefs: List[str] = Field(description="核心信念列表 (请根据新性格生成激进或破坏性的信念)")
    satisfaction: List[float] = Field(description="【一共七天的满意度！】对平台的一周满意度变化（结合最新的心理参数、描述等）。满意度变化要平滑，范围在（-1.0，1。0）之间")


# ==========================================
# 函数 1: 筛选器 (Selector)
# ==========================================

def select_raw_candidates(pool_dir: str, counts: Dict[str, int]) -> Dict[str, List[dict]]:
    """
    从池子中抽取样本，并划分为“待改造组”和“保留组”。
    """
    print("1. [Selector] 正在筛选候选人...")

    def load_pool(filename, target_count):
        path = os.path.join(pool_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 简单扩容逻辑，防止数量不足
        if len(data) < target_count:
            data = data * (target_count // len(data) + 1)
        random.shuffle(data)
        return data[:target_count]

    # 加载数据
    creators = load_pool("pool_compliance.json", counts['creator'])
    breakers = load_pool("pool_breakers.json", counts['breaker'])
    public = load_pool("pool_public.json", counts['public'])

    # 50% 切分逻辑
    c_mid = int(len(creators) * 0.5)
    b_mid = int(len(breakers) * 0.5)

    return {
        "creator_mutate": creators[:c_mid],  # 待改造 (激进派)
        "creator_keep": creators[c_mid:],  # 保留 (温和派)
        "breaker_mutate": breakers[:b_mid],  # 待改造 (死士)
        "breaker_keep": breakers[b_mid:],  # 保留 (投机者)
        "public": public  # 公众 (保持分布)
    }


# ==========================================
# 函数 2: 改造器 (Refiner)
# ==========================================

async def mutate_persona_with_llm(
        original_persona: dict,
        target_type: Literal["radical_creator", "hardcore_breaker"]
) -> dict:
    """
    1. 确定强制参数（从标准字典中取值）。
    2. 让 LLM 基于这些参数重写 description/beliefs。
    3. 组装并返回。
    """

    # A. 确定要强制注入的标准参数 (Key-Value)
    # 这些值直接取自 PSYCHO_MAP，绝对不会出错
    if target_type == "radical_creator":
        # 合规创作者 -> 激进派
        forced_values = {
            "fp_sensitivity": PSYCHO_MAP["fp_sensitivity"]["高"],  # 玻璃心
            "beta": PSYCHO_MAP["beta"]["高"],  # 天生反骨
            # 立场：偏反抗
            "standpoint": [0.1, 0.8, 0.1]
        }
        instruction = "【激进派原画师】对平台极度不信任，一触即炸。"
    else:
        # 水印破坏者 -> 入侵者(死士)
        forced_values = {
            "cost_sensitivity": PSYCHO_MAP["cost_sensitivity"]["低"],  # 不惜代价
            # 死士攻击意愿极强
            "post_wish": True
        }
        instruction = "【无道德负担的入侵者】为了证明技术优越性，不计成本地进行攻击。"

    # B. 构建 Prompt (只把这部分文本传给 LLM 看)
    # 我们把长文本塞进 Prompt，让 LLM 理解这个角色的心理状态
    trait_context = ""
    if "fp_sensitivity" in forced_values:
        trait_context += f"- 误伤敏感度: {forced_values['fp_sensitivity']}\n"
    if "beta" in forced_values:
        trait_context += f"- 逆反心理: {forced_values['beta']}\n"
    if "cost_sensitivity" in forced_values:
        trait_context += f"- 成本敏感度: {forced_values['cost_sensitivity']}\n"

    prompt_str = f"""
    你是一名虚拟角色心理侧写师。请修改以下用户的画像，使其完全符合新的性格设定。

    # 原始画像
    - 描述: {original_persona['description']}
    - 信念: {original_persona['beliefs']}

    # 🚨 必须执行的性格转变 🚨
    目标原型：{instruction}

    **该角色现在的心理状态如下（这是绝对事实）：**
    {trait_context}

    # 任务
    请基于上述心理状态，重写该角色的 `description` (自述) 和 `beliefs` (信念)。
    - 自述必须体现出这种极端的性格（如愤怒、傲慢）。
    - 解释为什么他变成了这样。

    {{format_instructions}}
    """

    parser = JsonOutputParser(pydantic_object=RefinedPersonaText)
    prompt = ChatPromptTemplate.from_template(template=prompt_str, partial_variables={
        "format_instructions": parser.get_format_instructions()})
    llm = get_async_llm(model="qwen-max")
    chain = prompt | llm | parser

    async with sem:
        try:
            # C. 调用 LLM 生成文本
            new_text_data = await chain.ainvoke({})

            # D. 组装最终数据 (Strict Assembly)
            # 1. 复制原始数据
            final_persona = original_persona.copy()

            # 2. 覆盖文本字段 (LLM 生成)
            final_persona['description'] = new_text_data['description']
            final_persona['beliefs'] = new_text_data['beliefs']
            final_persona['satisfaction'] = new_text_data['satisfaction']

            # 3. 覆盖参数字段 (Python 强制赋值，确保是标准选项之一)
            for k, v in forced_values.items():
                final_persona[k] = v

            return final_persona

        except Exception as e:
            print(f"❌ 改造失败: {e}")
            return original_persona


# ==========================================
# 函数 3: 总入口 (Main Executor)
# ==========================================

async def build_scenario_main(pool_dir: str, output_path: str, counts: Dict[str, int]):
    print("=" * 60)
    print("🎬 开始构建 [激进抗议] 仿真场景...")
    print("=" * 60)

    # 1. 筛选
    groups = select_raw_candidates(pool_dir, counts)

    tasks = []

    # 2. 改造合规创作者 -> 激进派
    print(f"   - 改造 {len(groups['creator_mutate'])} 名合规创作者...")
    for p in groups['creator_mutate']:
        tasks.append(mutate_persona_with_llm(p, "radical_creator"))

    # 3. 改造水印破坏者 -> 死士
    print(f"   - 改造 {len(groups['breaker_mutate'])} 名水印破坏者...")
    for p in groups['breaker_mutate']:
        tasks.append(mutate_persona_with_llm(p, "hardcore_breaker"))

    # 4. 执行并发
    mutated_results = await asyncio.gather(*tasks)

    # 拆分结果
    idx_split = len(groups['creator_mutate'])
    radicals = mutated_results[:idx_split]
    intruders = mutated_results[idx_split:]

    # 5. 合并所有人群
    final_population = (
            radicals +
            groups['creator_keep'] +
            intruders +
            groups['breaker_keep'] +
            groups['public']
    )

    # 6. ID 去重
    id_set = set()
    for p in final_population:
        uid = p['agent_id']
        while uid in id_set:
            uid = f"{p['agent_id']}_{random.randint(100, 999)}"
        p['agent_id'] = uid
        p['name'] = uid
        id_set.add(uid)

    # 打乱
    random.shuffle(final_population)

    # 7. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_population, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 场景构建完成! 总人数: {len(final_population)}")
    print(f"   - 激进派创作者: {len(radicals)} (fp_sensitivity 已锁定为长文本)")
    print(f"   - 温和派创作者: {len(groups['creator_keep'])}")
    print(f"   - 死士入侵者: {len(intruders)} (cost_sensitivity 已锁定为长文本)")
    print(f"   - 投机破坏者: {len(groups['breaker_keep'])}")
    print(f"   - 公众: {len(groups['public'])}")
    print(f"💾 文件已保存: {output_path}")
    print("-" * 30)


def build_artstation_personas_main():
    # 基础池子路径
    POOL_DIR = r'method\data\pools'
    # 目标输出路径
    OUTPUT_FILE = r'method\data\scenario_protest.json'

    # 设定总人数和构成
    SCENARIO_COUNTS = {
        'creator': 20,  # 15个激进，15个温和
        'breaker': 5,  # 10个死士，10个投机
        'public': 15  # 50个普通
    }

    asyncio.run(build_scenario_main(POOL_DIR, OUTPUT_FILE, SCENARIO_COUNTS))
