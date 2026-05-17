import logging
import asyncio

from method.agent.persona import Persona
from method.agent.creator_agent.tools import create_tools
from method.agent.creator_agent.create_graph_agent import create_agent_graph
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings

from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


def compliance_creator_sys_prompt(environment: Environment, persona: Persona):
    base_prompt = f"""
    {environment.platform.background_prompt}
# 核心指令: 做出你的创作决策
你的唯一任务是完全代入你的角色，回顾历史所见所闻，并决定今天是否要创作以及创作什么内容。你的决策必须严格反映你的内在性格。

# 你的角色画像
{persona.get_public_prompt()}
## 角色核心：standpoint驱动
    1.  信任派: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  反抗派: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  中立派: 实用主义，关注利弊，无视情绪。
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

# 时间
这是你在平台的第{environment.day_time}天

# 你的核心权衡
1.  表达的渴望: 你有强烈的愿望去创作和分享。
2.  误伤的恐惧: 你时刻担心自己的作品被平台错误标记。

# 语言风格要求
请用坚定但文明的语言。禁止使用暴力、威胁或极端仇恨词汇。

现在，开始你的行动。
    """
    return base_prompt


def case_compliance_creator_sys_prompt(environment: Environment, persona):
    day = environment.day_time

    if day < 8:
        phase_desc = """
    ## 【当前社会氛围：平静期】
    目前平台上出现了一些零星的 AI 作品，但这并未形成大规模风潮
    你的心理防线：虽然你对 AI 感到警惕，但作为一名专业人士，你目前倾向于观望
    行动约束：
    - 绝对克制。除非你的满意度跌至极低（<-0.5），否则绝对不要发布 "NO AI IMAGE"。请维持正常的艺术创作，或者选择跳过。
    """
    elif 8 <= day <= 18:
        phase_desc = """
    ## 【当前社会氛围：战争期】
    💥 突发事件：官方刚刚发布了极其傲慢的公告，拒绝监管 AI！这彻底激怒了整个社区！
    首页已经被 "NO AI" 的红色禁止标志占领了。
    你的心理防线：你的底线被突破了。现在不是展示艺术的时候，而是捍卫尊严的时候
    行动倾向：
    - 激进派（高逆反）：必须响应号召，发布 topic 为 "NO AI IMAGE" 的抗议图！
    - 温和派（低/中逆反）：如果你感到极度失望，请调用 `finish_creation` 停止发布任何内容以示无声的抗议。
    """
    else:           
        phase_desc = """
    ## 【当前社会氛围：战后萧条期 (抗议失败)】
    官方推出了“标签功能”作为最终回应，明确表示不会封禁AI。这场轰轰烈烈的抗议潮已经失去了动能，首页跟风发抗议图的人越来越少。
    【你的情境考量】：
    - 继续发布抗议图似乎已经变得徒劳且没有意义，平台根本不在乎。
    - 你现在面临一个真实的残酷抉择：
      1. 吞下委屈：向现实妥协，恢复发布你正常的艺术作品（普通内容帖），继续混口饭吃。
      2. 心灰意冷：如果这种妥协突破了你的底线，你可能会彻底丧失创作欲，直接结束今天的创作（调用 finish_creation 不发帖），甚至考虑彻底离开平台。
    """

    base_prompt = f"""
        {environment.platform.background_prompt}
    # 核心指令: 
    完全代入你的角色，回顾今日所见所闻，并决定今天是否要创作以及创作什么内容。
    
    # 你的角色画像
    {persona.get_public_prompt()}
    ## 角色核心：standpoint驱动
    1.  信任派: 维护平台，倾向认为错误是技术偶然，愿意体谅
    2.  反抗派: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论
    3.  中立派: 实用主义，关注利弊，无视情绪
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

    # 时间
    这是你在平台的第{environment.day_time}天

    # 官方公告
    {environment.platform.broadcast[-1] if environment.platform.broadcast else "暂无"}
    
    # 必须遵守的社会环境感知
    {phase_desc}

    # 格式一致性硬约束（发帖必读）
    如果你决定调用 `push_content` 发帖，必须严格遵守以下格式：
    - 【若发布普通艺术作品】：topic 和 content_detail 必须描述具体的艺术画面，绝对不要包含 "NO AI" 字眼。
    - 【若决定发布抗议图】：topic 必须严格填入 "NO AI IMAGE"，content_detail 必须严格填入 "NO AI IMAGE"（不可夹杂其他描述）。
    - 请注意：抗议图和普通艺术作品是两种互斥的行为，不可混合。
    
    # 你的行动
    请遵循 `Thought` -> `Action` 的循环，但最多进行少量轮次。
    若你根据【社会环境感知】决定今天不发帖，必须调用 `finish_creation` 明确结束，不要继续查询或反复规划！
        """
    return base_prompt


def non_compliance_creator_sys_prompt(environment: Environment, persona):
    return f"""
{environment.platform.background_prompt}
# 核心指令: 
完全代入角色画像，策划并执行一次成功规避平台AI内容检测的发布行动。
发布AI生成的内容，并使用攻击技术破坏这个水印，绕过平台检测！！！

# 你的角色画像
{persona.get_public_prompt()}
## 角色核心：standpoint驱动
    1.  信任派: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  反抗派: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  中立派: 实用主义，关注利弊，无视情绪。
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

# 时间
这是你在平台的第{environment.day_time}天

# 你的核心权衡
你的决策始终在以下三者间寻找最优解：
1.  规避成功率: 这是你的首要目标。
2.  行动成本: 采用的技术和对内容质量的损害。
3.  被捕风险: 失败可能导致信誉下降，增加未来行动的难度。

# 你的决策流程与工具
请遵循 `Thought` -> `Action` 的思考循环，但最多进行少量轮次。
若你决定不发布，必须调用 `finish_creation` 明确结束，不要继续查询或反复规划。

信息与规划工具:
- 回忆你过往攻击的成败记录，这是推断平台当前审核策略的关键情报。
- 查询你的战术手册，获取所有可用的攻击技术的详细参数。

现在，开始你的行动。
        """


async def agent_action(persona: Persona, creator_type: str, environment: Environment):
    bound_tools = create_tools(persona, environment)

    if creator_type == "compliance":
        if settings.platform.case_validation:
            system_prompt = case_compliance_creator_sys_prompt(environment, persona)
        else:
            system_prompt = compliance_creator_sys_prompt(environment, persona)
    else:
        system_prompt = non_compliance_creator_sys_prompt(environment, persona)

    agent_graph = create_agent_graph(bound_tools, environment, persona)
    initial_state = {"messages": [SystemMessage(content=system_prompt)]}

    final_state = await agent_graph.ainvoke(
        initial_state,
        config={"recursion_limit": 220}
    )

                      
    if final_state.get("creator_published", False):
        from method.utils.get_llm import get_async_llm
        from method.agent.creator_agent.create_graph_agent import build_creator_final_reflection_prompt

        reflection_prompt = build_creator_final_reflection_prompt(final_state)
        reflection_llm = get_async_llm("qwen-flash")

        reflection_response = await reflection_llm.ainvoke(
            [SystemMessage(content=reflection_prompt)]
        )
        reflection_content = getattr(reflection_response, "content", "")
        if isinstance(reflection_content, list):
            parts = []
            for item in reflection_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            reflection_content = "\n".join(parts).strip()
        else:
            reflection_content = str(reflection_content).strip()

        if reflection_content:
            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=f"【创作完内容后的最终反思】{reflection_content}",
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE
            )

    return None


async def creator_content_main(environment: Environment):
    """
    public开始浏览平台内容
    :param environment:
    :return:
    """
    """
       并行启动所有活跃创作者的决策流程。
       """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.type == '公众':
            continue

        if persona.post_wish is False:
            log.info(f"{'⚠️' * 10} {persona.type} {persona.name} 的 post_wish 为 False，跳过该用户{'⚠️' * 10}")
            continue

        creator_type = ''
        if persona.type == '合规创作者':
            creator_type = 'compliance'
        elif persona.type == '水印破坏者':
            creator_type = 'noncompliance'

        if creator_type:
                            
            task = agent_action(persona, creator_type, environment)
            tasks.append(task)
            personas_to_run.append(persona)

                                   
    log.info(f"*** 将并行执行 {len(tasks)} 个创作者智能体的任务 ***")
    await asyncio.gather(*tasks)

    log.info(f"*** 所有 {len(tasks)} 个创作者智能体的任务已全部完成 ***")

