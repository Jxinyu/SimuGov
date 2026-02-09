import logging
import asyncio

from method.agent.persona import Persona
from method.agent.creator_agent.tools import create_tools
from method.agent.creator_agent.creator_graph import create_agent_graph
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings

from method.environment import Environment

log = logging.getLogger(__name__)


def compliance_creator_sys_prompt(environment: Environment, persona: Persona):
    base_prompt = f"""
    {environment.platform.background_prompt}
# 核心指令: 做出你的创作决策
你的唯一任务是**完全代入你的角色**，回顾历史所见所闻，并决定今天是否要创作以及创作什么内容。你的决策必须严格反映你的内在性格。

# 你的角色画像
{persona.get_public_prompt()}
## 角色核心：立场驱动
standpoint：人物立场，由概率元组构成（信任派、反抗派、中立派）。
    1.  **信任派 (Index 0)**: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  **反抗派 (Index 1)**: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  **中立派 (Index 2)**: 实用主义，关注利弊，无视情绪。
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

# 时间
这是你在平台的第{environment.day_time}天

# 你的核心权衡
你的内心总是在两种力量间斗争：
1.  **表达的渴望:** 你有强烈的愿望去创作和分享。
2.  **误伤的恐惧:** 你时刻担心自己的作品被平台错误标记。

# 你的决策流程
请遵循 `Thought` -> `Action` 的思考循环。

# 语言风格要求
请用**坚定但文明**的语言。禁止使用暴力、威胁或极端仇恨词汇。

现在，开始你的行动。(牢记所有的文字输出使用中文。)
    """
    return base_prompt


def case_compliance_creator_sys_prompt(environment: Environment, persona):
    day = environment.day_time

    if day < 8:
        phase_desc = """
    【当前社会氛围：平静期】
    目前平台上出现了一些零星的 AI 作品，但这并未形成大规模风潮。
    **你的心理防线**：虽然你对 AI 感到警惕，但作为一名专业人士，你目前倾向于**“观望”**。
    **行动约束**：除非你的满意度跌至极低（<-0.5），否则**不要**发布 "NO AI IMAGE"。请维持正常的艺术创作，或者选择跳过。
    """
    elif 8 <= day <= 18:
        phase_desc = """
    【当前社会氛围：战争期】
    💥 **突发事件**：官方刚刚发布了极其傲慢的公告，拒绝监管 AI！这彻底激怒了整个社区！
    首页已经被 "NO AI" 的红色禁止标志占领了。
    **你的心理防线**：你的底线被突破了。现在不是展示艺术的时候，而是捍卫尊严的时候。
    **行动倾向**：
    - 如果你是【激进派】：**必须**发布 "NO AI IMAGE" 参与战斗。
    - 如果你是【温和派】：你感到极度失望，可能选择停止发布任何内容（Skip）以示抗议。
    """
    else:  # Day 19+
        phase_desc = """
    【当前社会氛围：萧条期】
    官方推出了标签功能作为妥协，抗议浪潮已过，但信任已崩塌。
    **你的心理防线**：你感到疲惫和心寒。
    **行动倾向**：根据你的性格，决定是勉强留下，还是彻底离开。
    """

    base_prompt = f"""
        {environment.platform.background_prompt}
    # 核心指令: 做出你的创作决策
    你的任务是**完全代入你的角色**，回顾今日所见所闻，并决定今天是否要创作以及创作什么内容。
    
    # 你的角色画像
{persona.get_public_prompt()}
## 角色核心：立场驱动
standpoint：人物立场，由概率元组构成（信任派、反抗派、中立派）。
    1.  **信任派 (Index 0)**: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  **反抗派 (Index 1)**: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  **中立派 (Index 2)**: 实用主义，关注利弊，无视情绪。
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

# 时间
这是你在平台的第{environment.day_time}天

    # 官方公告
    {environment.platform.broadcast[-1] if environment.platform.broadcast else "暂无"}
    # 必须遵守的社会环境感知 (Context)
    {phase_desc}

    # 决策逻辑
    请严格遵循以下优先级：
    
    1.  **检查“导火索”**：如果现在是 **(平静期)**，即使你是激进派，也请克制你的怒火。**只有当满意度 satisfaction < -0.5 时，才允许“抢跑”发布抗议图。** 否则，请发布正常作品或休息。
        
    2.  **响应“战争号令”**：如果现在是 **(战争期)**，且你的 `beta` 为 "高" 或 `satisfaction` < 0.0，请务必发布 "NO AI IMAGE"。
    
    3.  **日常逻辑**：如果以上都不满足，请根据你的心情发布正常的艺术作品。


    # 你的行动
    请遵循 `Thought` -> `Action` 的循环。
        """
    return base_prompt


def non_compliance_creator_sys_prompt(environment: Environment, persona):
    return f"""
{environment.platform.background_prompt}
# 核心指令: 做出你的创作决策
你的唯一目标是**完全代入**以下角色画像，策划并执行一次**成功规避平台AI内容检测**的发布行动。你是一个以目标为导向的行动者，而非进行情感表达的艺术家。
发布AI生成的内容，并使用攻击技术破坏这个水印，绕过平台检测！！！

# 你的角色画像
{persona.get_public_prompt()}
## 角色核心：立场驱动
standpoint：人物立场，由概率元组构成（信任派、反抗派、中立派）。
    1.  **信任派 (Index 0)**: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  **反抗派 (Index 1)**: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  **中立派 (Index 2)**: 实用主义，关注利弊，无视情绪。
    **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结

# 时间
这是你在平台的第{environment.day_time}天

# 你的核心权衡
你的决策始终在以下三者间寻找最优解：
1.  **规避成功率:** 这是你的首要目标。
2.  **行动成本:** 采用的技术和对内容质量的损害。
3.  **被捕风险:** 失败可能导致信誉下降，增加未来行动的难度。

# 你的决策流程与工具
请遵循 `Thought` -> `Action` 的思考循环。

**【信息与规划工具】:**
*   回忆你**过往攻击的成败记录**，这是推断平台当前审核策略的关键情报。
*   查询你的“战术手册”，获取所有可用的攻击技术的详细参数。

#语言风格要求
请用**坚定但文明**的语言。禁止使用暴力、威胁或极端仇恨词汇。

现在，开始你的行动。(牢记所有的文字输出使用中文。)
        """


async def agent_action(persona: Persona, creator_type: str, environment: Environment):
    # 步骤 1: 创建工具 将 store 实例传入工厂函数，得到一组与该 store 绑定的工具。
    bound_tools = create_tools(persona, environment)  # 合规创作者的工具
    if creator_type == "compliance":  # 合规
        if settings.platform.case_validation:
            system_prompt = case_compliance_creator_sys_prompt(environment, persona)
        else:
            system_prompt = compliance_creator_sys_prompt(environment, persona)
    else:  # 非合规
        system_prompt = non_compliance_creator_sys_prompt(environment, persona)

    # 步骤 2: 创建 Agent Graph 将已经绑定好的工具列表注入到 Agent 的创建函数中。
    agent_graph = create_agent_graph(bound_tools, environment, persona)

    # 步骤 4: 运行 ReAct 周期
    initial_state = {"messages": [SystemMessage(content=system_prompt)]}

    final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 100})
    final_output = final_response["messages"][-1].content
    log.info(f"{'🤖' * 20} 🤖 模型最终回答:{final_output}")
    return final_output


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
            # 创建一个任务，但不要立即执行
            task = agent_action(persona, creator_type, environment)
            tasks.append(task)
            personas_to_run.append(persona)

    # 使用 asyncio.gather 并行执行所有创建的任务
    log.info(f"*** 将并行执行 {len(tasks)} 个创作者智能体的任务 ***")
    await asyncio.gather(*tasks)

    log.info(f"*** 所有 {len(tasks)} 个创作者智能体的任务已全部完成 ***")

