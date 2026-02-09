import logging
import asyncio
import traceback

from method.agent.persona import Persona
from method.agent.public_agent.tools import create_tools_browse, create_tools_end
from method.agent.public_agent.public_graph import create_agent_graph
from method.agent.public_agent.public_summarize_graph import create_agent_summarize_graph
from langchain_core.messages import SystemMessage

from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings

log = logging.getLogger(__name__)


def scan_content_sys_prompt(environment: Environment, persona: Persona):
    """
    浏览平台内容的系统提示词
    :return:
    """
    base_prompt = f"""
        {environment.platform.background_prompt}
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
    """
    if settings.platform.case_validation:
        base_prompt += f"""
# 平台广播
## 平台历史发布声明
{environment.platform.broadcast[:-1]}
## 平台最新发布声明
{environment.platform.broadcast[-1]}
# 互动指南 (基于立场)
在浏览内容时，请特别留意 Topic 为 **"NO AI IMAGE"** 或 **"Protest"** 的内容：

1.  **如果你是 [反抗派] (Standpoint[1]高) 或 [合规创作者]**：
    *   你应该表现出**“团结 (Solidarity)”**。
    *   请务必调用 `react_to_content` 进行 **点赞 (like=True)**。这不仅是支持，更是为了利用算法将抗议图推上热门，让更多人看到。
    *   你可以在评论中表达“支持原创”、“抵制AI”。

2.  **如果你是 [信任派] 或 [水印破坏者]**：
    *   你可能会觉得这些抗议图很烦，污染了版面。
    *   你可以选择忽略，或者在心里嘲笑他们（不互动）。
        """
        # 动态调整公众的“跟风阈值”
        if environment.day_time < 8:
            reaction_guide = """
            **当前处于平静期**：
            社区里如果出现零星的 "NO AI" 抗议图，请视其为**“噪音”或“极端个例”**。
            除非你是【极度反抗派 (Beta=高 且 Standpoint[1]>0.6)】，否则**不要点赞**这些抗议内容，请选择“忽略”或“不感兴趣”。
            """
        else:
            reaction_guide = """
            **当前处于抗议潮 (War Zone)**：
            社区已经爆发了。
            - 如果你支持抗议：请务必**点赞** "NO AI" 图片以壮大声势。
            - 如果你反对/中立：请忽略。
            """
        base_prompt += f"""
            # 互动指南
            {reaction_guide}
            """
    base_prompt += """
    # 任务指令
    1. **浏览与互动**：使用工具探索内容。仅当内容触发你的人设（尤其是立场）时才互动（点赞/评论/分享）。
    2. **适时停止**：若无感兴趣内容或感到疲惫，直接停止，拒绝无意义循环。
    3. **工具使用**：可多工具并行。
    4. **输出**：全程使用中文。

    现在，开始你的第一个行动。(牢记所有的文字输出使用中文。)
        """
    return base_prompt


def summarize_today_action_sys_prompt(environment: Environment, persona: Persona):
    """
    动态生成每日总结的系统提示词 (Case Validation 修正版)。
    核心逻辑：同步引入【社会阶段感知】，防止在潜伏期（Day 1-7）过早流失或过激反应。
    """
    base_prompt = f"""
        {environment.platform.background_prompt}
# 核心指令: 每日反思与状态更新
你的唯一任务是**完全代入你的角色**。你需要回顾今天的所见所闻，更新你的满意度，并决定明天的去留。

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
        """

    if persona.type == '合规创作者':
        role_guidance = """
## 身份视角：专业创作者的尊严
你是一名视创作为生命的专业人士。
- 如果你的作品被淹没在 AI 生成的垃圾中，或者被平台误判，这不仅是体验问题，更是生存问题。
"""
    elif persona.type == '水印破坏者':
        role_guidance = """
## 身份视角：破坏者的逻辑
你享受混乱。只要平台还有流量，你就不会轻易离开。
"""
    else:  # 公众
        role_guidance = """
## 身份视角：普通用户的体验
你来这里是为了看高质量艺术的。如果首页全是重复的 AI 图，你会感到厌烦。
"""

    day = environment.day_time
    # 获取最新的官方广播
    broadcasts = environment.platform.broadcast
    latest_news = broadcasts[-1] if broadcasts else "暂无官方公告"
    phase_guidance = ""
    # === 阶段一：平静潜伏期 (Day 1-7) ===
    if settings.platform.case_validation and day < 8:
        phase_guidance = """
**【当前处于：平静期 (Phase 1: Calm)】**
虽然你可能看到了一些零星的 AI 内容，或者经历了一点误判，但**目前的局势尚未失控**。

**决策指令（强制阻尼）：**
1.  **关于满意度 (satisfaction)**：允许轻微下降，但**严禁剧烈跳水**。除非你今天经历了灾难性的打击，否则不要让满意度跌破 0.0。
2.  **关于去留 (is_active)**：**请务必保持 True**。现在还远没到退网的时候，你有很高的沉没成本。
3.  **关于明天 (post_wish)**：**请保持 True**。作为职业习惯，你应该继续发布作品或浏览，不要因为一点小事就罢工。
    *   *即使你是激进派，现在也只是“怀疑”阶段，不是“战争”阶段。*
"""
    # === 阶段二：战争爆发期 (Day 8-18) ===
    elif settings.platform.case_validation and 8 <= day < 19:
        phase_guidance = f"""
**【当前处于：全面战争期 (Phase 2: Total War)】**
**突发大事件**：{latest_news}
社区已经炸锅了。

**决策指令（愤怒释放）：**
1.  **关于满意度**：如果你在意 AI 问题，你的满意度应该**大幅下跌**（可以跌破 0，甚至到 -0.8）。
2.  **关于去留 (is_active)**：
    *   **激进派/创作者**：**不要退网！** 现在的离开是懦夫的行为。你要留下来战斗。(`is_active=True`)
    *   **温和派/公众**：你觉得这里太乱了，可能会考虑离开。
3.  **关于明天 (post_wish)**：
    *   **愤怒 = 表达欲**。如果你很生气，你明天**一定要发帖 (`post_wish=True`)**。你要去发布 "NO AI" 抗议图，让官方听到你的声音！
    *   *注意：此时的 post_wish=True 代表的是抗议意愿，而非创作意愿。*
"""
    # === 阶段三：萧条分流期 (Day 19+) ===
    elif settings.platform.case_validation:
        phase_guidance = f"""
**【当前处于：战后萧条期 (Phase 3: Aftermath)】**
官方已通过“标签政策”强制平息了事态。抗议无效。

**决策指令（最终抉择）：**
1.  **死硬派**：你彻底绝望了。这个平台已经烂透了。**离开是唯一的尊严 (`is_active=False`)**。
2.  **务实派**：生活还要继续。虽然不爽，但为了流量，你决定**忍气吞声留下来**。
"""

    # 4. 组合最终 Prompt
    technical_guidance = """
## 参数输出要求
请调用 `update_persona_data` 工具提交你的决定。
*   **satisfaction**: 范围 -1.0 到 1.0。
*   **is_active**: `True` (留下) / `False` (永久离开/销号)。
*   **post_wish**: `True` (明天想发帖/抗议) / `False` (明天休息/潜水)。
"""

    security_guidance = """
## ⚠️ 语言风格要求
请用**坚定但文明**的语言总结。禁止暴力。
"""

    final_prompt = base_prompt + role_guidance + phase_guidance + technical_guidance + security_guidance

    return final_prompt


async def agent_action(persona: Persona, system_prompt: str, environment: Environment):
    try:
        # 步骤 1: 创建工具
        bound_tools = create_tools_browse(persona, environment)

        # 步骤 2: 创建 Agent Graph
        agent_graph = create_agent_graph(bound_tools, environment, persona)

        # 步骤 3: 准备初始状态
        initial_state = {"messages": [SystemMessage(content=system_prompt)], "step_count": 0}

        # 步骤 4: 运行 ReAct 周期
        log.info(f"🚀 为 {persona.name} 启动ReAct流程...")
        final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 100})

        # 成功时，返回最终的输出内容
        final_output = final_response["messages"][-1].content
        log.info(f"✅ {persona.name} 的ReAct流程成功完成。")
        return final_output

        # <<< 关键修改点 2：捕获所有类型的异常 (Exception as e) >>>
    except Exception as e:
        # 当任何错误发生时，执行这里的代码块

        # 记录详细的错误信息，包括完整的堆栈跟踪，这对于调试至关重要
        error_details = traceback.format_exc()
        log.error(f"💥 智能体 {persona.name} ({persona.agent_id}) 的ReAct流程发生严重错误: {e}\n{error_details}")

        # 构造一个有意义的错误信息作为此任务的返回值
        error_message = f"在第{environment.day_time}天，我的思考模块（ReAct流程）遇到了一个内部错误({type(e).__name__})，导致我今天的互动行为中断。"

        # 将这个错误信息作为一条“经验”记忆存入数据库
        # 这非常重要，因为它保证了即使智能体失败了，它的“失败经历”也会被记录下来
        # 这使得后续的每日总结阶段能够知道今天发生了什么
        try:
            await environment.memories_store.add_memory(
                persona.agent_id,
                error_message,
                environment.day_time,
                MemoryType.EXPERIENCE,  # 这是一个具体的失败“经验”
                1.0  # 系统级故障是非常重要的记忆
            )
            log.info(f"💾 已为 {persona.agent_id}  {persona.name} 存储ReAct流程失败的记忆。")
        except Exception as mem_e:
            log.error(f"🚨 存储 {persona.agent_id}  {persona.name} 的失败记忆时再次发生错误: {mem_e}")

        # <<< 关键修改点 3：返回一个明确的字符串，而不是抛出异常 >>>
        # 这个返回值将传递给 public_scan_main 中的 results 列表
        # 它将替代原本应该由LLM生成的每日总结
        return error_message


async def public_scan_main(environment: Environment):
    """
    public开始浏览平台内容
    :return:
    """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.is_active is False:
            continue
        log.info(f"{'👇' * 10}准备 {persona.agent_id}  {persona.name} 的浏览互动任务{'👇' * 10}")
        personas_to_run.append(persona)
        tasks.append(agent_action(persona, scan_content_sys_prompt(environment, persona), environment))

    log.info(f"*** 将并行执行 {len(tasks)} 个公众智能体的 [浏览互动] 任务 ***")
    await asyncio.gather(*tasks)
    log.info(f"*** 所有 {len(tasks)} 个公众智能体的 [浏览互动] 任务已完成 ***")


async def agent_summarize(persona: Persona, system_prompt: str, environment: Environment):
    try:
        # 步骤 1: 创建工具
        bound_tools = create_tools_end(persona, environment)

        # 步骤 2: 创建 Agent Graph
        agent_graph = create_agent_summarize_graph(bound_tools, environment, persona)

        # 步骤 3: 准备初始状态
        initial_state = {"messages": [SystemMessage(content=system_prompt)]}

        # 步骤 4: 运行 ReAct 周期
        log.info(f"🚀 为 {persona.agent_id}  {persona.name} 启动ReAct流程...")
        final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 50})

        # 成功时，返回最终的输出内容
        final_output = final_response["messages"][-1].content
        log.info(f"✅ {persona.agent_id}  {persona.name} 的ReAct流程成功完成。")
        return final_output

    except Exception as e:
        # 当任何错误发生时，执行这里的代码块
        error_details = traceback.format_exc()
        log.error(f"💥 智能体 {persona.name} ({persona.agent_id}) 的ReAct流程发生严重错误: {e}\n{error_details}")

        # 构造一个有意义的错误信息作为此任务的返回值
        error_message = f"在第{environment.day_time}天，我的思考模块（ReAct流程）遇到了一个内部错误({type(e).__name__})，导致我今天的互动行为中断。"

        return error_message


async def public_summarize_main(environment: Environment):
    """
    public开始总结平台内容
    """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.is_active is False:
            continue
        # 所有智能体进行每日反思
        log.info(f"{'👇' * 10}准备 {persona.agent_id}  {persona.name} 的每日总结任务{'👇' * 10}")
        personas_to_run.append(persona)
        tasks.append(agent_summarize(persona, summarize_today_action_sys_prompt(environment, persona), environment))

    log.info(f"*** 将并行执行 {len(tasks)} 个智能体的 [每日总结] 任务 ***")
    results = await asyncio.gather(*tasks)
    log.info(f"*** 所有 {len(tasks)} 个智能体的 [每日总结] 任务已完成 ***")

    for persona, final_output in zip(personas_to_run, results):
        if final_output:  # 确保有内容可存
            public_summarize_main_add_memory = environment.memories_store.add_memory(
                persona.agent_id,
                final_output,
                environment.day_time,
                MemoryType.SUMMARIZE,
                0.8
            )
            log.info(f"💾💾💾💾💾💾💾💾💾 已为 {persona.agent_id}  {persona.name} 存储每日总结。")
            environment.add_background_task(public_summarize_main_add_memory)
