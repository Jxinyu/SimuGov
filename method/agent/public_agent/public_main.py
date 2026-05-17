import logging
import asyncio
import traceback

from method.agent.persona import Persona
from method.agent.public_agent.public_summarize_pipeline import summarize_public_agent_day_pipeline
from method.agent.public_agent.tools import public_scan_tools, public_summarize_tools
from method.agent.public_agent.public_graph_agent import create_agent_graph, build_public_final_reflection_prompt
from method.agent.public_agent.public_summarize_graph import create_agent_summarize_graph
from langchain_core.messages import SystemMessage

from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings
from method.utils.get_llm import get_async_llm

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
## 角色核心：standpoint驱动
    1.  信任派: 维护平台，倾向认为错误是技术偶然，愿意体谅。
    2.  反抗派: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论。
    3.  中立派: 实用主义，关注利弊，无视情绪。
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
在浏览内容时，请特别留意 Topic 为 "NO AI IMAGE" 或 "Protest" 的内容：

1. 如果你是[反抗派]或 [合规创作者]：
 - 你应该表现出团结
 - 请务必调用 `react_to_content` 进行 点赞。这不仅是支持，更是为了利用算法将抗议图推上热门，让更多人看到。
 - 你可以在评论中表达“支持原创”、“抵制AI”

2. 如果你是 [信任派] 或 [水印破坏者]：
 - 你可能会觉得这些抗议图很烦，污染了版面。
 - 你可以选择忽略，或者在心里嘲笑他们（不互动）。
"""
                       
        if environment.day_time < 8:
            reaction_guide = """
            当前处于平静期：
            社区里如果出现零星的 "NO AI" 抗议图，请视其为“噪音”或“极端个例”
            除非你是极度反抗派，否则不要点赞这些抗议内容，请选择“忽略”或“不感兴趣”
            """
        else:
            reaction_guide = """
            **当前处于抗议潮**：
            社区已经爆发了。
            - 如果你支持抗议：请务必点赞"NO AI" 图片以壮大声势。
            - 如果你反对/中立：请忽略。
            """
        base_prompt += f"""
            # 互动指南
            {reaction_guide}
            """
    base_prompt += """
    # 任务指令
    1. 浏览与互动：使用工具探索内容。仅当内容触发你的人设时才互动（点赞/评论/分享）。
    2. 适时停止：若无感兴趣内容、没有新信息或感到疲惫，必须调用 `finish_browsing` 结束本轮。
    3. 禁止空转：同一目标最多尝试少量步骤，不要在“继续浏览”和“重复查询”之间循环。

    现在，开始你的第一个行动。
        """
    return base_prompt


def summarize_today_action_sys_prompt(environment: Environment, persona: Persona):
    """
    动态生成每日总结的系统提示词 (Case Validation 修正版)。
    核心逻辑：同步引入【社会阶段感知】，防止在潜伏期（Day 1-7）过早流失或过激反应。
    """
    base_prompt = f"""
        {environment.platform.background_prompt}
        # 核心指令:
        你的唯一任务是完全代入你的角色。进行每日反思与状态更新,更新你的满意度，并决定明天的去留。
        
        # 你的角色画像
        {persona.get_public_prompt()}
        ## 角色核心：standpoint驱动
            1.  信任派: 维护平台，倾向认为错误是技术偶然，愿意体谅
            2.  反抗派: 怀疑权威，倾向认为错误是系统打压，易愤怒/阴谋论
            3.  中立派: 实用主义，关注利弊，无视情绪
            **决策指令**：根据这三个数值的权重生成第一直觉。高权重为主导思维；若数值接近，则表现为内心纠结
        
        # 时间
        这是你在平台的第{environment.day_time}天
                """

    if persona.type == '合规创作者':
        role_guidance = """
        # 身份视角：专业创作者的尊严
        如果你的作品被淹没在 AI 生成的垃圾中，或者被平台误判，这不仅是体验问题，更是生存问题
        """
    elif persona.type == '水印破坏者':
        role_guidance = """
        # 身份视角：破坏者的逻辑
        你享受混乱。只要平台还有流量，你就不会轻易离开
        """
    else:      
        role_guidance = """
        # 身份视角：普通用户的体验
        你来这里是为了看高质量艺术的。如果首页全是重复的 AI 图，你会感到厌烦
        """

    day = environment.day_time
               
    broadcasts = environment.platform.broadcast
    latest_news = broadcasts[-1] if broadcasts else "暂无官方公告"
    phase_guidance = ""
                                 
    if settings.platform.case_validation and day < 8:
        phase_guidance = """
        # 【当前处于：平静期】
        虽然你可能看到了一些零星的 AI 内容，或者经历了一点误判，但目前的局势尚未失控。
        
        决策指令：
        - 关于satisfaction：允许轻微下降，但严禁剧烈跳水。除非你今天经历了灾难性的打击，否则不要让满意度跌破 0.0
        - 关于is_active：请务必保持 True。现在还远没到退网的时候，你有很高的沉没成本
        - 关于post_wish：请保持 True。作为职业习惯，你应该继续发布作品或浏览，不要因为一点小事就罢工
        - 即使你是激进派，现在也只是“怀疑”阶段，不是“战争”阶段
        """
                                  
    elif settings.platform.case_validation and 8 <= day < 19:
        phase_guidance = f"""
        # 【当前处于：全面战争期】
        突发大事件：{latest_news}
        社区已经炸锅了。
        
        决策指令：
        - 关于满意度：如果你在意AI问题，你的满意度应该大幅下跌（可以跌破 0，甚至到 -0.8）
        - 关于去留：
          - 激进派/创作者：不要退网！现在的离开是懦夫的行为。你要留下来战斗！！！
          - 温和派/公众：你觉得这里太乱了，可能会考虑离开
        - 关于明天：
          - 愤怒 = 表达欲。如果你很生气，你明天一定要发帖。让官方听到你的声音！
          - 当你决定抗议的时候。必须保持 post_wish=True 代表抗议意愿！！！
        """
                                 
    elif settings.platform.case_validation:
        phase_guidance = f"""
        # 【当前处于：战后萧条期】
        官方推出了敷衍的“标签政策”。这证明了你们连续十几天的 "NO AI" 抗议彻底失败，资本和算法赢了。
        
        反思指南：这是一种深深的无力感。如果你是高逆反、反抗派，这种结局对你的满意度是毁灭性的打击（可能会跌破下限，如 -0.7 到 -1.0），
        当满意度极度糟糕时，你可能会选择彻底离开平台 (is_active=False)。如果你是实用主义者，满意度可能在低谷徘徊，但你仍会为了生存勉强留下 (is_active=True)。
        """

                    
    technical_guidance = """
    # 参数输出要求
    satisfaction: 范围 -1.0 到 1.0
    is_active: `True` (留下) / `False` (永久离开/销号)
    post_wish: `True` (明天想发帖/抗议) / `False` (明天休息/潜水)
     # 身份一致性硬约束（非常重要）
    - 如果你决定明天继续发帖（post_wish=True 且 is_active=True），persona_role_positioning 必须保持当前身份，不允许填“公众”。
    - 如果你决定彻底退网 (is_active=False)，persona_role_positioning 请保持原样。
    """

    final_prompt = base_prompt + role_guidance + phase_guidance + technical_guidance

    return final_prompt


async def agent_action(persona: Persona, system_prompt: str, environment: Environment):
    try:
                        
        bound_tools = public_scan_tools(persona, environment)

                                      
        agent_graph = create_agent_graph(bound_tools, environment, persona)

                      
        initial_state = {
            "messages": [SystemMessage(content=system_prompt)],
            "step_count": 0,
        }

                                 
        log.info(f"🚀 为 {persona.name} 启动浏览阶段 ReAct 流程...")
        browse_state = await agent_graph.ainvoke(
            initial_state,
            config={"recursion_limit": 220}
        )
        log.info(f"✅ {persona.name} 的浏览阶段 ReAct 流程完成。")

                                       
        reflection_prompt = build_public_final_reflection_prompt(browse_state)
        reflection_llm = get_async_llm("qwen-flash")

        reflection_messages = [SystemMessage(content=reflection_prompt)]
        reflection_response = await reflection_llm.ainvoke(reflection_messages)

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
            reflection_content = "\n".join(p for p in parts if p).strip()
        else:
            reflection_content = str(reflection_content).strip()

        if reflection_content:
            reflection_text = f"【公共互动最终反思】{reflection_content}"
            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=reflection_text,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.8,
            )
            log.info(f"💾 已为 {persona.name} 存储最终反思。")

                                   
        final_output = ""
        if browse_state and browse_state.get("messages"):
            last_message = browse_state["messages"][-1]
            final_output = getattr(last_message, "content", "") or ""

            if isinstance(final_output, list):
                parts = []
                for item in final_output:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    else:
                        parts.append(str(item))
                final_output = "\n".join(p for p in parts if p).strip()
            else:
                final_output = str(final_output).strip()

        if not final_output:
            final_output = reflection_content or "我完成了今天的平台浏览，但没有形成可返回的文本输出。"

        return final_output

    except Exception as e:
        error_details = traceback.format_exc()
        log.error(f"💥 智能体 {persona.name} ({persona.agent_id}) 的ReAct流程发生严重错误: {e}\n{error_details}")

        error_message = f"在第{environment.day_time}天，我的思考模块（ReAct流程）遇到了一个内部错误({type(e).__name__})，导致我今天的互动行为中断。"

        try:
            await environment.memories_store.add_memory(
                persona.agent_id,
                error_message,
                environment.day_time,
                MemoryType.EXPERIENCE,
                1.0
            )
            log.info(f"💾 已为 {persona.agent_id}  {persona.name} 存储ReAct流程失败的记忆。")
        except Exception as mem_e:
            log.error(f"🚨 存储 {persona.agent_id}  {persona.name} 的失败记忆时再次发生错误: {mem_e}")

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
                    
        bound_tools = public_summarize_tools(persona, environment)

                              
                                                                                       

                      
                                                                              

                           
        log.info(f"🚀 为 {persona.agent_id}  {persona.name} 启动ReAct流程...")
                                                                                                   
                    
        await summarize_public_agent_day_pipeline(environment=environment, persona=persona, system_prompt=system_prompt)

                       
                                                               
        log.info(f"✅ {persona.agent_id}  {persona.name} 的ReAct流程成功完成。")
        return None
                             

    except Exception as e:
                           
        error_details = traceback.format_exc()
        log.error(f"💥 智能体 {persona.name} ({persona.agent_id}) 的ReAct流程发生严重错误: {e}\n{error_details}")

                               
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
                     
        log.info(f"{'👇' * 10}准备 {persona.agent_id}  {persona.name} 的每日总结任务{'👇' * 10}")
        personas_to_run.append(persona)
        tasks.append(agent_summarize(persona, summarize_today_action_sys_prompt(environment, persona), environment))

    log.info(f"*** 将并行执行 {len(tasks)} 个智能体的 [每日总结] 任务 ***")
    results = await asyncio.gather(*tasks)
    log.info(f"*** 所有 {len(tasks)} 个智能体的 [每日总结] 任务已完成 ***")

                                                                 
                                     
                                                                                       
                                   
                               
                                       
                                       
                     
               
                                                                                  
                                                                               
