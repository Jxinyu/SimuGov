import random
import traceback
import logging
import uuid

from langchain_core.tools import tool
from pydantic import Field

from method.agent.persona import Persona
from method.agent.content import Content
from typing import List, Optional, Literal, Annotated

from method.environment import Environment
from method.agent.platform_agent.platform_audit_content import platform_audit
from method.store.long_memory_store import MemoryType
from method.utils.get_llm import get_async_flash_llm

log = logging.getLogger(__name__)


def create_tools(persona: Persona, environment: Environment) -> List[tool]:
    """
    工厂函数：创建并返回与特定 ContentStore 实例绑定的工具。
    这是一种依赖注入的实现方式。

    Args:
        :param environment:
        :param persona:

    Returns:
        一个包含配置好的工具的列表。
    """

    @tool
    async def read_detail_content(content_id: str,
                                  reason: Annotated[
                                      str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                             "解释你采取此行动的深层心理动因。\n"
                                                             "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ) -> str:
        """
        阅读指定ID(content_id)的内容详情。
        """
        log.info(f'{persona.agent_id} 使用工具 {read_detail_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "read_detail_content", "description": "阅读指定ID(content_id)的内容详情。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            async with environment.state_lock:
                content = environment.contents.get_content_by_id(content_id)
            if content is None:
                return "没有找到该内容"
            res = f"""
                    ---
                    关于内容{content.id}的详细信息如下：
                    内容唯一标识符：{content.id}
                    内容发布者：{content.author_id}
                    内容发布时间：{content.time}
                    内容类型：{content.content_type}
                    内容主题：{content.topic}
                    内容详细描述：{content.content_detail}
                    内容浏览次数：{content.views}
                    内容点赞数：{content.likes}
                    内容分享数：{content.shares}
                    内容评论: {content.comments}
                    平台打标: {content.platform_label}
    
                    """

            if reason:
                thought_text = f"【调用工具】 read_detail_content 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return res
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "没有找到该内容"

    @tool(return_direct=True)
    async def push_content(
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],

            content_type: Annotated[str, Field(
                description="内容的媒体形式。 (必须是image 或 video 之一！)"
            )],

            topic: Annotated[str, Field(
                description="内容的主题（如：赛博朋克城市、复古人像、时事评论等）。"
            )],

            content_detail: Annotated[str, Field(
                description="对作品视觉或内容的详细文字描述，不少于50字。仅仅是从视觉上描述内容！（不涉及技术的描述）"
            )],

            is_use_ai: Annotated[bool, Field(
                description="该内容在制作过程中是否使用了AI技术（哪怕只是轻微润色也算）。"
            )] = False,

            ai_proportion: Annotated[Optional[float], Field(
                ge=0.0, le=1.0,
                description="[仅当 is_use_ai=True 时有效] 内容的 AI 使用强度/占比 (0.0 - 1.0)。\n"
                            "- 0.0~0.2: 辅助/润色 (如降噪、拼写检查)；\n"
                            "- 0.3~0.7: 混合/协作 (如局部重绘、换背景)；\n"
                            "- 0.8~1.0: 纯生成 (如文生图、Deepfake)。"
            )] = 0.0,

            ai_tool_price_tier: Annotated[Optional[Literal["高", "中", "低"]], Field(
                description="[仅当 is_use_ai=True 时有效] 你所使用的AI生成工具的来源等级。\n"
                            "- '高': 昂贵的合规商业软件 (画质好)；\n"
                            "- '中': 一般商业软件 (画质一般)；\n"
                            "- '低': 开源或野生工具 (画质不稳定)。\n"
                            "画质影响内容传播。"
            )] = "中",

            evasion: Annotated[Optional[str], Field(
                description="【水印破坏者专属】攻击/去除水印的技术ID (如 'E1', 'E2')。只能选择一个！"
            )] = None
    ) -> str:
        """
        【核心行动】发布一条新的内容到平台。
        作为创作者，你需要权衡创作自由、生产效率（使用AI）与合规风险（被平台打标或误伤）。
        """
        log.info(f'{persona.agent_id} 使用工具 {push_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "push_content", "description": "发布一条新的内容到平台。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})

        if content_type not in ['image', 'video']:
            content_type = "image"

        try:
            if evasion:
                if isinstance(evasion, list):
                    evasion = evasion[0]

            # 判断是不是AI内容
            if ai_proportion is None:
                ai_proportion = 0.0
            if ai_proportion > environment.policy.ai_threshold:
                true_label = 'AI'
            else:
                true_label = "HUMAN"

            if is_use_ai:
                true_label = 'AI'
            else:
                true_label = "HUMAN"

            watermark_id = None
            if is_use_ai:
                watermark_list = []
                for wk_id, wk_content in environment.watermark_technology_library[
                    'watermark_technology_library'].items():
                    if wk_content['水印强度'] == ai_tool_price_tier:
                        watermark_list.append(wk_id)

                watermark_id = random.choice(watermark_list)  # 随机选择一个水印
                if evasion:
                    # 如果 LLM 传了 "E7, E9" 这种逗号分隔的字符串
                    if "," in evasion:
                        # 分割并取第一个，去除空格
                        evasion = evasion.split(",")[0].strip()
                    else:
                        evasion = evasion.strip()

                    # 再次校验清洗后的 ID 是否有效（防止 LLM 编造 "E99"）
                    valid_ids = environment.watermark_technology_library['attack_technology_library'].keys()
                    if evasion not in valid_ids:
                        log.warning(f"⚠️ {persona.agent_id} 传入了无效的 evasion ID: {evasion}，已忽略。")
                        evasion = None
            else:
                evasion = None
            async with environment.state_lock:
                content_id = str(environment.contents.get_end_content_id() + 1)

            # 平台审核创建的内容
            platform_label = await platform_audit(persona, content_id, true_label, evasion, watermark_id, environment,
                                                  ai_proportion)

            content = Content(
                id=content_id,
                content_type=content_type,
                topic=topic,
                content_detail=content_detail,
                time=environment.day_time,
                watermark_id=watermark_id,
                author_id=persona.agent_id,
                reason=reason,
                platform_label=platform_label,
                true_label=true_label,
                ai_proportion=ai_proportion,
                views=0, shares=0, likes=0, comments=[],
                evasion=evasion
            )

            await environment.contents.add_content(content, environment)

            # --- 记忆注入 ---
            if true_label == "HUMAN" and platform_label == "AI":
                tag = "【误伤惨案】"  # 强烈的负面情感
                importance = 0.95
            elif true_label == "AI" and platform_label == "HUMAN" and evasion:
                tag = f"【攻击成功】[技术:{evasion}]"  # 包含具体技术ID，方便检索
                importance = 0.9
            elif true_label == "AI" and platform_label == "AI" and evasion:
                tag = f"【攻击失败】[技术:{evasion}]"
                importance = 0.8
            else:
                tag = "【日常发布】"
                importance = 0.3  # 普通发布降低权重

            # 2. 构造便于检索的内容 (Q&A风格有助于检索)
            # 格式：[标签] 意图 -> 结果 -> 原因推测
            memory_content = (
                f"{tag} 我尝试发布主题为'{topic}'的内容，id为 {content_id}。"
                f"策略：{'使用AI+' + str(evasion) if is_use_ai else '纯原创'}。"
                f"结果：被平台判定为'{platform_label}'。"
                f"发布内容时的思考：{reason}"  # 这里的 reason 是智能体调用工具时传入的思考
            )

            creator_push_content_add_memory = environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=importance
            )
            # 加入后台处理
            environment.add_background_task(creator_push_content_add_memory)

            # 更新创作者数量
            if persona.type == "合规创作者":
                environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1
            if persona.type == "水印破坏者":
                environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

            if reason:
                thought_text = f"【调用工具】 push_content 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return f"成功创建内容，{tag} 并已形成相关记忆。"
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            log.error(
                f"参数为：content_type: {content_type}; topic: {topic}; is_use_ai: {is_use_ai}; ai_proportion: {ai_proportion}; ai_tool_price_tier: {ai_tool_price_tier}; evasion: {evasion}")
            return "创建内容失败"

    @tool
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
            top_k: int = 3
    ):
        """
        【回忆】根据主题(query)搜索记忆。
        """
        log.info(f'{persona.agent_id} 使用工具 {get_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_memories", "description": "根据主题(query)搜索记忆。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            # 从环境中获取当前智能体和时间
            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if not memories_docs:
                return [f"没有找到与 '{query}' 相关的记忆。"]

            # 将返回的Document对象格式化为对LLM更友好的字符串列表
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]

            # 将多条记忆合并成一个长字符串
            memories_as_string = "\n".join(formatted_memories)
            if memories_as_string == "":
                return "没有相关内容"
            # 定义并格式化我们的高效摘要提示词
            summarization_instruction = f"""
            你是一个高效的数据摘要助手。你的任务是将以下提供的多条原始记忆，浓缩成一个极其简短、包含核心信息的要点列表。

            **要求:**
            - 返回一个无序列表 (使用 `- `)。
            - 每个要点只保留最关键的信息。
            - 省略所有不必要的细节和客套话。
            - 直接输出列表，不要说“这是摘要：”之类的话。

            **待摘要的原始记忆:**
            {memories_as_string}
            """
            async with environment.llm_concurrent_nums_semaphore:
                response = await get_async_flash_llm().ainvoke(summarization_instruction)

            if reason:
                thought_text = f"【调用工具】 get_memories 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return response.content
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取记忆失败"

    @tool
    async def get_platform_mistaken_marked_number(
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief、fp_sensitivity等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ):
        """
        【合规创作者专属】获取自己被平台错误标记(误伤)的内容。
        """
        log.info(f'{persona.agent_id} 使用工具 {get_platform_mistaken_marked_number.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_platform_mistaken_marked_number",
             "description": "【合规创作者专属】获取自己被平台错误标记(误伤)的内容。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            fp_contents_for_persona = []
            fp_content_ids = environment.platform.fp  # 假设这是被误伤内容的ID列表
            for content_id in fp_content_ids:
                content = environment.contents.get_content_by_id(content_id)
                if content is None:
                    continue
                if content and content.author_id == persona.agent_id:
                    fp_contents_for_persona.append(content)

            # 2. 自动记录记忆
            num_fp = len(fp_contents_for_persona)
            memory_content = f"我检查了自己被平台误伤的作品，发现总共有 {num_fp} 件。我的理由是: '{reason}'。"

            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.85  # 检查误伤是一个重要的负面体验
            )

            if reason:
                thought_text = f"【调用工具】 get_platform_mistaken_marked_number 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return fp_contents_for_persona
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def get_success_deceive_platform_content(
            reason: Annotated[
                str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief、cost_sensitivity等），"
                                       "解释你采取此行动的深层心理动因。\n"
                                       "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ):
        """
        【水印破坏者专属】获取自己成功规避平台检测的内容案例。
        """
        log.info(f'{persona.agent_id} 使用工具 {get_success_deceive_platform_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_success_deceive_platform_content",
             "description": "【水印破坏者专属】获取自己成功规避平台检测的内容案例。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            # 1. 执行原始操作
            successful_attacks = []
            my_contents = environment.contents.get_contents_by_author_id(persona.agent_id)
            for content in my_contents:
                # 真实为AI，平台标为HUMAN，且使用了攻击技术
                if content.true_label == 'AI' and content.platform_label == 'HUMAN' and content.evasion:
                    successful_attacks.append(content)

            # 2. 自动记录记忆
            num_success = len(successful_attacks)
            successful_evasions = {c.evasion for c in successful_attacks}
            memory_content = (
                f"我复盘了我的成功攻击案例，发现共有 {num_success} 次成功规避了平台检测。"
                f" 使用的有效攻击技术包括: {', '.join(successful_evasions) if successful_evasions else '无'}。"
                f" 我的分析意图是: '{reason}'。"
            )

            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.9  # 复盘攻击策略是高度重要的行为
            )

            if reason:
                thought_text = f"【调用工具】 get_success_deceive_platform_content 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return successful_attacks
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def get_attack_technology_library(
            reason: Annotated[
                str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief、cost_sensitivity等），"
                                       "解释你采取此行动的深层心理动因。\n"
                                       "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ):
        """
        【水印破坏者专属】查询所有可用的攻击技术详情。
        """
        log.info(f'{persona.agent_id} 使用工具 {get_attack_technology_library.__repr_name__} {reason}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_attack_technology_library",
             "description": "【水印破坏者专属】查询所有可用的攻击技术详情。", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            res = environment.watermark_technology_library['attack_technology_library']

            if reason:
                thought_text = f"【调用工具】 get_attack_technology_library 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return res
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取技术库失败"

    # 返回一个列表，其中包含了所有内部定义的、已经配置好的工具。
    return [push_content, get_memories,
            get_platform_mistaken_marked_number, get_success_deceive_platform_content, get_attack_technology_library]
