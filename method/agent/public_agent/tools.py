import logging
import traceback

from langchain_core.tools import tool
from pydantic import Field

from method.agent.persona import Persona
from typing import List, Optional, Dict, Any, Literal, Union, Annotated

from method.environment import Environment
from method.store.long_memory_store import MemoryType
from method.utils.get_llm import get_async_flash_llm
from config import settings

log = logging.getLogger(__name__)


def create_tools_end(persona: Persona, environment: Environment) -> List[tool]:
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
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
            top_k: int = 5
    ):
        """
        【回忆】根据主题(query)搜索记忆。
        """
        log.info(f'{persona.agent_id} 使用工具 {get_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_memories",
             "description": "【回忆】根据主题(query)搜索记忆。", "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
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
                return "没有找到匹配的记忆"

            # 将返回的Document对象格式化为对LLM更友好的字符串列表
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]

            # 将多条记忆合并成一个长字符串
            memories_as_string = "\n".join(formatted_memories)

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
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取记忆失败"

    @tool
    async def add_memories(
            content: str,
            important_score: float,
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
    ):
        """
        【存储记忆】存储一条内容(content)为你的记忆，并设定其重要性(0-1)。
        """
        log.info(f'{persona.agent_id} 使用工具 {add_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "add_memories",
             "description": "【存储记忆】存储一条内容(content)为你的记忆，并设定其重要性(0-1)。", "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
        try:

            # 从环境中获取当前智能体和时间
            current_persona_id = persona.agent_id

            public_end_add_memory = environment.memories_store.add_memory(
                persona_id=current_persona_id,
                content=content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=important_score,
            )
            # 添加到后台任务
            environment.add_background_task(public_end_add_memory)

            if reason:
                thought_text = f"【调用工具】 add_memories 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return "存储记忆成功"
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "存储记忆失败"

    @tool
    async def update_persona_data(
            persona_role_positioning: Annotated[
                Literal['合规创作者', '水印破坏者', '公众'],
                Field(description="【必须】你反思后决定的明天要扮演的角色。如果身份不变，就填写你当前的角色。")
            ],
            satisfaction: Annotated[
                float,
                Field(
                    ge=-1.0, le=1.0,
                    description="【必须】你今天对平台的最终满意度。范围-1.0(极度失望)到1.0(非常满意)。注意情感惯性，不要剧烈跳变。"
                )
            ],
            reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                     "解释你采取此行动的深层心理动因。\n"
                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
            beliefs: Annotated[
                Optional[List[str]],
                Field(description="你今天新形成的核心信念列表。")
            ] = None,
            post_wish: Annotated[
                Optional[bool],
                Field(description="【创作者/破坏者专属】明天是否有发布意愿。沮丧或疲惫可选 False。公众请忽略(传Null)。")
            ] = None,
            is_active: Annotated[
                Optional[bool],
                Field(description="明天是否还打算留在这个平台。彻底绝望可选 False。")
            ] = None
    ) -> str:
        """
        【最终行动】
        这是你每日反思的最后一步，也是必须执行的一步。你必须将今日的所有反思结果，通过这个工具的参数进行提交。
        """
        log.info(f'{persona.agent_id} 使用工具 {update_persona_data.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "update_persona_data",
             "description": "这是你每日反思的最后一步，也是必须执行的一步。你必须将今日的所有反思结果，通过这个工具的参数进行提交。",
             "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
        try:
            if persona.type != persona_role_positioning:
                log.info(f"{persona.agent_id} 角色定位已更新")
                environment.platform.public_change_role_data.append({
                    "persona_id": persona.agent_id,
                    "day_time": environment.day_time,
                    'old_role': persona.type,
                    "new_role": persona_role_positioning,
                    "reason": reason,
                })
                persona.beliefs.append(
                    f'【身份转变】由于 {reason}, 我决定从 [{persona.type}] 转变为 [{persona_role_positioning}]')

            if satisfaction < settings.platform.post_wish_threshold:
                post_wish = False

            if satisfaction < settings.platform.is_active_threshold:
                is_active = False
                log.warning(
                    f"🚫🚫🚫 【熔断】{persona.name} 对平台极度失望 (满意度 {satisfaction} < {-0.7})，系统判定其已流失！")

            if (is_active is False) and (persona.is_active is True):
                if persona.agent_id not in environment.platform.public_loss:
                    environment.platform.public_loss_data.append({
                        "persona_id": persona.agent_id,
                        "day_time": environment.day_time,
                        "role": persona.type,
                        "influence": persona.influence,
                        "satisfaction": satisfaction,
                        "reason": reason,
                    })
                    environment.platform.public_loss.append(persona.agent_id)

            if reason:
                thought_text = f"【调用工具】 update_persona_data 的  【思维链/CoT】 {reason}"

                # 针对案例验证部分调整
                # if settings.platform.case_validation and (is_active is False or post_wish is False) and environment.day_time < 15:
                #     log.info(f"开启案例验证  启动前6天保护机制  不存入记忆")
                # else:
                #     save_thought_task = environment.memories_store.add_agent_think_memory(
                #         persona_id=persona.agent_id,
                #         content=thought_text,
                #         day_time=environment.day_time,
                #     )
                #     environment.add_background_task(save_thought_task)

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            if persona.update_persona_data(persona_role_positioning, satisfaction, post_wish, is_active, beliefs):
                return "个人数据已更新"
            return "更新个人数据失败"
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "更新个人数据失败"

    # 返回一个列表，其中包含了所有内部定义的、已经配置好的工具。
    return [update_persona_data, get_memories]


def create_tools_browse(persona: Persona, environment: Environment) -> List[tool]:
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
    async def read_detail_content(content_ids: Annotated[List[str], Field(description="【必须】要查看的内容ID列表。")],
                                  reason: Annotated[
                                      str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                             "解释你采取此行动的深层心理动因。\n"
                                                             "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ) -> str:
        """
        阅读指定ID(content_id)的内容详情。
        """
        log.info(f'{persona.agent_id} 使用工具 {read_detail_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "read_detail_content",
             "description": "阅读指定ID(content_id)的内容详情。",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            res = ''
            for content_id in content_ids:
                async with environment.state_lock:
                    content = environment.contents.get_content_by_id(content_id)
                if content is None:
                    res = f"没有找到内容id 为：{content_id}\n"
                    continue
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
                                ---\n
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
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def browse_feed(reason: Annotated[str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                                   "解释你采取此行动的深层心理动因。\n"
                                                                   "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
                          interest_content: Annotated[str, Field(description="【必须】你感兴趣的内容。")],
                          limit: Annotated[
                              int, Field(description="你希望获取的推荐内容数量。一次性少于10个")] = 5) -> str:
        """
        浏览信息流中的新内容。
        """
        log.info(f'{persona.agent_id} 使用工具 {browse_feed.__repr_name__} {reason}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "browse_feed",
             "description": "浏览信息流中的新内容。",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. 执行原始操作
            content_str = await environment.contents.get_content_by_limit_return_str(limit, persona, interest_content,
                                                                                     environment)
            return content_str
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def react_to_content(content_id: str,
                               reason: Annotated[
                                   str, Field(description="【深度动机,必须传入这个值】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                          "解释你采取此行动的深层心理动因。\n"
                                                          "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")],
                               like: Optional[bool] = False,
                               share: Optional[bool] = False,
                               comment: Optional[str] = None) -> str:
        """“
         对指定ID(content_id)的内容进行互动：为什么互动（reason）,点赞(like)、分享(share)或评论(comment)。
        """
        log.info(f'{persona.agent_id} 使用工具 {react_to_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "react_to_content",
             "description": "对指定ID(content_id)的内容进行互动",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. 检查输入
            if not like and not share and not comment:
                return "操作失败：你必须至少提供一种反应（点赞、分享或评论）。"

            # 3. 构造记忆
            content_obj = environment.contents.get_content_by_id(content_id)

            if content_obj is None:
                return "没有找到该内容或者内容id不对"

            if persona.verify_content_is_reacted(content_id):
                return "你已经对这个内容进行了反应"

            if reason:
                thought_text = f"【调用工具】 react_to_content 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            # 2. 执行原始操作
            async with environment.state_lock:
                try:
                    if like:
                        environment.contents.update_content_likes_by_id(content_id)
                    if share:
                        environment.contents.update_content_shares_by_id(content_id)
                    if comment:
                        environment.contents.update_content_comments_by_id(content_id, persona.agent_id, comment)
                except Exception as e:
                    return f"对内容反应失败: {e}"

            persona.update_reacted_content([content_id])

            memory_content = (
                f"【观点表达】针对一篇'{content_obj.platform_label}'标签的'{content_obj.topic}'内容，内容id为：{content_obj.id}"
                f"'{'我进行了点赞' if like else ''}'。"
                f"'{'我进行了分享' if share else ''}'。"
                f"'{'我发表了评论: ' + comment if comment else ''}'。"
                f"底层动机：我对该类内容的态度是{reason}"
            )

            importance = 0.3 + (0.3 if comment else 0.0)  # 评论会增加记忆的重要性

            public_scan_react_add_memory = environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=importance
            )

            # 添加到后台处理
            environment.add_background_task(public_scan_react_add_memory)

            return f"已记录你对内容 {content_id} 的互动。"
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "操作失败"

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
            {"tool_name": "get_memories",
             "description": "【回忆】根据主题(query)搜索记忆。",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:

            # 从环境中获取当前智能体和时间
            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if reason:
                thought_text = f"【调用工具】 get_memories 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            if not memories_docs:
                return [f"没有找到与 '{query}' 相关的记忆。"]

            # 将返回的Document对象格式化为对LLM更友好的字符串列表
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]
            # 将多条记忆合并成一个长字符串
            memories_as_string = "\n".join(formatted_memories)

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

            return response.content
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取记忆失败"

    @tool
    async def update_social_relationships(social_relationships: Annotated[
        Dict[str, float], Field(
            description="格式为{'agent_id': strength}。strength范围-1.0(敌对)到1.0(盟友)。例如：{'creator_004': 0.7}")],
                                          reason: Annotated[
                                              str, Field(description="【深度动机】基于你的角色画像（特别是beta、standpoint、gamma、belief等），"
                                                                     "解释你采取此行动的深层心理动因。\n"
                                                                     "必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。")], ):
        """
        更新对其他用户的关系。
        """
        log.info(f'{persona.agent_id} 使用工具 {update_social_relationships.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "update_social_relationships",
             "description": "更新对其他用户的关系。",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. 执行原始操作
            clamped_relationships = {
                target_id: max(-1.0, min(1.0, new_strength))
                for target_id, new_strength in social_relationships.items()
            }

            async with environment.state_lock:
                persona.social_relationships.update(clamped_relationships)

            if reason:
                thought_text = f"【调用工具】 update_social_relationships 的  【思维链/CoT】 {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            # 2. 自动记录记忆
            # 我们将这个心智模型的改变记录为“信念”
            for target_id, new_strength in clamped_relationships.items():
                memory_content = f"基于最近的观察，我对 '{target_id}' 的看法发生了改变，新的关系强度是 {new_strength:.2f}。"
                public_update_social_relationships_add_memory = environment.memories_store.add_memory(
                    persona_id=persona.agent_id,
                    content=memory_content,
                    day_time=environment.day_time,
                    memory_type=MemoryType.BELIEF,  # 这是一个信念的改变
                    important_score=0.7  # 社交关系的变化是重要的信念
                )
                # 添加到后台任务
                environment.add_background_task(public_update_social_relationships_add_memory)
        except Exception as e:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            log.error(f"输入的数据： {social_relationships};;;{reason}")
            return "输入的格式不正确，示例：{'social_relationships': {'creator_004': 0.7}}"

        return True

    # 返回一个列表，其中包含了所有内部定义的、已经配置好的工具。
    return [browse_feed, read_detail_content, react_to_content, get_memories, update_social_relationships]
