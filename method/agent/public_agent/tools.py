import logging
import traceback

from langchain_core.tools import tool
from pydantic import Field

from method.agent.persona import Persona
from typing import List, Optional, Dict, Any, Literal, Union, Annotated

from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings
from method.utils.psychological_parameter_mapping_table import psycho_numeric_for_recall

log = logging.getLogger(__name__)


def public_summarize_tools(persona: Persona, environment: Environment) -> List[tool]:
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
            top_k: int = 3
    ):
        """
        【回忆】根据主题(query)搜索记忆。
        """
        log.info(
            f"[get_memories]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[query={query}\n\t\treason={reason}\n\t\t"
            f"top_k={top_k}]\n\t"
            f"env=[day_time={environment.day_time}]")
        try:

                            
            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
                gamma=psycho_numeric_for_recall(getattr(persona, "gamma", None)),
            )

            if not memories_docs:
                return "没有找到匹配的记忆"

                                             
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]

                            
            memories_as_string = "\n".join(formatted_memories)

            return memories_as_string
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
        log.info(
            f"[add_memories]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[content={content}\n\t\treason={reason}\n\t\t"
            f"important_score={important_score}]\n\t"
            f"env=[day_time={environment.day_time}]")
        try:

                            
            current_persona_id = persona.agent_id

            public_end_add_memory = environment.memories_store.add_memory(
                persona_id=current_persona_id,
                content=content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=important_score,
            )
                     
            environment.add_background_task(public_end_add_memory)

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
                Field(description="你今天新形成的核心信念列表。最多5条，每条不超过50字。")
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
        这是你每日反思的最后一步，也是必须执行的一步。你必须将今日的所有反思结果，通过这个工具的参数进行提交。
        """
        log.info(
            f"[update_persona_data]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[persona_role_positioning={persona_role_positioning}\n\t\treason={reason}\n\t\t"
            f"satisfaction={satisfaction}\n\t\tbeliefs={beliefs}\n\t\tpost_wish={post_wish}\n\t\tis_active={is_active}]\n\t"
            f"env=[day_time={environment.day_time}]")

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

            if persona.update_persona_data(persona_role_positioning, satisfaction, post_wish, is_active, beliefs):
                return "个人数据已更新"
            return "更新个人数据失败"
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "更新个人数据失败"

                                   
    return [update_persona_data, get_memories]


def public_scan_tools(persona: Persona, environment: Environment) -> List[tool]:
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
    async def read_detail_content(content_ids: Annotated[List[str], Field(description="内容ID列表。")],
                                  reason: Annotated[str, Field(description="必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)]) -> str:
        """
        阅读指定ID(content_id)的内容详情。
        """
        log.info(
            f"\n[read_detail_content]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[content_ids={content_ids}\n\t\treason={reason}]\n\t\tenv=[day_time={environment.day_time}]")

        try:
            res = ''
            beliefs = getattr(persona, "beliefs", []) or []
            stance_text = " ".join([str(b) for b in beliefs if b]).strip()
                                                     
            gamma = psycho_numeric_for_recall(getattr(persona, "gamma", None), default=0.5)

            for content_id in content_ids:
                async with environment.state_lock:
                    content = environment.contents.get_content_by_id(content_id, viewer_persona=persona)
                if content is None:
                    res = f"没有找到内容id 为：{content_id}\n"
                    continue

                                   
                filtered_comments_str = "暂无评论。"
                try:
                                          
                    if stance_text and hasattr(environment.memories_store, "comments_vectorstore"):
                        topic = getattr(content, "topic", "") or ""
                        query_text = f"{topic}\n{stance_text}" if topic else stance_text

                        search_k = 20
                        results = await environment.memories_store.comments_vectorstore.asimilarity_search_with_score(
                            query=query_text,
                            k=search_k,
                            filter={"content_id": {"$eq": content.id}},
                        )

                                                 
                        base_threshold = 0.25
                        threshold = base_threshold + 0.4 * gamma                

                        candidates = []
                        for doc, distance in results:
                            sim = 1.0 - (float(distance) / 2.0)
                            sim = max(0.0, min(1.0, sim))
                            if sim >= threshold:
                                meta = getattr(doc, "metadata", {}) or {}
                                author_id = meta.get("author_id", "unknown")
                                text = getattr(doc, "page_content", "") or ""
                                                
                                if "评论:" in text:
                                    text = text.split("评论:", 1)[1].strip()
                                candidates.append((sim, author_id, text))

                        if candidates:
                                    
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            lines = [f"- [{author}] {text}" for _, author, text in candidates]
                            filtered_comments_str = "\n".join(lines)
                except Exception as e:
                    log.error(f"基于向量的评论过滤失败: {e}")
                               
                    filtered_comments_str = str(content.comments)

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
                                内容评论(已按你的立场筛选): 
{filtered_comments_str}
                                平台打标: {content.platform_label}
                                ---\n
                                """
            return res
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def browse_feed(reason: Annotated[str, Field(description="必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)],
                          interest_content: Annotated[str, Field(description="你感兴趣的内容。", max_length=20)],
                          limit: Annotated[
                              int, Field(description="你希望获取的推荐内容数量。")] = 5) -> str:
        """
        浏览信息流中的新内容。
        """
        log.info(
            f"\n[browse_feed]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[reason={reason}\n\t\tinterest_content={interest_content}\n\t\tlimit={limit}]\n\tenv=[day_time={environment.day_time}]")
        try:
                       
            content_str = await environment.contents.get_content_by_limit_return_str(limit, persona, interest_content,
                                                                                     environment)
            return content_str
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def react_to_content(content_id: Annotated[str, Field(description="内容ID。")],
                               reason: Annotated[
                                   str, Field(description="必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)],
                               like: Optional[bool] = False,
                               share: Optional[bool] = False,
                               comment: Optional[str] = None) -> str:
        """“
         对指定ID(content_id)的内容进行互动：为什么互动（reason）,点赞(like)、分享(share)或评论(comment)。
        """
        log.info(
            f"\n[react_to_content]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[content_id={content_id}\n\t\treason={reason}\n\t\tlike={like}\n\t\tshare={share}\n\t\tcomment={comment}]\n\tenv=[day_time={environment.day_time}]")

        try:
                     
            if not like and not share and not comment:
                return "操作失败：你必须至少提供一种反应（点赞、分享或评论）。"

                     
            content_obj = environment.contents.get_content_by_id(content_id)

            if content_obj is None:
                return "没有找到该内容或者内容id不对"

            if persona.verify_content_is_reacted(content_id):
                return "你已经对这个内容进行了反应"
                       
            async with environment.state_lock:
                try:
                    if like:
                        environment.contents.update_content_likes_by_id(content_id)
                    if share:
                        environment.contents.update_content_shares_by_id(content_id)
                    if comment:
                        environment.contents.update_content_comments_by_id(content_id, persona.agent_id, comment)
                                                
                        try:
                            await environment.memories_store.add_comment_to_db(
                                content_obj=content_obj,
                                comment_text=comment,
                                author_id=persona.agent_id,
                                day_time=environment.day_time,
                            )
                        except Exception as e:
                            log.error(f"写入评论向量库失败: {e}")
                except Exception as e:
                    return f"对内容反应失败: {e}"

            persona.update_reacted_content([content_id])

            return f"已记录你对内容 {content_id} 的互动。"
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "操作失败"

    @tool
    async def update_social_relationships(social_relationships: Annotated[
        Dict[str, float], Field(
            description="格式为{'agent_id': strength}。strength范围-1.0(敌对)到1.0(盟友)。例如：{'creator_004': 0.7}")],
                                          reason: Annotated[
                                              str, Field(description="必须以第一人称('我')的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)], ):
        """
        更新对其他用户的关系。
        """
        log.info(
            f"\n[update_social_relationships]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[social_relationships={social_relationships}\n\t\treason={reason}]\n\tenv=[day_time={environment.day_time}]")

        try:
                       
            clamped_relationships = {
                target_id: max(-1.0, min(1.0, new_strength))
                for target_id, new_strength in social_relationships.items()
            }
            async with environment.state_lock:
                persona.social_relationships.update(clamped_relationships)

        except Exception as e:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            log.error(f"输入的数据： {social_relationships};;;{reason}")
            return "输入的格式不正确，示例：{'social_relationships': {'creator_004': 0.7}}"

        return True

    @tool(return_direct=True)
    async def finish_browsing(
            reason: Annotated[
                str,
                Field(description="结束本轮浏览的原因，必须是第一人称内心独白。", max_length=120)
            ]
    ) -> str:
        """
        【结束动作】当你认为没有新信息、已完成目标或达到疲惫阈值时，调用该工具结束本轮浏览。
        """
        log.info(
            f"\n[finish_browsing]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[reason={reason}]\n\tenv=[day_time={environment.day_time}]")
        return f"已结束浏览。原因：{reason}"

                                   
    return [browse_feed, read_detail_content, react_to_content, update_social_relationships, finish_browsing]
