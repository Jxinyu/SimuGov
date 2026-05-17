import random
import traceback
import logging

from langchain_core.tools import tool
from pydantic import Field

from method.agent.persona import Persona
from method.agent.content import Content
from typing import List, Optional, Literal, Annotated

from method.environment import Environment
from method.agent.platform_agent.platform_audit_content import platform_audit
from method.store.long_memory_store import MemoryType
from method.utils.psychological_parameter_mapping_table import psycho_numeric_for_recall

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

    @tool(return_direct=True)
    async def push_content(
            reason: Annotated[str, Field(description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)],

            content_type: Annotated[Literal["image", "video"], Field(
                description="内容的媒体形式。"
            )],

            topic: Annotated[str, Field(
                description="内容的主题"
            )],

            content_detail: Annotated[str, Field(
                description="对作品视觉或内容的详细文字描述，不少于50字。（不涉及技术的描述）", max_length=100
            )],

            is_use_ai: Annotated[bool, Field(
                description="该内容在制作过程中是否使用了AI技术。"
            )] = False,

            ai_proportion: Annotated[Optional[float], Field(
                ge=0.0, le=1.0,
                description="[仅当 is_use_ai=True 时有效] 内容的 AI 使用强度/占比 (0.0 - 1.0)。\n"
                            "- 0.0~0.2: 辅助/润色 \n"
                            "- 0.3~0.7: 混合/协作\n"
                            "- 0.8~1.0: 纯生成"
            )] = 0.0,

            ai_tool_price_tier: Annotated[Optional[Literal["高", "中", "低"]], Field(
                description="[仅当 is_use_ai=True 时有效] 你所使用的AI生成工具的来源等级。\n"
                            "- '高': 昂贵的合规商业软件 ；\n"
                            "- '中': 一般商业软件；\n"
                            "- '低': 开源或野生工具 。\n"
                            "画质影响内容传播。"
            )] = "中",

            evasion: Annotated[Optional[str], Field(
                description="【水印破坏者专属】攻击/去除水印的技术ID。只能选择一个！"
            )] = None
    ) -> str:
        """
        【核心行动】发布一条新的内容到平台。
        作为创作者，你需要权衡创作自由、生产效率（使用AI）与合规风险（被平台打标或误伤）。
        """

        if topic == 'NO AI IMAGE' and environment.day_time < 6:
            log.info(
                f"\n[push_content]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
                f"params=[reason={reason}\n\t\tcontent_type={content_type}\n\t\t"
                f"topic={topic}\n\t\tcontent_detail={content_detail}\n\t\t"
                f"is_use_ai={is_use_ai}\n\t\tai_proportion={ai_proportion}\n\t\t"
                f"ai_tool_price_tier={ai_tool_price_tier}\n\t\tevasion={evasion}]\n\t"
                f"env=[day_time={environment.day_time}]")
            res = input("请输入是否执行(y/n)：")
            if res == 'y':
                pass
            else:
                raise Exception("用户取消操作")

        log.info(
            f"\n[push_content]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[reason={reason}\n\t\tcontent_type={content_type}\n\t\t"
            f"topic={topic}\n\t\tcontent_detail={content_detail}\n\t\t"
            f"is_use_ai={is_use_ai}\n\t\tai_proportion={ai_proportion}\n\t\t"
            f"ai_tool_price_tier={ai_tool_price_tier}\n\t\tevasion={evasion}]\n\t"
            f"env=[day_time={environment.day_time}]")

        if content_type not in ['image', 'video']:
            content_type = "image"

        try:
            if evasion:
                if isinstance(evasion, list):
                    evasion = evasion[0]

                       
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

                watermark_id = random.choice(watermark_list)            
                if evasion:
                                                   
                    if "," in evasion:
                                      
                        evasion = evasion.split(",")[0].strip()
                    else:
                        evasion = evasion.strip()

                                                       
                    valid_ids = environment.watermark_technology_library['attack_technology_library'].keys()
                    if evasion not in valid_ids:
                        log.warning(f"⚠️ {persona.agent_id} 传入了无效的 evasion ID: {evasion}，已忽略。")
                        evasion = None
            else:
                evasion = None
            async with environment.state_lock:
                content_id = str(environment.contents.get_end_content_id() + 1)

                       
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

                          
            if true_label == "HUMAN" and platform_label == "AI":
                tag = "【误伤惨案】"           
                importance = 0.95
            elif true_label == "AI" and platform_label == "HUMAN" and evasion:
                tag = f"【攻击成功】[技术:{evasion}]"                 
                importance = 0.9
            elif true_label == "AI" and platform_label == "AI" and evasion:
                tag = f"【攻击失败】[技术:{evasion}]"
                importance = 0.8
            else:
                tag = "【日常发布】"
                importance = 0.3            

                                       
                                      
            memory_content = (
                f"{tag} 发布主题为'{topic}'的内容，id为 {content_id}。"
                f"策略：{'使用AI+' + str(evasion) if is_use_ai else '纯原创'}。"
                f"结果：被平台判定为'{platform_label}'。"
                f"发布内容时的思考：{reason}"                             
            )

                     
            if persona.type == "合规创作者":
                environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1
            if persona.type == "水印破坏者":
                environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

            if memory_content:
                await environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=f"{memory_content}",
                    day_time=environment.day_time,
                )

            return f"成功创建内容。{memory_content}"
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "创建内容失败"

    @tool
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)],
            top_k: int = 3
    ):
        """
        【回忆】根据主题(query)搜索记忆。
        """
        log.info(
            f"\n[get_memories]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[reason={reason}\n\t\tquery={query}\n\t\t"
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
                return [f"没有找到与 '{query}' 相关的记忆。"]

                                             
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]

                            
            memories_as_string = "\n".join(formatted_memories)
            if memories_as_string == "":
                return "没有相关内容"

            return memories_as_string
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取记忆失败"

    @tool
    async def get_platform_mistaken_marked_number_by_compliance(
            reason: Annotated[str, Field(description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)], ):
        """
        获取自己被平台错误标记(误伤)的内容。
        """
        log.info(
            f"\n[get_platform_mistaken_marked_number_by_compliance]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[reason={reason}]\n\t"
            f"env=[day_time={environment.day_time}]")
        try:
            fp_contents_for_persona = []
            fp_content_ids = environment.platform.fp                  
            for content_id in fp_content_ids:
                content = environment.contents.get_content_by_id(content_id)
                if content is None:
                    continue
                if content and content.author_id == persona.agent_id:
                    fp_contents_for_persona.append(content)

            return fp_contents_for_persona
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def get_success_deceive_platform_content_by_breaker(
            reason: Annotated[
                str, Field(description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)], ):
        """
        获取自己成功规避平台检测的内容案例。
        """
        log.info(
            f"\n[get_success_deceive_platform_content_by_breaker]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[reason={reason}]\n\t"
            f"env=[day_time={environment.day_time}]")

        try:
                       
            successful_attacks = []
            my_contents = environment.contents.get_contents_by_author_id(persona.agent_id)
            for content in my_contents:
                                          
                if content.true_label == 'AI' and content.platform_label == 'HUMAN' and content.evasion:
                    successful_attacks.append(content)

                       
            num_success = len(successful_attacks)
            successful_evasions = {c.evasion for c in successful_attacks}
            memory_content = (
                f"我复盘了我的攻击案例，一共攻击了{len(my_contents)}次，发现共有 {num_success} 次成功规避了平台检测。"
                f" 使用的有效攻击技术包括: {', '.join(successful_evasions) if successful_evasions else '无'}。"
            )

            return memory_content
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取内容失败"

    @tool
    async def get_attack_technology_library_by_breaker(
            reason: Annotated[
                str, Field(description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。", max_length=100)], ):
        """
        查询所有可用的攻击技术详情。
        """
        log.info(
            f"\n[get_attack_technology_library_by_breaker]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
            f"params=[reason={reason}]\n\t"
            f"env=[day_time={environment.day_time}]")

        try:
            attack_lib = environment.watermark_technology_library['attack_technology_library']

                                        
            def _tier(v, *, low: float = 0.33, high: float = 0.66) -> str:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return "未知"
                if v <= low:
                    return "低"
                if v <= high:
                    return "中"
                return "高"

            def _effect_tier(v, *, low: float = 0.33, high: float = 0.66) -> str:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return "未知"
                if v <= low:
                    return "弱"
                if v <= high:
                    return "中"
                return "强"

                                                                     
            attack_items = list(attack_lib.values()) if isinstance(attack_lib, dict) else attack_lib

            simplified = []
            for item in attack_items:
                if not isinstance(item, dict):
                    continue
                simplified.append({
                    "攻击标识": item.get("攻击标识"),
                    "攻击类型": item.get("攻击类型"),
                    "技术大类": item.get("技术大类"),
                    "资源消耗": _tier(item.get("资源消耗值")),
                    "成功收益": _tier(item.get("成功收益值")),
                    "质量损失": _tier(item.get("质量损失")),
                    "攻击效果": _effect_tier(item.get("攻击强度")),
                })

            return simplified
        except:
            error_traceback = traceback.format_exc()
            log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "获取技术库失败"

    @tool(return_direct=True)
    async def finish_creation(
            reason: Annotated[
                str, Field(description="结束本轮创作的原因，必须是第一人称内心独白。", max_length=120)
            ]
    ) -> str:
        """
        【结束动作】当你决定今天不发布，或已经完成发布后需要结束流程时，调用该工具结束本轮创作。
        """
        log.info(
            f"\n[finish_creation]\n\tpersona=[agent_id={persona.agent_id}]\n\tparams=[reason={reason}]\n\tenv=[day_time={environment.day_time}]")
        return f"已结束创作流程。原因：{reason}"

    tools = [push_content, finish_creation, get_memories]
    if persona.type == '水印破坏者':
        tools.extend([get_success_deceive_platform_content_by_breaker, get_attack_technology_library_by_breaker])
        return tools
    tools.extend([get_platform_mistaken_marked_number_by_compliance])
    return tools
