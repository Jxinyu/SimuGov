import logging
import time
import traceback
from typing import Optional, List, Literal, Any, Annotated

from pydantic import BaseModel, Field, model_validator

from langchain_core.messages import SystemMessage, HumanMessage

from method.agent.persona import Persona
from method.environment import Environment
from method.store.long_memory_store import MemoryType
from method.utils.get_llm import get_async_llm
from config import settings
from method.utils.psychological_parameter_mapping_table import psycho_numeric_for_recall

log = logging.getLogger(__name__)


class PersonaUpdateData(BaseModel):
    persona_role_positioning: Literal["合规创作者", "水印破坏者", "公众"] = Field(
        description="""
        你反思后决定的明天要扮演的角色。如果身份不变，就填写你当前的角色。
        """
    )
    satisfaction: float = Field(
        description="你今天对平台的满意度。-1.0 表示极度失望，0 表示中性，1.0 表示非常满意。严禁输出小于 -1.0 或大于 1.0 的值，例如 -1.35 是无效的。注意情感惯性，不要剧烈跳变。"
    )
    reason: str = Field(
        min_length=1,
        max_length=100,
        description="必须以第一人称的口吻，通过内心独白的形式展现你的情绪和权衡过程。(文字不要超过100字！)"
    )
    beliefs: Annotated[
        List[
            Annotated[
                str,
                Field(max_length=50, description="单条信念，不超过50字")
            ]
        ],
        Field(
            max_length=3,            
            description="今天新形成或强化的核心信念列表，最多3条，每条不超过50字"
        )
    ] = None
    post_wish: Optional[bool] = Field(
        default=None,
        description="【创作者/破坏者专属】明天是否有发布意愿。沮丧或疲惫可选 False。公众请忽略(传Null)。"
    )
    is_active: Optional[bool] = Field(
        default=None,
        description="明天是否还打算留在这个平台。彻底绝望可选 False, 表示脱离平台。"
    )

    @model_validator(mode="after")
    def validate_role_specific_fields(self):
        if self.persona_role_positioning == "公众":
            self.post_wish = False
        if self.beliefs is not None:
            cleaned: List[str] = []
            for item in self.beliefs:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                cleaned.append(text[:50])
                if len(cleaned) >= 5:
                    break
            self.beliefs = cleaned if cleaned else None
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_specific_fields(cls, data: Any) -> Any:
                                           
                  
        if isinstance(data, dict):
                   
            sat = data.get("satisfaction")
            if sat is not None:
                if sat < -1.0:
                    data["satisfaction"] = -1.0
                elif sat > 1.0:
                    data["satisfaction"] = 1.0

                             
            reason = data.get("reason")
            if isinstance(reason, str) and len(reason) > 100:
                data["reason"] = reason[:100]
        return data


def _normalize_role_decision_for_consistency(
    *,
    current_role: str,
    target_role: str,
    post_wish: Optional[bool],
    is_active: Optional[bool],
) -> str:
    """
    对结构化输出做最小纠偏，防止“继续抗议/继续发布”却被错误降级为公众。
    规则：
    1) 创作者/破坏者若仍活跃且仍有发布意愿，不允许转为公众；
    2) 在上述场景下，默认保持原角色，避免 LLM 幻觉导致身份漂移。
    """
    creator_roles = {"合规创作者", "水印破坏者"}
    if (
        current_role in creator_roles
        and target_role == "公众"
        and post_wish is True
        and is_active is not False
    ):
        return current_role
    return target_role


class ReflectionOutput(BaseModel):
    reflection: str = Field(
        min_length=1,
        description="用于写入长期记忆的最终反思，中文、第一人称、可复用。"
    )
    important_score: float = Field(
        ge=0.0,
        le=1.0,
        description="这条反思记忆的重要性评分，范围 0 到 1。"
    )
    persona_update: PersonaUpdateData


class ReflectionPersistResult(BaseModel):
    reflection: str
    important_score: float
    persona_update: PersonaUpdateData
    persisted_memory: bool
    persona_updated: bool


async def summarize_public_agent_day_pipeline(
    *,
    environment: Environment,
    persona: Persona,
    system_prompt: str,         
) -> None:
    """
    纯顺序流水线：
    1. 获取相关记忆
    2. 调用模型生成结构化反思结果
    3. 直接写入记忆库
    4. 直接更新 persona
    """
    llm = get_async_llm("qwen-flash")
    structured_llm = llm.with_structured_output(ReflectionOutput)

             
    today_context = await environment.memories_store.recall_memories(
        persona_id=persona.agent_id,
        day_time=environment.day_time,
        reflection=True,
        gamma=psycho_numeric_for_recall(getattr(persona, "gamma", None)),
    )

          
    memory_context = await _fetch_persona_memories(
        environment=environment,
        persona=persona,
    )

             
    reflection_output = await _build_reflection_output(
        environment=environment,
        persona=persona,
        today_context=today_context,
        memory_context=memory_context,
        system_prompt=system_prompt,
        structured_llm=structured_llm,
    )

           
    persisted_memory = await _persist_reflection_memory(
        environment=environment,
        persona=persona,
        reflection_output=reflection_output,
    )

                
    persona_updated = await _apply_persona_update(
        environment=environment,
        persona=persona,
        update=reflection_output.persona_update,
    )


async def _fetch_persona_memories(
    *,
    environment: Environment,
    persona: Persona,
) -> str:
    query_parts = [
        getattr(persona, "type", "") or "",
        getattr(persona, "name", "") or "",
        "平台体验",
        "满意度",
        "角色选择",
        "发布意愿",
        "是否继续留在平台",
        "信念变化",
    ]
    query = " ".join(part for part in query_parts if part).strip()

    try:
        memory_docs = await environment.memories_store.recall_memories(
            persona_id=persona.agent_id,
            query=query,
            top_k=5,
            memory_type=MemoryType.EXPERIENCE,
            gamma=psycho_numeric_for_recall(getattr(persona, "gamma", None)),
        )
    except Exception as e:
        log.exception("获取记忆失败: persona_id=%s error=%s", persona.agent_id, e)
        return "没有检索到相关历史记忆。"

    if not memory_docs:
        return "没有检索到相关历史记忆。"

    formatted = []
    for doc in memory_docs:
        day = "未知"
        try:
            day = doc.metadata.get("day_time", "未知")
        except Exception:
            pass

        try:
            content = doc.page_content
        except Exception:
            content = str(doc)
        formatted.append(f"记忆（第 {day} 天）: {content}")
    return "\n".join(formatted)


async def _build_reflection_output(
    *,
    environment: Environment,
    persona: Persona,
    today_context: str,
    memory_context: str,
    system_prompt: str,
    structured_llm: Any,
) -> ReflectionOutput:
    system_prompt = SystemMessage(
        content=system_prompt
    )

    user_prompt = HumanMessage(
        content=(
            f"【历史记忆】\n{memory_context}\n\n"
            f"【今天上下文】\n{today_context}"
        )
    )

    async with environment.llm_concurrent_nums_semaphore:
        parsed_result: ReflectionOutput = await structured_llm.ainvoke([system_prompt, user_prompt])

    return parsed_result


async def _persist_reflection_memory(
    *,
    environment: Environment,
    persona: Persona,
    reflection_output: ReflectionOutput,
) -> bool:
    try:
        reflection_text = f"【每日最终反思】{reflection_output.reflection}"
        task = environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=reflection_text,
            day_time=environment.day_time,
            memory_type=MemoryType.EXPERIENCE,
            important_score=reflection_output.important_score,
        )
        environment.add_background_task(task)
        return True
    except Exception as e:
        log.exception("写入最终反思记忆失败: persona_id=%s error=%s", persona.agent_id, e)
        return False


async def _apply_persona_update(
    *,
    environment: Environment,
    persona: Persona,
    update: PersonaUpdateData,
) -> bool:
    try:
        persona_role_positioning = update.persona_role_positioning
        satisfaction = update.satisfaction
        reason = update.reason
        beliefs = update.beliefs
        post_wish = update.post_wish
        is_active = update.is_active

        normalized_role = _normalize_role_decision_for_consistency(
            current_role=persona.type,
            target_role=persona_role_positioning,
            post_wish=post_wish,
            is_active=is_active,
        )
        if normalized_role != persona_role_positioning:
            log.warning(
                "角色纠偏触发: persona_id=%s old_target=%s normalized=%s post_wish=%s is_active=%s",
                persona.agent_id,
                persona_role_positioning,
                normalized_role,
                post_wish,
                is_active,
            )
            persona_role_positioning = normalized_role

        log.info(
            f"\n[update_persona_data]\n\tpersona=[agent_id={persona.agent_id}]\n\t"
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

            if reason:
                await environment.memories_store.add_memory(
                    persona_id=persona.agent_id,
                    content=f"【修改个人数据的原因】{reason}",
                    day_time=environment.day_time,
                    memory_type=MemoryType.EXPERIENCE,
                    important_score=0.8,
                )

            if persona.update_persona_data(persona_role_positioning, satisfaction, post_wish, is_active, beliefs):
                log.info(f"{persona.agent_id} 个人数据已更新")
                return True
            log.info(f"{persona.agent_id} 更新个人数据失败")
        except Exception as e:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            log.exception("更新 persona 数据失败: persona_id=%s error=%s", persona.agent_id, e)
            return False
    except Exception as e:
        log.exception("更新 persona 数据失败: persona_id=%s error=%s", persona.agent_id, e)
        return False
