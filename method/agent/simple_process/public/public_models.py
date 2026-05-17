from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class PersonaUpdateParams(BaseModel):
    """定义需要更新的智能体参数。所有字段都是可选的。"""
                  
    new_role: Optional[Literal['合规创作者', '水印破坏者', '公众']] = Field(
        default=None,
        description="反思后决定的新角色定位。如果角色不变，则为 null。"
    )
    new_satisfaction: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0,
        description="根据今天的经历更新的对平台的新满意度分数。"
    )
    new_post_wish: Optional[bool] = Field(default=None, description="更新发布意愿。")
    is_active: Optional[bool] = Field(default=None, description="是否决定继续留在平台。")


class DailyReflection(BaseModel):
    """定义每日总结的完整输出。"""
    new_belief: str = Field(
        max_length=50,
        description="根据今天的经历，提炼出的一个关于世界或平台的新核心信念（每条不超过50字）。"
    )
    daily_summary: str = Field(description="一句话高度概括今天的整体感受。")
    updates: PersonaUpdateParams = Field(description="需要对自身 persona 进行的参数更新。")


class AgentInteractionDecision(BaseModel):
    """定义单个智能体对单个内容的具体互动行为。"""
    content_id: str = Field(description="决定要进行互动的内容的唯一ID。")
    action_type: Literal["like", "comment", "share"] = Field(
        description="要执行的互动类型：点赞、评论或分享。"
    )
    comment_text: Optional[str] = Field(
        default=None,
        description="如果action_type是'comment'，这里必须包含评论的具体内容。"
    )
    reason: str = Field(description="为什么你决定对这个内容进行此项互动。")


class SingleAgentBatchResult(BaseModel):
    """定义LLM为单个智能体返回的所有互动决策。"""
    agent_id: str = Field(description="做出这些决策的智能体的ID。")
    interactions: List[AgentInteractionDecision] = Field(
        description="该智能体决定要执行的互动列表。如果对所有内容都无动于衷，则返回空列表[]。"
    )


class BatchInteractionResult(BaseModel):
    """定义整个批次的最终输出，包含所有智能体的决策。"""
    agent_decisions: List[SingleAgentBatchResult] = Field(
        description="一个列表，包含本次批次中每一个智能体的决策结果。"
    )


class ReactionRule(BaseModel):
    """
    定义一条具体的互动规则。
    """
    target_content_type: Literal['AI_LABELED', 'HUMAN_LABELED', 'ANY'] = Field(
        description="规则适用的内容标签类型。"
    )

    action_type: Literal['IGNORE', 'LIKE', 'SHARE', 'COMMENT'] = Field(
        description="触发规则后的互动行为。"
    )

    probability: float = Field(
        ge=0.0, le=1.0,
        description="触发该互动的概率。"
    )

    satisfaction_impact: float = Field(
        ge=-0.1, le=0.1,
        description="单次互动对个体满意度的微小影响。例如：看到喜欢的内容点赞(+0.01)，看到讨厌的内容虽然忽视但心里不爽(-0.02)。"
    )


class PublicGroupPolicy(BaseModel):
    """
    群体宏观反应策略。
    """
    group_name: str
    rules: List[ReactionRule] = Field(description="该群体的一组反应规则列表。")
    base_satisfaction_change: float = Field(
        description="除具体互动外，因整体环境氛围导致的每日满意度自然变化量。"
    )
    churn_probability: float = Field(
        description="该群体中今天感到绝望而流失的概率。"
    )
