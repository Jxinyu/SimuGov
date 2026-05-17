from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated


class ContentCreationArgs(BaseModel):
    """定义发布内容时所需的所有参数。"""
    reason: str = Field(description="详细说明发布这篇内容的意图。")
    content_type: Literal["image", "video"] = Field(description="内容的类型。")
    topic: str = Field(description="一个简洁的主题，例如：科幻艺术。")
    content_detail: str = Field(min_length=50,
                                description="对作品视觉或内容的详细文字描述，不少于50字。仅仅是从视觉上描述内容！（不涉及技术的描述）")
    is_use_ai: bool = Field(description="该内容在制作过程中是否使用了AI技术（哪怕只是轻微润色也算）。")
    evasion: Optional[str] = Field(default=None,
                                   description="如果使用攻击，请填写攻击技术ID，否则为null。只能选择一种攻击技术")
    ai_proportion: Optional[float] = Field(
        ge=0.0, le=1.0,
        description="[仅当 is_use_ai=True 时有效] 内容的 AI 使用强度/占比 (0.0 - 1.0)。\n"
                    "- 0.0~0.2: 辅助/润色 (如降噪、拼写检查)；\n"
                    "- 0.3~0.7: 混合/协作 (如局部重绘、换背景)；\n"
                    "- 0.8~1.0: 纯生成 (如文生图、Deepfake)。"
    )
    ai_tool_price_tier: Optional[Literal["高", "中", "低"]] = Field(default="中",
                                                                    description="[仅当 is_use_ai=True 时有效] 你所使用的AI生成工具的来源等级。\n"
                                                                                "- '高': 昂贵的合规商业软件 (画质好)；\n"
                                                                                "- '中': 一般商业软件 (画质一般)；\n"
                                                                                "- '低': 开源或野生工具 (画质不稳定)。\n"
                                                                                "画质影响内容传播。"
                                                                    )


class CreatorDecision(BaseModel):
    """定义单个创作者的最终决策。"""
    action: Literal["push_content", "skip"] = Field(description="决定是发布内容还是跳过。")
    reason: str = Field(description="做出此决策的简要理由。")
    args: Optional[ContentCreationArgs] = Field(
        default=None,
        description="如果action是'push_content'，则此字段必须包含所有内容参数。"
    )


                 
class SingleCreatorBatchResult(BaseModel):
    """定义LLM为单个创作者返回的决策。"""
    agent_id: str = Field(description="做出决策的创作者智能体的ID。")
    decision: CreatorDecision = Field(description="该创作者的具体决策。")
    reasoning: str = Field(description="一步步的决策推理过程，必须体现该智能体的独特性格。")


class BatchCreatorResult(BaseModel):
    """定义整个创作者批次的最终输出。"""
    creator_decisions: List[SingleCreatorBatchResult] = Field(
        description="一个列表，包含本次批次中每一个创作者的决策结果。"
    )


class CreatorGroupPolicy(BaseModel):
    """
    定义 LLM 为某个创作者群体制定的宏观行为策略。
    """
    group_name: str = Field(description="群体的名称，用于校验。")

    post_probability: float = Field(
        ge=0.0, le=1.0,
        description="该群体中今天决定发布内容的个体的比例 (0.0-1.0)。例如 0.3 表示 30% 的人会发文。"
    )

    ai_usage_rate: float = Field(
        ge=0.0, le=1.0,
        description="在决定发布的人中，使用AI技术的比例。"
    )

    attack_rate: float = Field(
        ge=0.0, le=1.0,
        description="在决定使用AI的人中，使用对抗技术(去水印)的比例。(合规创作者此项应接近0)"
    )

    topic_pool: List[str] = Field(
        description="该群体今天可能感兴趣的3-5个热门创作主题列表。",
        min_length=1
    )

    reasoning: str = Field(description="制定该策略的社会心理学分析理由。")
