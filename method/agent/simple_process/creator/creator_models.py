from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated


class ContentCreationArgs(BaseModel):
    """Defines all parameters required when publishing content."""
    reason: str = Field(description="Detailed explanation of the intent for publishing this content.")
    content_type: Literal["image", "video"] = Field(description="The type of content.")
    topic: str = Field(description="A concise topic, e.g., Sci-fi Art.")
    content_detail: str = Field(min_length=50,
                                description="A detailed textual description of the work's visuals or content, no less than 50 words. Only describe visuals! (No technical descriptions involved)")
    is_use_ai: bool = Field(description="Whether AI technology was used during the production of this content (even slight polishing counts).")
    evasion: Optional[str] = Field(default=None,
                                   description="If an attack is used, provide the attack technology ID; otherwise, null. Only one attack technology can be selected.")
    ai_proportion: Optional[float] = Field(
        ge=0.0, le=1.0,
        description="[Valid only when is_use_ai=True] The intensity/proportion of AI usage in the content (0.0 - 1.0).\n"
                    "- 0.0~0.2: Assistant/Polishing (e.g., noise reduction, spell check);\n"
                    "- 0.3~0.7: Mixed/Collaboration (e.g., partial repainting, background replacement);\n"
                    "- 0.8~1.0: Pure generation (e.g., text-to-image, Deepfake)."
    )
    ai_tool_price_tier: Optional[Literal["高", "中", "低"]] = Field(default="中",
                                                                    description="[Valid only when is_use_ai=True] The source level of the AI generation tool you used.\n"
                                                                                "- '高': Expensive compliant commercial software (high image quality);\n"
                                                                                "- '中': General commercial software (average image quality);\n"
                                                                                "- '低': Open-source or raw tools (unstable image quality).\n"
                                                                                "Image quality affects content dissemination."
                                                                    )


class CreatorDecision(BaseModel):
    """Defines the final decision of a single creator."""
    action: Literal["push_content", "skip"] = Field(description="Decide whether to publish content or skip.")
    reason: str = Field(description="A brief reason for making this decision.")
    args: Optional[ContentCreationArgs] = Field(
        default=None,
        description="If the action is 'push_content', this field must contain all content parameters."
    )


class SingleCreatorBatchResult(BaseModel):
    """Defines the decision returned by the LLM for a single creator."""
    agent_id: str = Field(description="The ID of the creator agent making the decision.")
    decision: CreatorDecision = Field(description="The specific decision of this creator.")
    reasoning: str = Field(description="A step-by-step decision reasoning process that must reflect the agent's unique personality.")


class BatchCreatorResult(BaseModel):
    """Defines the final output of the entire creator batch."""
    creator_decisions: List[SingleCreatorBatchResult] = Field(
        description="A list containing the decision results for each creator in this batch."
    )


class CreatorGroupPolicy(BaseModel):
    """
    Defines the macro behavioral strategy formulated by the LLM for a certain group of creators.
    """
    group_name: str = Field(description="The name of the group, used for validation.")

    post_probability: float = Field(
        ge=0.0, le=1.0,
        description="The proportion of individuals in this group who decide to publish content today (0.0-1.0). For example, 0.3 means 30% of the people will post."
    )

    ai_usage_rate: float = Field(
        ge=0.0, le=1.0,
        description="Among those who decide to publish, the proportion using AI technology."
    )

    attack_rate: float = Field(
        ge=0.0, le=1.0,
        description="Among those who decide to use AI, the proportion using adversarial techniques (watermark removal). (This value should be close to 0 for compliant creators)"
    )

    topic_pool: List[str] = Field(
        description="A list of 3-5 popular creation topics that this group might be interested in today.",
        min_length=1
    )

    reasoning: str = Field(description="Social psychological analysis reasoning for formulating this strategy.")
