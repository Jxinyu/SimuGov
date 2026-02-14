from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class PersonaUpdateParams(BaseModel):
    """Define agent parameters that need to be updated. All fields are optional."""
    # --- New Fields ---
    new_role: Optional[Literal['合规创作者', '水印破坏者', '公众']] = Field(
        default=None,
        description="New role positioning decided after reflection. If the role remains unchanged, it is null."
    )
    new_satisfaction: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0,
        description="Updated satisfaction score for the platform based on today's experiences."
    )
    new_post_wish: Optional[bool] = Field(default=None, description="Update posting willingness.")
    is_active: Optional[bool] = Field(default=None, description="Whether decided to continue staying on the platform.")


class DailyReflection(BaseModel):
    """Define the complete output of daily reflection."""
    new_belief: str = Field(description="A new core belief about the world or the platform distilled from today's experiences.")
    daily_summary: str = Field(description="A single sentence highly summarizing today's overall feeling.")
    updates: PersonaUpdateParams = Field(description="Parameter updates that need to be applied to the persona.")


class AgentInteractionDecision(BaseModel):
    """Define specific interaction behaviors of a single agent toward a single content item."""
    content_id: str = Field(description="The unique ID of the content decided for interaction.")
    action_type: Literal["like", "comment", "share"] = Field(
        description="The type of interaction to perform: like, comment, or share."
    )
    comment_text: Optional[str] = Field(
        default=None,
        description="If action_type is 'comment', this field must contain the specific content of the comment."
    )
    reason: str = Field(description="Why you decided to perform this interaction on this content.")


class SingleAgentBatchResult(BaseModel):
    """Define all interaction decisions returned by the LLM for a single agent."""
    agent_id: str = Field(description="The ID of the agent making these decisions.")
    interactions: List[AgentInteractionDecision] = Field(
        description="List of interactions the agent decided to perform. If indifferent to all content, return an empty list []."
    )


class BatchInteractionResult(BaseModel):
    """Define the final output of the entire batch, containing decisions of all agents."""
    agent_decisions: List[SingleAgentBatchResult] = Field(
        description="A list containing the decision results for each agent in this batch."
    )


class ReactionRule(BaseModel):
    """
    Define a specific interaction rule.
    """
    target_content_type: Literal['AI_LABELED', 'HUMAN_LABELED', 'ANY'] = Field(
        description="The type of content label the rule applies to."
    )

    action_type: Literal['IGNORE', 'LIKE', 'SHARE', 'COMMENT'] = Field(
        description="The interaction behavior after the rule is triggered."
    )

    probability: float = Field(
        ge=0.0, le=1.0,
        description="The probability of triggering this interaction."
    )

    satisfaction_impact: float = Field(
        ge=-0.1, le=0.1,
        description="The subtle impact of a single interaction on individual satisfaction. For example: liking content you like (+0.01), feeling disgruntled while ignoring content you dislike (-0.02)."
    )


class PublicGroupPolicy(BaseModel):
    """
    Group macro reaction strategy.
    """
    group_name: str
    rules: List[ReactionRule] = Field(description="A list of reaction rules for this group.")
    base_satisfaction_change: float = Field(
        description="Natural daily change in satisfaction due to the overall environment atmosphere, excluding specific interactions."
    )
    churn_probability: float = Field(
        description="The probability of group members churning today due to despair."
    )
