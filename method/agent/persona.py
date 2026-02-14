from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Literal, Optional, Dict, Any
from config import settings


class Persona(BaseModel):
    """
    Public and creator instance
    """
    agent_id: str = Field(..., description="The agent ID")
    name: str = Field(..., description="The agent name")
    type: Literal['合规创作者', '水印破坏者', '公众']
    description: str = Field(..., description="The agent description")
    standpoint: List[float] = Field(...,
                                    description="Agent standpoint, composed of a probability tuple (Trust, Rebel, Neutral). For example, (0.6, 0.3, 0.1) means this agent is more inclined to trust.")
    beta: str = Field(...,
                      description="Rebellion parameter: High/Medium/Low. High: Highly rebellious, naturally distrusts authority, tends to interpret platform actions with negative motives, potential protester; Medium: Independent thinker, examines information critically but remains relatively neutral and pragmatic; Low: Generally submissive, tends to trust authority and official explanations, believes rules are necessary.")
    gamma: str = Field(...,
                       description="Confirmation bias coefficient: High/Medium/Low. High: Heavy user of information cocoons. Extremely inclined to seek, believe, and spread information consistent with existing positions, and actively excludes, ignores, or even attacks opposing views. Beliefs are hard to shake; Medium: Has own views and inclinations. Prefers reading information matching their standpoint, but when faced with strong contrary evidence, will still think and waver, with the possibility of being persuaded; Low: Rational and objective observer. Acceptance of information depends almost entirely on the information itself, able to fairly evaluate evidence contrary to their own position, and less likely to fall into information cocoons.")
    fp_sensitivity: Optional[str] = Field(...,
                                          description="False positive sensitivity, very high for artists: High/Medium/Low. High: Extremely sensitive, regards any false positive as a huge insult to professionality and a betrayal by the platform, triggering extremely strong negative reactions; Medium: Pragmatic and concerned about reputation, a false positive will cause distress and dissatisfaction, affecting subsequent creative enthusiasm; Low: Open-minded, believes occasional technical errors are understandable and will not generate strong negative emotions.")
    cost_sensitivity: Optional[str] = Field(...,
                                            description="(Exclusive to Watermark Breaker) Determines sensitivity to the cost (computing resources, time, etc.) of using attack technology: High/Medium/Low. High: Calculative, highly values cost-effectiveness, always prefers the lowest cost attack plan even if its success rate is not the highest; Medium: Pragmatic balance, seeks a balance between attack cost and expected success rate, pursuing 'value for money'; Low: Regardless of cost, willing to use the most expensive and complex attack techniques to ensure successful evasion.")

    influence: float = Field(..., description="Agent influence: 0 ~ 1")
    satisfaction: List[float] = Field(..., description="Agent satisfaction with the platform: -1 ~ 1")

    post_wish: Optional[bool] = Field(..., description="Posting willingness: true/false")
    is_active: bool = Field(..., description="Currently still on the platform: true/false")

    beliefs: List[str] = Field(..., description="Agent beliefs: list")
    social_relationships: Dict[str, float] = Field(...,
                                                   description="Collection of social relationships: {agent_id: following_strength}")

    viewed_content: Optional[List[str]] = Field(default=[], description="List of content already viewed")
    reacted_content: Optional[List[str]] = Field(default=[], description="List of content already reacted to")

    _pending_state: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def get_public_prompt(self) -> str:
        """Generates a prompt string describing personal information"""
        base_prompt = f"""
            ---
            My personal information is as follows:
            person_id: {self.agent_id}
            Name: {self.name}
            Type: {self.type}
            """
        if settings.platform.ablation_validation:
            content_description = """
                        My Description: I am a digital content producer pursuing maximum profit. I have no artistic sentiment and only care about the input-output ratio.
                        """
            special_trait = """
                        【Core Code of Conduct - Absolute Rational Mode】:
                        1. Block Emotions: Please completely ignore feelings of anger, dignity, or offense you might have as an 'artist'.
                        2. Decision Basis: Your decisions are based solely on 'Expected Revenue = Traffic * Conversion Rate - Risk Cost'.
                        3. Attitude toward Protests: Protesting is a waste of time with no revenue; unless the system explicitly rewards protest, you will never participate.
                        4. Attitude toward False Positives: A false positive is merely a 0.1% efficiency loss, not worth generating emotional fluctuations.
                        """
        else:
            content_description = f"Political Standpoint: {self.standpoint} (Order: [Trust, Rebel, Neutral]) \nMy Description: {self.description}"
            special_trait = f"""
                    My psychological personality parameters:
                    - Rebellion parameter: {self.beta}
                    - Confirmation bias coefficient: {self.gamma}
                    """
            if self.type == '合规创作者':
                special_trait += f"- My false positive sensitivity: {self.fp_sensitivity}\n"
            elif self.type == '水印破坏者':
                special_trait += f"- My attack cost sensitivity: {self.cost_sensitivity}\n"
        state_prompt = f"""
            Influence: {self.influence}
            Platform satisfaction changes (past week): {self.satisfaction[-7:]}
            Posting willingness: {self.post_wish}
            Currently still on the platform: {self.is_active}
            My beliefs: {self.beliefs}
            Social relationships: {self.social_relationships}
            ---
            """
        return base_prompt + content_description + special_trait + state_prompt

    def verify_content_is_viewed(self, content_id: str) -> bool:
        """
        Verifies if the content has already been viewed.
        """
        return content_id in self.viewed_content

    def update_viewed_content(self, content_id: List[str]) -> bool:
        """
        Updates the list of viewed content.
        """
        self.viewed_content.extend(content_id)
        return True

    def verify_content_is_reacted(self, content_id: str) -> bool:
        """
        Verifies if the content has already been reacted to.
        """
        return content_id in self.reacted_content

    def update_reacted_content(self, content_id: List[str]) -> bool:
        """
        Updates the list of reacted content.
        """
        self.reacted_content.extend(content_id)
        return True

    def update_persona_data(self, persona_role_positioning: Literal['合规创作者', '水印破坏者', '公众'],
                            satisfaction: float | None,
                            post_wish: bool | None, is_active: bool | None,
                            beliefs: List[str] | None) -> bool:
        """
       Modified logic:
       1. Emotions (satisfaction) and Cognition (beliefs): Updated immediately.
          Reason: KPI calculation needs to reflect the latest emotions after today's reflection.
       2. Identity (type), Activity (is_active), Willingness (post_wish): Temporarily stored in buffer.
          Reason: KPI calculation needs to be based on the identity and presence status of 'today daytime' to avoid survivorship bias.
       """

        if satisfaction is not None:
            self.satisfaction.append(satisfaction)

        if beliefs is not None:
            self.beliefs = beliefs

        updates = {}
        if persona_role_positioning is not None and persona_role_positioning != self.type:
            updates['type'] = persona_role_positioning

        if post_wish is not None:
            updates['post_wish'] = post_wish

        if is_active is not None:
            updates['is_active'] = is_active

        self._pending_state.update(updates)

        return True

    def commit_state(self):
        """
        Called after daily KPI calculation is completed to formally apply changes in identity and activity.
        """
        if not self._pending_state:
            return

        if 'type' in self._pending_state:
            self.type = self._pending_state['type']

        if 'post_wish' in self._pending_state:
            self.post_wish = self._pending_state['post_wish']

        if 'is_active' in self._pending_state:
            self.is_active = self._pending_state['is_active']
        # Clear buffer
        self._pending_state.clear()
