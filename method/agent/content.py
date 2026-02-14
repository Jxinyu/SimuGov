from __future__ import annotations
import math
from typing import Dict, Literal, Optional, List, Annotated

from pydantic import Field, BaseModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from method.environment import Environment
    from method.agent.persona import Persona


class Content(BaseModel):
    """
    A simple in-memory content storage class.
    During the ReAct cycle, agents interact with instances of this class through tools
    to store and retrieve content generated during runtime.
    """
    id: str = Field(..., description="A globally unique identifier ensuring every piece of content is traceable.")
    author_id: str = Field(..., description="The ID of the agent who published this content, used to link agent behavior and attributes.")
    time: int = Field(..., description="The publishing time step t, used to calculate timeliness and memory decay.")
    content_type: Literal['image', 'video'] = Field(..., description="The type of content, which can be an image or a video.")
    topic: str = Field(..., description="The topic of the content, used for classification and searching.")
    content_detail: str = Field(..., description="Detailed description of the content.")
    reason: str = Field(..., description="The reason for publishing this content.")

    watermark_id: Optional[str] = Field(description="Content watermark ID, used to identify the source of the content.")
    ai_proportion: float = Field(default=0.0, description="The objective AI generation proportion of the content.")

    views: int = Field(default=0, description="The number of views, used to evaluate the quality of the content.")
    likes: int = Field(default=0, description="The number of likes, used to evaluate the popularity of the content.")
    shares: int = Field(default=0, description="The number of shares, used to evaluate the dissemination effect of the content.")
    comments: Optional[List[Dict[str, str]]] = Field(..., description="The comments of the content, used to evaluate interactivity. Format: {Commenter ID: Content}")

    evasion: Optional[str] = Field(..., description="The attack methods the content has undergone.")
    platform_label: Optional[Literal['AI', 'HUMAN']] = Field(...,
                                                             description="The label assigned by the platform to distinguish between AI-generated and human-generated content.")
    true_label: Literal['AI', 'HUMAN'] = Field(..., description="The ground truth label of the content.")


class ContentStore:

    def __init__(self):
        self._data: Dict[str, Content] = {}  # Content list

    async def add_content(self, content: Content, environment: Environment) -> bool:
        """
        Add content
        :param environment:
        :param content:
        :return:
        """
        try:
            # Store in memory
            if content.id in self._data.keys():
                return False
            self._data[content.id] = content

            # Store in vector database
            if environment and environment.memories_store:
                # It is recommended to await directly to ensure data is stored before retrieval
                await environment.memories_store.add_content_to_db(content)
        except:
            return False

        return True

    def add_content_list(self, content_list: List[Content], environment: Environment) -> bool:
        """
        Batch add content
        :param environment:
        :param content_list:
        :return:
        """
        for content in content_list:
            if not self.add_content(content, environment):
                return False
        return True

    def get_content_by_id(self, content_id: str) -> Content | None:
        """
        Get content by ID
        :param content_id:
        :return:
        """
        try:
            content = self._data[content_id]
            # self.update_content_views_by_id(content_id=content.id)
        except:
            return None
        return content

    async def get_content_by_limit_return_str(self, limit: int, persona: Persona, interest_content: str,
                                              environment: Environment) -> str:
        """
        Get content
        :param interest_content: Point of interest
        :param environment:
        :param persona:
        :param limit:
        :return:
        """

        # Filter out content that has already been viewed
        content_ids = await environment.memories_store.recommend_contents(persona, interest_content,
                                                                          environment.day_time,
                                                                          limit)

        if len(content_ids) == 0:
            return "You have already viewed all available content."

        contents = []
        for content_id in content_ids:
            content = self.get_content_by_id(content_id)
            self.update_content_views_by_id(content_id)
            if content:
                contents.append(content)

        # Return content details
        return_content = ""
        for content in contents:
            return_content += f"""
        ---
        Information about content {content.id}:
        Unique Identifier: {content.id}
        Publisher: {content.author_id}
        Type: {content.content_type}
        Topic: {content.topic}
        Summary: {content.content_detail[:30]} + ....
        Platform Label: {content.platform_label}
        ---
        """
        return return_content

    def update_content_likes_by_id(self, content_id: str) -> bool:
        """
        Update likes count
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].likes += 1
        except:
            return False
        return True

    def update_content_shares_by_id(self, content_id: str) -> bool:
        """
        Update shares count
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].shares += 1
        except:
            return False
        return True

    def update_content_views_by_id(self, content_id: str) -> bool:
        """
        Increase views count
        :param content_id:
        :return:
        """
        try:
            self._data[content_id].views += 1
        except:
            return False
        return True

    def update_content_comments_by_id(self, content_id: str, persona_id: str, comment: str) -> bool:
        """
        Add a new comment
        :param content_id: ID of the content being commented on
        :param persona_id: Agent ID of the commenter
        :param comment: Comment content
        :return:
        """
        try:
            self._data[content_id].comments.append({persona_id: comment})
        except:
            return False
        return True

    def search_by_topic_return_str(self, topic: str, limit: int = 10) -> str:
        """
        Search by topic and return details as a string
        """
        contents = [content for k, content in self._data.items() if content.topic == topic]
        for content in contents:
            self.update_content_views_by_id(content_id=content.id)

        # Return content details
        return_content = ""
        for content in contents:
            return_content += f"""
        ---
        Information about content {content.id}:
        Unique Identifier: {content.id}
        Publisher: {content.author_id}
        Type: {content.content_type}
        Topic: {content.topic}
        Platform Label: {content.platform_label}
        ---
        """
        return return_content

    def get_end_content_id(self) -> int:
        """
        Get the ID of the last content item
        :return:
        """
        max_id = 0

        for content_id in self._data.keys():
            if content_id.isdigit():
                val = int(content_id)
                if val > max_id:
                    max_id = val

        return max_id

    def get_contents_by_author_id(self, author_id: str) -> list:
        """
        Get content by author ID
        :param author_id:
        :return:
        """
        contents = [v for k, v in self._data.items() if v.author_id == author_id]
        for content in contents:
            self.update_content_views_by_id(content_id=content.id)
        return contents

    def get_all_contents(self) -> list:
        """
        Get all content items
        :return:
        """
        return list(self._data.values())

    def get_all_contents_dict(self) -> list:
        """
        Get all content items (list of Content objects)
        :return:
        """
        return list(self._data.values())

    def get_all_content_ids(self) -> list:
        """
        Get all content IDs
        :return:
        """
        return list(self._data.keys())

    def calculate_content_influence(self, content: Content, environment, initial_score: bool = False) -> float:
        """
        Calculate the influence of content on agents (and the value of the content).
        """
        current_day = environment.day_time

        # --- 1. Dynamically calculate the standard threshold ---
        standard_x0 = calculate_dynamic_x0(environment)

        influence_weights = {
            'views': 0.1,  # Base score per view
            'likes': 0.2,  # Each like equals 2 views
            'shares': 0.5,  # Each share equals 5 views
            'comments': 0.3  # Each comment equals 3 views
        }

        time_decay_halflife_days = 2.0

        # --- 2. Calculate base influence and interaction-weighted influence ---
        views = content.views
        likes = content.likes
        shares = content.shares
        comments = content.comments
        comments_count = len(comments)

        raw_influence = (
                views * influence_weights['views'] +
                likes * influence_weights['likes'] +
                shares * influence_weights['shares'] +
                comments_count * influence_weights['comments']
        )

        # --- 3. Calculate time decay factor ---
        content_day = content.time
        days_passed = max(0, current_day - content_day)

        # Exponential decay formula
        decay_factor = 0.5 ** (days_passed / time_decay_halflife_days)

        # --- 4. Calculate final Raw Influence Score ---
        final_influence_score = raw_influence * decay_factor

        # If only the raw score is needed (e.g., for calculating regulatory penalties), return directly
        if initial_score:
            return final_influence_score

        if days_passed < 1.0:
            dynamic_x0 = standard_x0 * 0.1
        elif days_passed < 3.0:
            dynamic_x0 = standard_x0 * 0.5
        else:
            dynamic_x0 = standard_x0

        # k: Curve steepness
        k = 0.05

        # --- Sigmoid Calculation ---
        try:
            # Standard Sigmoid: Range [0, 1]
            sigmoid_value = 1 / (1 + math.exp(-k * (final_influence_score - dynamic_x0)))

            value = sigmoid_value
        except OverflowError:
            value = 0.1  # Fallback value in case of overflow

        return value


def calculate_dynamic_x0(environment):
    """
    Dynamically calculate the influence value midpoint x0 based on current environment scale.
    """
    # --- 1. Get critical agent counts ---
    total_audience = environment.initial_persona_count

    if total_audience == 0:
        return 1.0  # Avoid division by zero, return a minimal base value

    # --- 2. Estimate interaction metrics for "moderately successful" content ---

    # Assume a "mini viral post" can be seen by 100% of the potential audience
    estimated_views = total_audience * 1.0

    # Among viewers, % will like it
    estimated_likes = estimated_views * 0.6

    # Among likers, % will comment
    estimated_comments = estimated_likes * 0.6

    # Among likers, % will share
    estimated_shares = estimated_likes * 0.2

    # --- 3. Use same weights as influence calculation to calculate theoretical raw_influence ---
    influence_weights = {
        'views': 0.1,  # Base score per view
        'likes': 0.2,  # Each like equals 2 views
        'shares': 0.5,  # Each share equals 5 views
        'comments': 0.3  # Each comment equals 3 views
    }

    theoretical_raw_influence = (
            estimated_views * influence_weights['views'] +
            estimated_likes * influence_weights['likes'] +
            estimated_shares * influence_weights['shares'] +
            estimated_comments * influence_weights['comments']
    )

    # --- 4. Return this theoretical value as dynamic x0 ---
    return max(0.5, theoretical_raw_influence)
