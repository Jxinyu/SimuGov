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
    Factory function: Creates and returns tools bound to a specific ContentStore instance.
    This is an implementation of dependency injection.

    Args:
        :param environment:
        :param persona:

    Returns:
        A list containing configured tools.
    """

    @tool
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                     "explain the deep psychological drivers for taking this action.\n"
                                                     "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
            top_k: int = 5
    ):
        """
        [Recall] Search memories based on topic (query).
        """
        log.info(f'{persona.agent_id} used tool {get_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_memories",
             "description": "[Recall] Search memories based on topic (query).", "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
        try:

            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if not memories_docs:
                return "No matching memories found"

            formatted_memories = [
                f"Memory (from Day {doc.metadata.get('day_time', 'Unknown')}): {doc.page_content}"
                for doc in memories_docs
            ]

            memories_as_string = "\n".join(formatted_memories)

            summarization_instruction = f"""
            You are an efficient data summary assistant. Your task is to condense the multiple raw memories provided below into an extremely concise list of bullet points containing core information.

            **Requirements:**
            - Return an unordered list (using `- `).
            - Keep only the most critical information for each point.
            - Omit all unnecessary details and pleasantries.
            - Directly output the list, do not say things like "Here is the summary:".

            **Raw memories to be summarized:**
            {memories_as_string}
            """
            async with environment.llm_concurrent_nums_semaphore:
                response = await get_async_flash_llm().ainvoke(summarization_instruction)

            if reason:
                thought_text = f"[Call Tool] get_memories - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return response.content
        except:
            error_traceback = traceback.format_exc()
            log.error("The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve memories"

    @tool
    async def add_memories(
            content: str,
            important_score: float,
            reason: Annotated[str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                     "explain the deep psychological drivers for taking this action.\n"
                                                     "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
    ):
        """
        [Store Memory] Store a piece of content (content) as your memory and set its importance (0-1).
        """
        log.info(f'{persona.agent_id} used tool {add_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "add_memories",
             "description": "[Store Memory] Store a piece of content (content) as your memory and set its importance (0-1).", "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
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

            if reason:
                thought_text = f"[Call Tool] add_memories - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return "Memory stored successfully"
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to store memory"

    @tool
    async def update_persona_data(
            persona_role_positioning: Annotated[
                Literal['合规创作者', '水印破坏者', '公众'],
                Field(description="[Mandatory] The role you decided to play tomorrow after reflection. If the identity remains unchanged, fill in your current role.")
            ],
            satisfaction: Annotated[
                float,
                Field(
                    ge=-1.0, le=1.0,
                    description="[Mandatory] Your final satisfaction with the platform today. Range -1.0 (extremely disappointed) to 1.0 (very satisfied). Note emotional inertia; avoid drastic jumps.")
            ],
            reason: Annotated[str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                     "explain the deep psychological drivers for taking this action.\n"
                                                     "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
            beliefs: Annotated[
                Optional[List[str]],
                Field(description="A list of core beliefs newly formed by you today.")
            ] = None,
            post_wish: Annotated[
                Optional[bool],
                Field(description="[Exclusive to Creators/Breakers] Willingness to publish tomorrow. Optional False for frustration or fatigue. Public please ignore (pass Null).")
            ] = None,
            is_active: Annotated[
                Optional[bool],
                Field(description="Whether you intend to stay on this platform tomorrow. Optional False for complete despair.")
            ] = None
    ) -> str:
        """
        [Final Action]
        This is the last step of your daily reflection and a mandatory step. You must submit all results of today's reflection through the parameters of this tool.
        """
        log.info(f'{persona.agent_id} used tool {update_persona_data.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "update_persona_data",
             "description": "This is the last step of your daily reflection and a mandatory step. You must submit all results of today's reflection through the parameters of this tool.",
             "reason": reason,
             "当前所在的流程阶段": "reflect", "day_time": environment.day_time})
        try:
            if persona.type != persona_role_positioning:
                log.info(f"{persona.agent_id} character positioning updated")
                environment.platform.public_change_role_data.append({
                    "persona_id": persona.agent_id,
                    "day_time": environment.day_time,
                    'old_role': persona.type,
                    "new_role": persona_role_positioning,
                    "reason": reason,
                })
                persona.beliefs.append(
                    f'[Identity Transition] Due to {reason}, I decided to change from [{persona.type}] to [{persona_role_positioning}]')

            if satisfaction < settings.platform.post_wish_threshold:
                post_wish = False

            if satisfaction < settings.platform.is_active_threshold:
                is_active = False
                log.warning(
                    f"🚫🚫🚫 [Circuit Breaker] {persona.name} is extremely disappointed with the platform (satisfaction {satisfaction} < {-0.7}), system determines they have churned!")

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
                thought_text = f"[Call Tool] update_persona_data - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            if persona.update_persona_data(persona_role_positioning, satisfaction, post_wish, is_active, beliefs):
                return "Personal data updated"
            return "Failed to update personal data"
        except:
            error_traceback = traceback.format_exc()
            log.error("The full stack trace is as follows:\n" + error_traceback)
            return "Failed to update personal data"

    return [update_persona_data, get_memories]


def create_tools_browse(persona: Persona, environment: Environment) -> List[tool]:
    """
    Factory function: Creates and returns tools bound to a specific ContentStore instance.
    This is an implementation of dependency injection.

    Args:
        :param environment:
        :param persona:

    Returns:
        A list containing configured tools.
    """

    @tool
    async def read_detail_content(content_ids: Annotated[List[str], Field(description="[Mandatory] List of content IDs to view.")],
                                  reason: Annotated[
                                      str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                             "explain the deep psychological drivers for taking this action.\n"
                                                             "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")], ) -> str:
        """
        Read the content details for the specified ID(s) (content_ids).
        """
        log.info(f'{persona.agent_id} used tool {read_detail_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "read_detail_content",
             "description": "Read content details for the specified ID(s).",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            res = ''
            for content_id in content_ids:
                async with environment.state_lock:
                    content = environment.contents.get_content_by_id(content_id)
                if content is None:
                    res = f"Content ID not found: {content_id}\n"
                    continue
                res = f"""
                                ---
                                Detailed information for content {content.id} is as follows:
                                Unique Identifier: {content.id}
                                Publisher: {content.author_id}
                                Publish Time: {content.time}
                                Type: {content.content_type}
                                Topic: {content.topic}
                                Detailed Description: {content.content_detail}
                                Views: {content.views}
                                Likes: {content.likes}
                                Shares: {content.shares}
                                Comments: {content.comments}
                                Platform Label: {content.platform_label}
                                ---\n
                                """

            if reason:
                thought_text = f"[Call Tool] read_detail_content - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return res
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve content"

    @tool
    async def browse_feed(reason: Annotated[str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                                   "explain the deep psychological drivers for taking this action.\n"
                                                                   "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
                          interest_content: Annotated[str, Field(description="[Mandatory] Content you are interested in.")],
                          limit: Annotated[
                              int, Field(description="The number of recommended contents you wish to obtain. Less than 10 at a time.")] = 5) -> str:
        """
        Browse new content in the feed.
        """
        log.info(f'{persona.agent_id} used tool {browse_feed.__repr_name__} {reason}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "browse_feed",
             "description": "Browse new content in the feed.",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. Execute original operation
            content_str = await environment.contents.get_content_by_limit_return_str(limit, persona, interest_content,
                                                                                     environment)
            return content_str
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve content"

    @tool
    async def react_to_content(content_id: str,
                               reason: Annotated[
                                   str, Field(description="[Deep Motivation, mandatory] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                          "explain the deep psychological drivers for taking this action.\n"
                                                          "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
                               like: Optional[bool] = False,
                               share: Optional[bool] = False,
                               comment: Optional[str] = None) -> str:
        """
        Interact with content of a specified ID (content_id): Reason for interaction (reason), like (like), share (share), or comment (comment).
        """
        log.info(f'{persona.agent_id} used tool {react_to_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "react_to_content",
             "description": "Interact with content of a specified ID",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. Check input
            if not like and not share and not comment:
                return "Operation failed: You must provide at least one reaction (like, share, or comment)."

            # 3. Construct memory
            content_obj = environment.contents.get_content_by_id(content_id)

            if content_obj is None:
                return "Content not found or invalid Content ID"

            if persona.verify_content_is_reacted(content_id):
                return "You have already reacted to this content"

            if reason:
                thought_text = f"[Call Tool] react_to_content - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            # 2. Execute original operation
            async with environment.state_lock:
                try:
                    if like:
                        environment.contents.update_content_likes_by_id(content_id)
                    if share:
                        environment.contents.update_content_shares_by_id(content_id)
                    if comment:
                        environment.contents.update_content_comments_by_id(content_id, persona.agent_id, comment)
                except Exception as e:
                    return f"Failed to react to content: {e}"

            persona.update_reacted_content([content_id])

            memory_content = (
                f"[Opinion Expression] Regarding content tagged with '{content_obj.platform_label}' on the topic '{content_obj.topic}', Content ID: {content_obj.id}. "
                f"'{'I liked it' if like else ''}' "
                f"'{'I shared it' if share else ''}' "
                f"'{'I commented: ' + comment if comment else ''}'. "
                f"Underlying motivation: My attitude towards this type of content is {reason}"
            )

            importance = 0.3 + (0.3 if comment else 0.0)  # Commenting increases memory importance

            public_scan_react_add_memory = environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=importance
            )

            # Add to background processing
            environment.add_background_task(public_scan_react_add_memory)

            return f"Interaction with content {content_id} recorded."
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Operation failed"

    @tool
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                     "explain the deep psychological drivers for taking this action.\n"
                                                     "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")],
            top_k: int = 3
    ):
        """
        [Recall] Search memories based on topic (query).
        """
        log.info(f'{persona.agent_id} used tool {get_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_memories",
             "description": "[Recall] Search memories based on topic (query).",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:

            # Get current agent and time from the environment
            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if reason:
                thought_text = f"[Call Tool] get_memories - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            if not memories_docs:
                return [f"No memories related to '{query}' found."]

            # Format Document objects into string list for LLM
            formatted_memories = [
                f"Memory (from Day {doc.metadata.get('day_time', 'Unknown')}): {doc.page_content}"
                for doc in memories_docs
            ]
            # Merge multiple memories into one long string
            memories_as_string = "\n".join(formatted_memories)

            # Define and format efficient summary prompt
            summarization_instruction = f"""
            You are an efficient data summary assistant. Your task is to condense the multiple raw memories provided below into an extremely concise list of bullet points containing core information.

            **Requirements:**
            - Return an unordered list (using `- `).
            - Keep only the most critical information for each point.
            - Omit all unnecessary details and pleasantries.
            - Directly output the list, do not say things like "Here is the summary:".

            **Raw memories to be summarized:**
            {memories_as_string}
            """
            async with environment.llm_concurrent_nums_semaphore:
                response = await get_async_flash_llm().ainvoke(summarization_instruction)

            return response.content
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve memories"

    @tool
    async def update_social_relationships(social_relationships: Annotated[
        Dict[str, float], Field(
            description="Format: {'agent_id': strength}. Strength range -1.0 (hostile) to 1.0 (ally). Example: {'creator_004': 0.7}")],
                                          reason: Annotated[
                                              str, Field(description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                                     "explain the deep psychological drivers for taking this action.\n"
                                                                     "Must be in a first-person ('I') tone, showing your emotions and trade-offs in the form of an inner monologue.")], ):
        """
        Update relationships with other users.
        """
        log.info(f'{persona.agent_id} used tool {update_social_relationships.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "update_social_relationships",
             "description": "Update relationships with other users.",
             "reason": reason,
             "当前所在的流程阶段": "scan", "day_time": environment.day_time})
        try:
            # 1. Execute original operation
            clamped_relationships = {
                target_id: max(-1.0, min(1.0, new_strength))
                for target_id, new_strength in social_relationships.items()
            }

            async with environment.state_lock:
                persona.social_relationships.update(clamped_relationships)

            if reason:
                thought_text = f"[Call Tool] update_social_relationships - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            # 2. Automatically record memories
            for target_id, new_strength in clamped_relationships.items():
                memory_content = f"Based on recent observations, my view on '{target_id}' has changed, with the new relationship strength being {new_strength:.2f}."
                public_update_social_relationships_add_memory = environment.memories_store.add_memory(
                    persona_id=persona.agent_id,
                    content=memory_content,
                    day_time=environment.day_time,
                    memory_type=MemoryType.BELIEF,  # This is a change in belief
                    important_score=0.7  # Change in social relations is an important belief
                )
                # Add to background task
                environment.add_background_task(public_update_social_relationships_add_memory)
        except Exception as e:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            log.error(f"Input data: {social_relationships};;;{reason}")
            return "Incorrect input format. Example: {'social_relationships': {'creator_004': 0.7}}"

        return True

    # Return a list containing all internally defined and configured tools.
    return [browse_feed, read_detail_content, react_to_content, get_memories, update_social_relationships]
