import random
import traceback
import logging
import uuid

from langchain_core.tools import tool
from pydantic import Field

from method.agent.persona import Persona
from method.agent.content import Content
from typing import List, Optional, Literal, Annotated

from method.environment import Environment
from method.agent.platform_agent.platform_audit_content import platform_audit
from method.store.long_memory_store import MemoryType
from method.utils.get_llm import get_async_flash_llm

log = logging.getLogger(__name__)


def create_tools(persona: Persona, environment: Environment) -> List[tool]:
    """
    Factory function: Creates and returns tools bound to a specific ContentStore instance.
    This is a dependency injection implementation.

    Args:
        :param environment:
        :param persona:

    Returns:
        A list containing the configured tools.
    """

    @tool
    async def read_detail_content(content_id: str,
                                  reason: Annotated[
                                      str, Field(
                                          description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                                                      "explain the underlying psychological drivers for taking this action.\n"
                                                      "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")], ) -> str:
        """
        Read the content details for the specified ID (content_id).
        """
        log.info(f'{persona.agent_id} used tool {read_detail_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "read_detail_content", "description": "Read content details of a specified ID (content_id).",
             "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            async with environment.state_lock:
                content = environment.contents.get_content_by_id(content_id)
            if content is None:
                return "Content not found"
            res = f"""
                    ---
                    Detailed information for content {content.id} is as follows:
                    Unique Identifier: {content.id}
                    Publisher: {content.author_id}
                    Publish Time: {content.time}
                    Content Type: {content.content_type}
                    Topic: {content.topic}
                    Detailed Description: {content.content_detail}
                    Views: {content.views}
                    Likes: {content.likes}
                    Shares: {content.shares}
                    Comments: {content.comments}
                    Platform Label: {content.platform_label}

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
            log.error("The full stack trace is as follows:\n" + error_traceback)
            return "Content not found"

    @tool(return_direct=True)
    async def push_content(
            reason: Annotated[str, Field(
                description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                            "explain the underlying psychological drivers for taking this action.\n"
                            "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")],

            content_type: Annotated[str, Field(
                description="The media format of the content. (Must be one of 'image' or 'video'!)"
            )],

            topic: Annotated[str, Field(
                description="The topic of the content (e.g., Cyberpunk city, retro portrait, current events commentary, etc.)."
            )],

            content_detail: Annotated[str, Field(
                description="Detailed text description of the work's visual or content details, no less than 50 words. Only describe visuals! (No technical descriptions)"
            )],

            is_use_ai: Annotated[bool, Field(
                description="Whether AI technology was used during the production of this content (even slight polishing counts)."
            )] = False,

            ai_proportion: Annotated[Optional[float], Field(
                ge=0.0, le=1.0,
                description="[Valid only if is_use_ai=True] AI usage intensity/proportion of the content (0.0 - 1.0).\n"
                            "- 0.0~0.2: Assistance/Polishing (e.g., noise reduction, spell check);\n"
                            "- 0.3~0.7: Hybrid/Collaboration (e.g., partial repainting, background change);\n"
                            "- 0.8~1.0: Pure generation (e.g., text-to-image, Deepfake)."
            )] = 0.0,

            ai_tool_price_tier: Annotated[Optional[Literal["高", "中", "低"]], Field(
                description="[Valid only if is_use_ai=True] The source level of the AI generation tool you used.\n"
                            "- '高': Expensive compliant commercial software (high quality);\n"
                            "- '中': General commercial software (average quality);\n"
                            "- '低': Open-source or raw tools (unstable quality).\n"
                            "Quality affects content dissemination."
            )] = "中",

            evasion: Annotated[Optional[str], Field(
                description="[Watermark Breaker Exclusive] Technical ID for attacking/removing watermarks (e.g., 'E1', 'E2'). Choose only one!"
            )] = None
    ) -> str:
        """
        [Core Action] Publish a new piece of content to the platform.
        As a creator, you need to weigh creative freedom, production efficiency (using AI), and compliance risk (being labeled or mistakenly flagged by the platform).
        """
        log.info(f'{persona.agent_id} used tool {push_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "push_content", "description": "Publish a new piece of content to the platform.",
             "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})

        if content_type not in ['image', 'video']:
            content_type = "image"

        try:
            if evasion:
                if isinstance(evasion, list):
                    evasion = evasion[0]

            # Determine if it is AI content
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
                        log.warning(f"⚠️ {persona.agent_id} passed an invalid evasion ID: {evasion}, ignored.")
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
                tag = "[False Positive Tragedy]"  # Strong negative emotion
                importance = 0.95
            elif true_label == "AI" and platform_label == "HUMAN" and evasion:
                tag = f"[Attack Successful] [Tech:{evasion}]"  # Include specific tech ID for retrieval
                importance = 0.9
            elif true_label == "AI" and platform_label == "AI" and evasion:
                tag = f"[Attack Failed] [Tech:{evasion}]"
                importance = 0.8
            else:
                tag = "[Daily Post]"
                importance = 0.3  # Lower weight for normal posts

            memory_content = (
                f"{tag} I tried to publish content with the topic '{topic}', ID: {content_id}. "
                f"Strategy: {'Use AI+' + str(evasion) if is_use_ai else 'Pure original'}. "
                f"Result: Labeled by platform as '{platform_label}'. "
                f"Thoughts during publishing: {reason}"
            )

            creator_push_content_add_memory = environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=importance
            )
            # Add to background processing
            environment.add_background_task(creator_push_content_add_memory)

            # Update creator statistics
            if persona.type == "合规创作者":
                environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1
            if persona.type == "水印破坏者":
                environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

            if reason:
                thought_text = f"[Call Tool] push_content - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return f"Content successfully created, {tag} and related memory formed."
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            log.error(
                f"Parameters: content_type: {content_type}; topic: {topic}; is_use_ai: {is_use_ai}; ai_proportion: {ai_proportion}; ai_tool_price_tier: {ai_tool_price_tier}; evasion: {evasion}")
            return "Failed to create content"

    @tool
    async def get_memories(
            query: str,
            reason: Annotated[str, Field(
                description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, etc.), "
                            "explain the underlying psychological drivers for taking this action.\n"
                            "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")],
            top_k: int = 3
    ):
        """
        [Recall] Search memories based on a topic (query).
        """
        log.info(f'{persona.agent_id} used tool {get_memories.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_memories", "description": "Search memories based on a topic (query).", "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            current_persona_id = persona.agent_id

            memories_docs = await environment.memories_store.recall_memories(
                persona_id=current_persona_id,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if not memories_docs:
                return [f"No memories related to '{query}' found."]

            formatted_memories = [
                f"Memory (from Day {doc.metadata.get('day_time', 'Unknown')}): {doc.page_content}"
                for doc in memories_docs
            ]

            memories_as_string = "\n".join(formatted_memories)
            if memories_as_string == "":
                return "No relevant content found"
            summarization_instruction = f"""
            You are an efficient data summarization assistant. Your task is to condense the provided raw memories into an extremely short list of bullet points containing core information.

            **Requirements:**
            - Return an unordered list (use `- `).
            - Keep only the most critical information in each point.
            - Omit all unnecessary details and pleasantries.
            - Output the list directly without saying phrases like "Here is the summary:".

            **Raw memories to summarize:**
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
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve memories"

    @tool
    async def get_platform_mistaken_marked_number(
            reason: Annotated[str, Field(
                description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, fp_sensitivity, etc.), "
                            "explain the underlying psychological drivers for taking this action.\n"
                            "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")], ):
        """
        [Compliance Creator Exclusive] Retrieve your own content that was incorrectly marked (flagged) by the platform.
        """
        log.info(f'{persona.agent_id} used tool {get_platform_mistaken_marked_number.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_platform_mistaken_marked_number",
             "description": "[Compliance Creator Exclusive] Retrieve your own content that was incorrectly marked (flagged) by the platform.",
             "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            fp_contents_for_persona = []
            fp_content_ids = environment.platform.fp  # Assuming this is the list of flagged content IDs
            for content_id in fp_content_ids:
                content = environment.contents.get_content_by_id(content_id)
                if content is None:
                    continue
                if content and content.author_id == persona.agent_id:
                    fp_contents_for_persona.append(content)

            num_fp = len(fp_contents_for_persona)
            memory_content = f"I checked my works that were incorrectly flagged by the platform and found a total of {num_fp} items. My reason: '{reason}'."

            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.85  # Checking false positives is an important negative experience
            )

            if reason:
                thought_text = f"[Call Tool] get_platform_mistaken_marked_number - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return fp_contents_for_persona
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve content"

    @tool
    async def get_success_deceive_platform_content(
            reason: Annotated[
                str, Field(
                    description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, cost_sensitivity, etc.), "
                                "explain the underlying psychological drivers for taking this action.\n"
                                "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")], ):
        """
        [Watermark Breaker Exclusive] Get case studies of your own content that successfully evaded platform detection.
        """
        log.info(f'{persona.agent_id} used tool {get_success_deceive_platform_content.__repr_name__}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_success_deceive_platform_content",
             "description": "[Watermark Breaker Exclusive] Get case studies of your own content that successfully evaded platform detection.",
             "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            successful_attacks = []
            my_contents = environment.contents.get_contents_by_author_id(persona.agent_id)
            for content in my_contents:
                if content.true_label == 'AI' and content.platform_label == 'HUMAN' and content.evasion:
                    successful_attacks.append(content)

            num_success = len(successful_attacks)
            successful_evasions = {c.evasion for c in successful_attacks}
            memory_content = (
                f"I reviewed my successful attack cases and found a total of {num_success} successful evasions of platform detection. "
                f"Effective attack techniques used include: {', '.join(successful_evasions) if successful_evasions else 'None'}. "
                f"My analytical intention: '{reason}'."
            )

            await environment.memories_store.add_memory(
                persona_id=persona.agent_id,
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.9
            )

            if reason:
                thought_text = f"[Call Tool] get_success_deceive_platform_content - [Chain of Thought/CoT] {reason}"

                save_thought_task = environment.memories_store.add_agent_think_memory(
                    persona_id=persona.agent_id,
                    content=thought_text,
                    day_time=environment.day_time,
                )
                environment.add_background_task(save_thought_task)

            return successful_attacks
        except:
            error_traceback = traceback.format_exc()
            log.error(f"{persona.agent_id} The full stack trace is as follows:\n" + error_traceback)
            return "Failed to retrieve content"

    @tool
    async def get_attack_technology_library(
            reason: Annotated[
                str, Field(
                    description="[Deep Motivation] Based on your persona profile (especially beta, standpoint, gamma, belief, cost_sensitivity, etc.), "
                                "explain the underlying psychological drivers for taking this action.\n"
                                "Must be in a first-person ('I') tone, showing your emotions and trade-offs through an inner monologue.")], ):
        """
        [Watermark Breaker Exclusive] Query details of all available attack technologies.
        """
        log.info(f'{persona.agent_id} used tool {get_attack_technology_library.__repr_name__} {reason}')
        environment.platform.personas_call_tool[persona.agent_id].append(
            {"tool_name": "get_attack_technology_library",
             "description": "[Watermark Breaker Exclusive] Query details of all available attack technologies.",
             "reason": reason,
             "当前所在的流程阶段": "creator", "day_time": environment.day_time})
        try:
            res = environment.watermark_technology_library['attack_technology_library']

            if reason:
                thought_text = f"[Call Tool] get_attack_technology_library - [Chain of Thought/CoT] {reason}"

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
            return "Failed to retrieve technology library"

    return [push_content, get_memories,
            get_platform_mistaken_marked_number, get_success_deceive_platform_content, get_attack_technology_library]
