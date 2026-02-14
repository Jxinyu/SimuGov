import asyncio
import logging
import random
import traceback
from typing import List, Dict

from method.agent.content import Content
from method.agent.persona import Persona
from method.agent.platform_agent.platform_audit_content import platform_audit
from method.agent.simple_process.creator.creator_models import CreatorGroupPolicy
from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


async def execute_group_creation_logic(
        environment: Environment,
        group_name: str,
        agents: List[Persona],
        policy: CreatorGroupPolicy
):
    """
    [Core Logic] Iterate through individuals in the group based on macro strategy,
    and decide specific behaviors via probabilistic dice rolls.
    """
    log.info(f"⚡️ Executing creation logic for group [{group_name}] (Size: {len(agents)})")

    active_count = 0

    attack_ids = list(environment.watermark_technology_library['attack_technology_library'].keys())

    watermark_map = {}  # Strength -> ID List
    for wk_id, wk_content in environment.watermark_technology_library['watermark_technology_library'].items():
        strength = wk_content['水印强度']
        if strength not in watermark_map: watermark_map[strength] = []
        watermark_map[strength].append(wk_id)

    tasks = []  # Parallel Content generation is not needed for now as add_content is already async

    for persona in agents:
        if random.random() > policy.post_probability:
            continue

        active_count += 1

        try:
            topic = random.choice(policy.topic_pool)
            is_use_ai = random.random() < policy.ai_usage_rate

            evasion = None
            watermark_id = None
            ai_proportion = 0.0
            true_label = "HUMAN"

            if is_use_ai:
                base_val = random.gauss(0.8, 0.15)
                ai_proportion = max(0.0, min(1.0, base_val))

                true_label = "AI" if ai_proportion > environment.policy.ai_threshold else "HUMAN"

                if "中" in watermark_map:
                    watermark_id = random.choice(watermark_map["中"])

                if random.random() < policy.attack_rate and attack_ids:
                    evasion = random.choice(attack_ids)

            content_id = str(environment.contents.get_end_content_id() + 1)

            platform_label = await platform_audit(
                persona, content_id, true_label, evasion, watermark_id, environment, ai_proportion
            )

            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                content_type="image",
                topic=topic,
                content_detail=f"[{group_name} Generated] Work about {topic}.",
                reason=f"Generated based on group policy (P={policy.post_probability})",
                watermark_id=watermark_id,
                platform_label=platform_label,
                true_label=true_label,
                ai_proportion=ai_proportion,
                evasion=evasion,
                is_ai_content=is_use_ai,  # If Content class has this field, otherwise ignore
                views=0, likes=0, shares=0, comments=[]
            )
            async with environment.state_lock:
                await environment.contents.add_content(content, environment)

            if persona.type == '合规创作者':
                environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1
            elif persona.type == '水印破坏者':
                environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

        except Exception as e:
            log.error(f"⚠️ Creator {persona.agent_id} failed to generate content: {e}")
            continue

    log.info(f"✅ Group [{group_name}] execution finished: {active_count}/{len(agents)} people published content.")


async def add_new_content_to_environment(
        persona: Persona,
        environment: Environment,
        args: dict
) -> tuple[Content | None, str, float]:
    """
    Core process for content creation and platform auditing.
    Returns the created content object, a result string for memory, and an importance score.
    """
    log.info(f"Executing content addition logic for {persona.name}...")
    try:
        # Safely deconstruct parameters from dictionary
        reason = args['reason']
        content_type = args['content_type']
        topic = args['topic']
        content_detail = args['content_detail']
        is_use_ai = args['is_use_ai']
        ai_tool_price_tier = args.get('ai_tool_price_tier', '中')
        ai_proportion = args.get('ai_proportion')
        evasion = args.get('evasion')

        watermark_id = None

        if ai_proportion is None:
            ai_proportion = 0.0

        if ai_proportion > environment.policy.ai_threshold:
            true_label = 'AI'
        else:
            true_label = "HUMAN"

        if is_use_ai:
            watermark_list = []
            for wk_id, wk_content in environment.watermark_technology_library['watermark_technology_library'].items():
                if wk_content['水印强度'] == ai_tool_price_tier:
                    watermark_list.append(wk_id)

            watermark_id = random.choice(watermark_list)  # Randomly select a watermark

        async with environment.state_lock:
            content_id = str(environment.contents.get_end_content_id() + 1)
            if content_type not in ['image', 'video']:
                content_type = "image"

            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                reason=reason,
                content_type=content_type,
                topic=topic,
                watermark_id=watermark_id,
                content_detail=content_detail,
                platform_label="AI",
                true_label=true_label,
                evasion=evasion,
                ai_proportion=ai_proportion,
                views=0,
                likes=0,
                shares=0,
                comments=[]
            )
            if not await environment.contents.add_content(content, environment):
                raise ValueError("Failed to create content, could not add to ContentStore.")

        # Platform auditing
        platform_label = await platform_audit(persona, content_id, true_label, evasion, watermark_id, environment,
                                              ai_proportion)

        content.platform_label = platform_label

        if persona.type == '合规创作者':
            environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1

        if persona.type == '水印破坏者':
            environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

        result_str, importance = "", 0.8
        if true_label == "HUMAN" and platform_label == "ai":
            result_str, importance = "Result: My original content was mistakenly flagged by the platform!", 0.95
        elif true_label == "AI" and platform_label == "HUMAN" and evasion:
            result_str, importance = "Result: I successfully deceived the platform's detection!", 0.9

        log.info(f"Content {content.id} created successfully. {result_str}")
        return content, result_str, importance

    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error(f"{persona.agent_id} full stack trace follows:\n" + error_traceback)
        log.error(f"Error in add_new_content_to_environment: {e}")
        return None, "An internal error occurred while creating content.", 0.5


async def add_decision_memory(
        persona: Persona,
        environment: Environment,
        decision_reason: str,
        action: str,
        content: Content = None,
        result_str: str = "",
        importance: float = 0.8
):
    """
    [Implemented Logic] Add the creator's decision and result as a memory.
    """
    memory_content = ""
    if action == "skip":
        memory_content = f"I decided not to publish content today because: '{decision_reason}'."
        importance = 0.6

    elif action == "push_content" and content:
        memory_content = (
            f"I published a piece of content (ID: {content.id}), topic is '{content.topic}'."
            f" My intention was: '{content.reason}'."
            f" I declared it as {'using AI' if content.true_label == 'AI' else 'original'}."
            f" {f'I used attack technology {content.evasion}.' if content.evasion else ''}"
            f" Finally, the platform label is '{content.platform_label}'. {result_str}"
        )

    if memory_content:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=memory_content,
            day_time=environment.day_time,
            memory_type=MemoryType.EXPERIENCE,
            important_score=importance
        )
        log.info(f"Added decision memory for {persona.name}.")


MICRO_STATES = [
    "Feeling energetic", "Feeling a bit tired", "Feeling very calm", "Feeling somewhat anxious at the moment",
    "Feeling sensitive to surroundings", "Feeling distracted", "Full of fighting spirit", "Seeking attention",
    "Wanting to stay low-key", "Feeling confused about rules", "Very confident", "A bit hesitant"
]


async def prepare_creator_batch_input(
        personas: List[Persona],
        environment: Environment
) -> Dict[str, str]:
    """Prepare all input data required for LLM calls for a batch of creators."""
    log.info(f"Preparing batch input data for {len(personas)} creators...")

    shuffled_personas = personas.copy()
    random.shuffle(shuffled_personas)

    personas_prompt_str = ""
    memories_prompt_str = ""

    for p in shuffled_personas:

        current_micro_state = random.choice(MICRO_STATES)
        personas_prompt_str += f"""
        --- Creator ID: {p.agent_id} ---
        {p.get_public_prompt()}
        [Current Temporary Micro-Psychological State]: {current_micro_state} (Please consider the subtle influence of this fleeting state on behavior when making decisions)
        ---------------------------
        """

        memories = await environment.memories_store.recall_memories(
            persona_id=p.agent_id,
            query="My recent content releases, platform feedback, and experiences related to AI content",
            top_k=5,
            memory_type=MemoryType.EXPERIENCE
        )
        memories_prompt_str += f"--- Memories of Agent ID: {p.agent_id} ---\n"
        if memories:
            for doc in memories:
                memories_prompt_str += f"- (Day {doc.metadata.get('day_time')}) {doc.page_content}\n"
        else:
            memories_prompt_str += "No relevant memories.\n"

    attack_ids = list(environment.watermark_technology_library['attack_technology_library'].keys())
    attack_ids_str = ", ".join([f"'{id}'" for id in attack_ids])

    return {
        "personas_prompt": personas_prompt_str,
        "memories_prompt": memories_prompt_str,
        "attack_ids_prompt": f"[{attack_ids_str}]"
    }


async def process_creator_batch_results(
        batch_personas: List[Persona],
        environment: Environment,
        results: Dict  # Dictionary returned from JsonParser
):
    """
    Handle batch creator decisions returned by LLM.
    """
    log.info("Starting to process creator (representative) batch decision results...")
    persona_map = {p.agent_id: p for p in batch_personas}
    creator_decisions = results.get('creator_decisions', [])

    tasks = []

    for result in creator_decisions:
        agent_id = result.get('agent_id')
        decision_data = result.get('decision', {})
        persona = persona_map.get(agent_id)

        if not persona:
            continue

        action = decision_data.get('action')
        reason = decision_data.get('reason')
        args = decision_data.get('args')

        persona._last_decision_action = action
        persona._last_decision_args = args

        tasks.append(execute_single_creator_decision(
            persona, environment, action, reason, args
        ))

    await asyncio.gather(*tasks)
    log.info("Creator (representative) batch decision results processed.")


async def generate_follower_shadow_content(
        representatives: List[Persona],
        followers: List[Persona],
        environment: Environment
):
    """
    Generate "shadow content" for followers.
    """
    if not followers or not representatives:
        return

    log.info(f"⚡️ Starting to generate entity content for {len(followers)} followers...")

    tasks = []

    active_reps = [r for r in representatives if getattr(r, '_last_decision_action', 'skip') == 'push_content']

    if not active_reps:
        log.info("  - No representatives posted; all followers remained silent.")
        return

    for follower in followers:
        role_model = random.choice(active_reps)
        args = role_model._last_decision_args

        if random.random() > 0.9:
            continue

        tasks.append(_create_single_shadow_content(follower, args, environment))

    await asyncio.gather(*tasks)
    log.info(f"✅ Follower content generation complete.")


async def _create_single_shadow_content(
        persona: Persona,
        args: dict,
        environment: Environment
):
    """
    [Internal Function] Create a content entity for a single follower.
    Does not call LLM; reuses parameters from a representative but re-runs platform Audit.
    """
    try:
        # Reuse parameters but add slight random perturbation
        reason = "Publish following group trends"
        content_type = args.get('content_type', 'image')
        topic = args.get('topic', 'Daily Sharing')
        content_detail = args.get('content_detail', '')

        is_use_ai = args.get('is_use_ai', False)
        evasion = args.get('evasion')
        base_prop = args.get('ai_proportion', 0.4)
        if base_prop is None:
            base_prop = 0
        ai_proportion = max(0.1, min(1.0, random.gauss(base_prop, 0.2)))

        true_label = "AI" if ai_proportion > environment.policy.ai_threshold else "HUMAN"

        watermark_id = None
        all_wks = list(environment.watermark_technology_library['watermark_technology_library'].keys())
        if is_use_ai or true_label == 'AI':
            if all_wks:
                watermark_id = random.choice(all_wks)

        if true_label == 'AI' and watermark_id is None:
            if all_wks:
                watermark_id = random.choice(all_wks)
                log.warning(f"Data consistency fix: Content judged as AI but no watermark; auto-completed.")
            else:
                true_label = 'HUMAN'

        # Generate ID
        async with environment.state_lock:
            content_id = str(environment.contents.get_end_content_id() + 1)

            # Create object
            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                reason=reason,
                content_type=content_type,
                topic=topic,
                content_detail=f"[Shadow] {content_detail}",
                platform_label="HUMAN",  # Pending audit
                true_label=true_label,
                ai_proportion=ai_proportion,
                evasion=evasion,
                watermark_id=watermark_id,
                views=0, likes=0, shares=0, comments=[]
            )
            await environment.contents.add_content(content, environment)

        # Platform audit
        platform_label = await platform_audit(
            persona, content_id, true_label, evasion, watermark_id, environment, ai_proportion
        )
        content.platform_label = platform_label

        # Update statistical counters
        key = '合规创作者发布内容数量' if persona.type == '合规创作者' else '水印破坏者发布内容数量'
        if environment.day_time in environment.platform.creator_data:
            environment.platform.creator_data[environment.day_time][key] += 1

    except Exception as e:
        log.error(f"Failed to create shadow content: {e}")


async def execute_single_creator_decision(
        persona: Persona,
        environment: Environment,
        action: str,
        reason: str,
        args: dict
):
    """Execute decision for a single creator, including content creation and memory addition."""
    try:
        if action == "push_content":
            if not args:
                raise ValueError("Decision is push_content, but no args provided.")

            content, result_str, importance = await add_new_content_to_environment(persona, environment, args)

            if content:
                await add_decision_memory(persona, environment, reason, action, content, result_str, importance)
            else:
                await add_decision_memory(persona, environment, reason, "skip",
                                          result_str="Tried to publish content but failed.")
        elif action == "skip":
            log.info(f"✅ Creator {persona.name} decided to skip publishing. Reason: {reason}")
            await add_decision_memory(persona, environment, reason, action)

    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error(f"{persona.agent_id} full stack trace follows:\n" + error_traceback)
        log.error(f"❌ Error executing decision for {persona.name}: {e}")
