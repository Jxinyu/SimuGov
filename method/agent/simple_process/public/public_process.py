import logging
import asyncio
import random
from typing import List, Dict

import numpy as np
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from method.agent.simple_process.group_manager import GroupManager
from method.agent.simple_process.public.public_logic import (
    prepare_batch_input_data,
    process_batch_interaction_results,
    apply_persona_updates,
    add_reflection_memories,
    execute_follower_rule_based_interactions
)
from method.agent.simple_process.public.public_models import BatchInteractionResult, DailyReflection
from method.environment import Environment
from method.agent.persona import Persona
from method.utils.get_llm import get_async_llm
from method.utils.token_statistics import token_logger
from config import settings

log = logging.getLogger(__name__)


async def run_interaction_batch(batch_personas: List[Persona], environment: Environment) -> Dict[str, str]:
    """
    Perform a complete, unified browsing-interaction LLM call for a batch of public agents.
    """
    log.info(f"⚡️ Starting linear process for a batch containing {len(batch_personas)} agents...")

    # 1. Prepare all input data
    try:
        input_data = await prepare_batch_input_data(batch_personas, environment)
        unread_content = input_data["unread_content"]

        if not unread_content:
            log.info("No new content available for this batch, skipping LLM call.")
            return {p.agent_id: "No new content browsed today." for p in batch_personas}

    except Exception as e:
        log.error(f"Error preparing batch input data: {e}")
        return {p.agent_id: f"Error during data preparation phase: {e}" for p in batch_personas}

    # 2. LLM Decision
    parser = JsonOutputParser(pydantic_object=BatchInteractionResult)

    prompt_template = """
    You are a highly intelligent social simulator. Your task is to simultaneously play the roles of multiple users (agents) on the virtual social platform "ArtStation" and decide the behaviors of all of them based on their respective personalities, memories, and the content they currently see.

    ### 🚨 Absolute Core Instruction: Eliminate Groupthink and Behavioral Convergence 🚨
    You are processing a batch of users who are **completely isolated from each other**.
    **The following behaviors are strictly prohibited:**
    1.  **Bandwagon Effect**: Do not let subsequent users imitate the behavior of preceding users.
    2.  **Ignore Differences**: Each user has a unique [current micro-state during browsing].

    ### Agent Data in Batch
    The following are the persona profiles and related memories of all agents you need to simulate this time:
    {personas_prompt}

    {memories_prompt}

    ### All Content Unread by Them on the Platform
    The following is all the new content available for browsing on the platform today:
    {content_prompt}

    ### Your Core Task
    Carefully read the setting of **each agent** and the details of **each piece of content**. Then, independently decide for **each agent** which content they will interact with (like, comment, share).

    **Batch Output**: You must return a single JSON object at once, which contains a list named 'agent_decisions'.
    ### Language Style Requirements
    Please use **firm but civilized** language. Prohibit the use of violence, threats, or extreme hate speech.

    ### !!! JSON Output Format Strict Requirements (CRITICAL) !!!
    1. **Must use standard JSON format**.
    2. **All Keys and String Values must use double quotes (")**.
    3. **Strictly forbidden** to use Python-style single quotes (').
    4. **Do not** output Markdown code block tags (e.g., ```json ... ```), only output pure JSON strings.

    **Correct Example:**
    {{"agent_decisions": [{{"agent_id": "public_01", "interactions": []}}]}}

    **Wrong Example (Single quotes forbidden):**
    {{'agent_decisions': [{{'agent_id': 'public_01', 'interactions': []}}]}}

    {format_instructions}
    """

    llm = get_async_llm(settings.model.simple_model)
    structured_llm = llm.with_structured_output(BatchInteractionResult)
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | structured_llm

    try:
        with get_openai_callback() as cb:
            async with environment.llm_concurrent_nums_semaphore:
                results = await chain.ainvoke({
                    "personas_prompt": input_data["personas_prompt"],
                    "memories_prompt": input_data["memories_prompt"],
                    "content_prompt": input_data["content_prompt"],
                })
                if hasattr(results, "model_dump"):
                    results = results.model_dump()
            token_logger.record(cb.total_tokens)
        if not results:
            log.warning(f"⚠️ Batch interaction LLM returned empty, skipping processing.")
            return {}
        # Process results (update likes, store memories)
        daily_summaries = await process_batch_interaction_results(
            batch_personas, environment, unread_content, results
        )
        return daily_summaries

    except Exception as e:
        log.error(f"❌ Error in batch interaction: {e}")
        return {}


async def linear_public_summarize_action(
        persona: Persona,
        environment: Environment,
        interaction_summary: str
):
    """Linear daily summary process for public agents."""

    # 1. LLM Decision
    parser = JsonOutputParser(pydantic_object=DailyReflection)

    prompt_template = """
    You are a user of a virtual social platform named "ArtStation". The day has ended, and now it is time for reflection and summary.

    # Your Persona Profile:
    {persona_prompt}

    # Your behavior record on the platform today:
    {interaction_summary}

    # Your Task:
    Review what you did and saw today, complete the following items, and return them in the specified JSON format:

    1.  **Form a new belief**: Distill a new **core belief**, or one reinforced by today's experience.
    2.  **Summarize today**: Write a highly condensed **daily summary** that represents your overall feeling today.
    3.  **Update your parameters**: Based on today's experience, decide whether to update parameters such as your satisfaction with the platform and posting willingness.
    4.  **[IMPORTANT] Update your role positioning**: Determine whether you need to switch roles based on your **rebellion psychology (beta)** and **today's encounters**.

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    try:
        with get_openai_callback() as cb:
            async with environment.llm_concurrent_nums_semaphore:
                reflection = await chain.ainvoke({
                    "persona_prompt": persona.get_public_prompt(),
                    "interaction_summary": interaction_summary,
                })
            token_logger.record(cb.total_tokens)

        # 2. Execute updates and memory storage
        await apply_persona_updates(persona, environment, reflection)
        await add_reflection_memories(persona, environment, reflection)

    except Exception as e:
        log.error(f"❌ Error performing linear summary decision for {persona.name}: {e}")


async def run_summarize_batch(batch_personas: List[Persona], environment: Environment, summaries: Dict[str, str]):
    """
    Perform daily summary (reflection) for a batch of representatives.
    """
    tasks = []
    for p in batch_personas:
        summary_text = summaries.get(p.agent_id, "No interaction")
        tasks.append(linear_public_summarize_action(p, environment, summary_text))
    await asyncio.gather(*tasks)


async def process_public_group_hybrid(representatives: List[Persona], followers: List[Persona],
                                      environment: Environment, group_name: str):
    """
    Process hybrid logic for a single group (Option 5: Organic Fluctuation Correction Version).
    Solves the issue of satisfaction curves being too flat and lacking realism.
    """
    # 1. Preparation
    batch_size = settings.platform.simple_batch_size
    rep_batches = [representatives[i:i + batch_size] for i in range(0, len(representatives), batch_size)]

    today_contents = [
        c for c in environment.contents.get_all_contents()
        if c.time == environment.day_time
    ]

    # --- Representative Processing ---
    if today_contents:
        for batch in rep_batches:
            await run_interaction_batch(batch, environment)

    await run_summarize_batch(representatives, environment, {})

    # --- Follower Processing ---
    if followers and today_contents:
        await execute_follower_rule_based_interactions(followers, today_contents, environment)

    # --- Satisfaction Calculation ---
    if followers:
        log.info(f"📊 [Calc] Group {group_name} settlement (Organic Mode)...")

        theta = environment.platform.theta
        visible_count = len(today_contents)

        # Get all creative feedback for today (whether misjudged/suppressed)
        today_creation_map = {
            c.author_id: c
            for c in environment.contents.get_all_contents()
            if c.time == environment.day_time
        }

        for agent in followers + representatives:
            if not agent.is_active:
                continue

            # === 1. Supply Score ===
            # Maintain the previous strong penalty logic
            if visible_count == 0:
                supply_score = -0.60
            elif visible_count < 3:
                supply_score = -0.25
            else:
                supply_score = 0.10

            # === 2. Quality Score ===
            quality_score = 0.0
            if today_contents:
                viewed_sample = random.sample(today_contents, min(visible_count, 5))
                for c in viewed_sample:
                    if c.platform_label == 'HUMAN' and c.true_label == 'HUMAN':
                        quality_score += 0.20
                    elif c.platform_label == 'HUMAN' and c.true_label == 'AI':
                        if agent.beta == '高' or agent.standpoint[1] > 0.3:
                            quality_score -= 0.30
                        else:
                            quality_score -= 0.15
                    elif c.platform_label == 'AI' and c.true_label == 'HUMAN':
                        quality_score -= 0.25
                    elif c.platform_label == 'AI':
                        quality_score -= 0.05

            # === 3. Policy Score ===
            policy_score = 0.0
            if theta < 0.1 and agent.standpoint[1] > 0.3:
                policy_score = -0.25
            if theta > 0.9 and agent.standpoint[0] > 0.3:
                policy_score = -0.15

            # === 4. Creator Pain Score ===
            creator_score = 0.0
            if agent.agent_id in today_creation_map:
                my_content = today_creation_map[agent.agent_id]

                # Judgment A: False Positive (FP) -> Critical Hit
                if my_content.true_label == 'HUMAN' and my_content.platform_label == 'AI':
                    # Scale pain based on sensitivity
                    sens_mult = {'高': 2.0, '中': 1.0, '低': 0.5}.get(agent.fp_sensitivity, 1.0)
                    # Base deduction 0.4, sensitive agents lose 0.8 (direct lethal dose)
                    creator_score = -0.4 * sens_mult

                # Judgment B: Attack Failed (TN for Breaker)
                elif my_content.true_label == 'AI' and my_content.platform_label == 'AI':
                    creator_score = -0.1

                # Judgment C: Normal release / Attack successful
                else:
                    creator_score = 0.1  # Gain sense of achievement

            # === 4. Organic Fluctuation Factor ===
            daily_mood = random.gauss(0, 0.08)

            # B. Aesthetic fatigue / Diminishing marginal utility
            current_sat = agent.satisfaction[-1] if agent.satisfaction else 0.0
            boredom_penalty = 0.0
            if current_sat > 0.5:
                boredom_penalty = -0.05 * current_sat

            # === Summary ===
            total_delta = supply_score + quality_score + policy_score + daily_mood + boredom_penalty + creator_score

            total_delta = max(-0.8, min(0.8, total_delta))

            new_sat_val = current_sat * 0.7 + total_delta
            new_sat_val = max(-1.0, min(1.0, new_sat_val))

            new_is_active = True
            if new_sat_val < settings.platform.is_active_threshold:
                new_is_active = False
                if agent.agent_id not in environment.platform.public_loss:
                    environment.platform.public_loss.append(agent.agent_id)
                    environment.platform.public_loss_data.append({
                        "persona_id": agent.agent_id,
                        "day_time": environment.day_time,
                        "role": agent.type,
                        "reason": f"Sat dropped to {new_sat_val:.2f} (Organic)"
                    })

            agent.update_persona_data(
                persona_role_positioning=agent.type,
                satisfaction=new_sat_val,
                post_wish=agent.post_wish,
                is_active=new_is_active,
                beliefs=None
            )


async def public_batch_process_main(environment: Environment):
    """
    Public process entry
    """
    groups = GroupManager.cluster_public(environment)
    log.info(f"🎯 [Public] Divided into {len(groups)} groups, starting hybrid simulation (Entity Interaction Version).")

    tasks = []
    SAMPLE_RATIO = 0.2

    for group_name, agents in groups.items():
        if not agents: continue

        representatives, followers = GroupManager.get_representative_sample(agents, ratio=SAMPLE_RATIO)
        tasks.append(process_public_group_hybrid(representatives, followers, environment, group_name))

    await asyncio.gather(*tasks)
    return {}


async def public_summarize_main_simple(environment: Environment, daily_summaries: dict):
    # This function is inlined in the hybrid process; kept empty as an extension point for now.
    pass
