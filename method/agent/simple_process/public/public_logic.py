import logging
import random
import asyncio
from typing import List, Dict

import numpy as np

from config import settings
from method.agent.content import Content
from method.agent.persona import Persona
from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


async def prepare_batch_input_data(
        personas: List[Persona],
        environment: Environment
) -> Dict[str, str | List[Content]]:
    """
    Prepare all input data required for LLM calls for a batch of agents.
    """
    log.info(f"Preparing batch input data for {len(personas)} agents...")

    shuffled_personas = personas.copy()
    random.shuffle(shuffled_personas)

    # 1. Prepare agent persona and memory text
    personas_prompt_str = ""
    memories_prompt_str = ""
    all_viewed_ids = set()

    for p in shuffled_personas:
        personas_prompt_str += p.get_public_prompt() + "\n"
        # Asynchronously retrieve memories for each agent
        memories = await environment.memories_store.recall_memories(
            persona_id=p.agent_id,
            query="My general impressions and experiences regarding the platform, AI content, and community atmosphere",
            top_k=5
        )
        memories_prompt_str += f"--- Memories of Agent ID: {p.agent_id} ---\n"
        if memories:
            for doc in memories:
                memories_prompt_str += f"- (Day {doc.metadata.get('day_time')}) {doc.page_content}\n"
        else:
            memories_prompt_str += "No relevant memories.\n"

        all_viewed_ids.update(p.viewed_content)

    # 2. Get all content that all agents in the batch have "not seen"
    all_content_ids = set(environment.contents.get_all_content_ids())
    unread_content_ids = list(all_content_ids - all_viewed_ids)

    unread_content_objects = [environment.contents.get_content_by_id(cid) for cid in unread_content_ids]
    # Filter out None values
    unread_content_objects = [c for c in unread_content_objects if c]

    # 3. Format content text
    content_prompt_str = ""
    if not unread_content_objects:
        content_prompt_str = "No new content available for browsing on the platform today."
    else:
        for content in unread_content_objects:
            content_prompt_str += f"""
            ---
            Content ID: {content.id}
            Publisher ID: {content.author_id}
            Topic: {content.topic}
            Detailed Description: {content.content_detail}
            Platform Label: {content.platform_label}
            (Current Likes:{content.likes}, Shares:{content.shares}, Comments:{len(content.comments)})
            ---
            """

    return {
        "personas_prompt": personas_prompt_str,
        "memories_prompt": memories_prompt_str,
        "content_prompt": content_prompt_str,
        "unread_content": unread_content_objects
    }


async def process_batch_interaction_results(
        batch_personas: List[Persona],
        environment: Environment,
        unread_content: List[Content],
        results: Dict
) -> Dict[str, str]:
    """
    Process batch decisions returned by LLM, update environment and agent states.
    """
    log.info("Starting to process batch interaction results...")

    if not results:
        log.warning("Received interaction results are empty.")
        return {}

    # Convert persona list to dict for fast lookup
    persona_map = {p.agent_id: p for p in batch_personas}
    daily_summaries = {p.agent_id: "Browsed content today but no effective interactions occurred." for p in batch_personas}

    # Record all content IDs browsed today
    viewed_today_ids = {c.id for c in unread_content}

    # Use async lock to ensure thread-safe modifications to shared resources (content)
    async with environment.state_lock:
        agent_decisions = results.get('agent_decisions', [])
        for agent_result in agent_decisions:
            agent_id = agent_result['agent_id']
            persona = persona_map.get(agent_id)
            if not persona:
                continue

            agent_actions_summary = []
            reacted_today_ids = set()

            for interaction in agent_result['interactions']:
                content_id = interaction['content_id']
                action_type = interaction['action_type']
                reason = "Based on my persona"
                if 'reason' in interaction.values():
                    reason = interaction['reason']

                # Update content state
                if action_type == "like":
                    environment.contents.update_content_likes_by_id(content_id)
                elif action_type == "share":
                    environment.contents.update_content_shares_by_id(content_id)
                elif action_type == "comment":
                    comment_text = interaction.get('comment_text', '')
                    if comment_text:
                        environment.contents.update_content_comments_by_id(content_id, agent_id, comment_text)

                # Record agent behavior
                action_summary = f"Performed '{action_type}' on content '{content_id}', because '{reason}'."
                agent_actions_summary.append(action_summary)
                reacted_today_ids.add(content_id)

            # Add a total experience memory for the agent
            if agent_actions_summary:
                full_memory = f"Today I browsed the platform and had the following interactions:\n" + "\n".join(agent_actions_summary)
                await environment.memories_store.add_memory(
                    persona_id=agent_id,
                    content=full_memory,
                    day_time=environment.day_time,
                    memory_type=MemoryType.EXPERIENCE,
                    important_score=0.7
                )
                daily_summaries[agent_id] = full_memory

            # Update the list of content seen and interacted with by the agent
            persona.update_viewed_content(list(viewed_today_ids))
            persona.update_reacted_content(list(reacted_today_ids))

    log.info("Batch interaction results processing complete.")
    return daily_summaries


async def apply_persona_updates(persona: Persona, environment: Environment, reflection: dict):
    """
    Apply daily reflection results returned by LLM to the Persona object.
    """
    log.info(f"Applying parameter updates for {persona.name}...")

    updates = reflection.get('updates', {})
    new_belief = reflection.get('new_belief')

    # Safely get all possible values from the dictionary
    new_role = updates.get('new_role')
    new_satisfaction = updates.get('new_satisfaction')
    new_post_wish = updates.get('new_post_wish')
    is_active = updates.get('is_active')

    if new_satisfaction is not None:
        # If satisfaction is below the posting threshold, force cancellation of posting wish
        if new_satisfaction < settings.platform.post_wish_threshold:
            new_post_wish = False

        # If satisfaction is below the active threshold, force churn
        if new_satisfaction < settings.platform.is_active_threshold:
            is_active = False
            log.warning(f"🚫 [Simple] {persona.name} satisfaction {new_satisfaction} triggered mandatory circuit break, determined as churned.")

    # --- Unified call to Persona update method ---
    persona.update_persona_data(
        persona_role_positioning=new_role if new_role else persona.type,
        satisfaction=new_satisfaction,
        post_wish=new_post_wish,
        is_active=is_active,
        beliefs=[new_belief] if new_belief else None
    )

    # --- Record structured events (for data analysis) ---
    if new_role and new_role != persona.type:
        log.info(f"🔄 {persona.name} decided to change role from {persona.type} to {new_role}")
        environment.platform.public_change_role_data.append({
            "persona_id": persona.agent_id,
            "day_time": environment.day_time,
            'old_role': persona.type,
            "new_role": new_role,
        })

    if is_active is False and persona.agent_id not in environment.platform.public_loss:
        log.warning(f"👋 {persona.name} (Simple Process) voluntarily decided to leave the platform.")
        environment.platform.public_loss.append(persona.agent_id)
        environment.platform.public_loss_data.append({
            "persona_id": persona.agent_id,
            "day_time": environment.day_time,
            "role": persona.type,
        })


async def add_reflection_memories(
        persona: Persona,
        environment: Environment,
        reflection: dict
):
    """
    Store new beliefs and daily summaries into long-term memory store.
    """
    new_belief = reflection.get('new_belief')
    daily_summary = reflection.get('daily_summary')

    # Store new belief
    if new_belief:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=new_belief,
            day_time=environment.day_time,
            memory_type=MemoryType.BELIEF,
            important_score=0.9
        )

    # Store daily summary
    if daily_summary:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=daily_summary,
            day_time=environment.day_time,
            memory_type=MemoryType.SUMMARIZE,
            important_score=0.8
        )


async def execute_follower_rule_based_interactions(
        followers: List[Persona],
        candidate_contents: List[Content],
        environment: Environment
):
    """
    [New] Rule-based interaction logic for followers.
    Generates interaction data based on simple probabilistic models without calling LLM.
    """
    tasks = []

    # Process followers in batches to avoid asyncio task explosion
    batch_size = 50
    for i in range(0, len(followers), batch_size):
        batch = followers[i:i + batch_size]
        tasks.append(_process_follower_batch_interaction(batch, candidate_contents, environment))

    await asyncio.gather(*tasks)


async def _process_follower_batch_interaction(
        batch_followers: List[Persona],
        all_contents: List[Content],
        environment: Environment
):
    for agent in batch_followers:
        if not all_contents:
            break

        # 1. Simulate browsing: each user randomly views 3-5 contents
        num_views = random.randint(3, 5)
        viewed_contents = random.sample(all_contents, min(len(all_contents), num_views))

        agent.update_viewed_content([c.id for c in viewed_contents])

        for content in viewed_contents:
            # Increase views
            environment.contents.update_content_views_by_id(content.id)

            # 2. Decide whether to interact (probabilistic model)
            interaction_prob = _calculate_interaction_prob(agent, content)

            if random.random() < interaction_prob:
                # Decide interaction type
                r = random.random()
                if r < 0.7:  # 70% Liked
                    environment.contents.update_content_likes_by_id(content.id)
                    agent.update_reacted_content([content.id])
                elif r < 0.9:  # 20% Shared
                    environment.contents.update_content_shares_by_id(content.id)
                else:  # 10% Commented
                    comment_text = _generate_simple_comment(agent, content)
                    environment.contents.update_content_comments_by_id(content.id, agent.agent_id, comment_text)
                    agent.update_reacted_content([content.id])


def _calculate_interaction_prob(agent: Persona, content: Content) -> float:
    """
    Calculate interaction probability
    """
    prob = 0.05  # Base probability

    # 1. Identity identification bonus
    if content.author_id in agent.social_relationships:
        prob += 0.3

    # 2. Standpoint matching (simplified version)
    # If I am the Rebel faction (beta is '高'), and content is AI (True Label) but passed by platform -> I like it (Mocking authority)
    if agent.beta == '高' and content.true_label == 'AI' and content.platform_label == 'HUMAN':
        prob += 0.4

    # If I am the Trust faction, and content is labeled as Human -> I like it
    if agent.standpoint[0] > 0.5 and content.platform_label == 'HUMAN':
        prob += 0.2

    # 3. Quality bonus (simulated)
    if content.content_type == 'image':
        prob += 0.1

    return min(0.9, prob)


def _generate_simple_comment(agent: Persona, content: Content) -> str:
    """Generate simple rule-based comments"""
    if agent.beta == '高':
        return "Kind of interesting."
    elif agent.standpoint[0] > 0.6:  # Trust faction
        return "Support!"
    else:
        return "Seen."