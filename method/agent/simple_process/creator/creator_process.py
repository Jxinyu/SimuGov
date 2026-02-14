import logging
import asyncio
import random
import traceback
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from openai import BadRequestError

from method.agent.simple_process.creator.creator_logic import prepare_creator_batch_input, \
    process_creator_batch_results, execute_group_creation_logic, generate_follower_shadow_content
from method.agent.simple_process.creator.creator_models import BatchCreatorResult, CreatorGroupPolicy
from method.agent.simple_process.group_manager import GroupManager
from method.utils.get_llm import get_async_llm
from method.environment import Environment
from method.agent.persona import Persona
from method.utils.token_statistics import token_logger
from langchain_community.callbacks import get_openai_callback
from config import settings

log = logging.getLogger(__name__)


async def run_creator_batch(batch_personas: List[Persona], environment: Environment):
    """Executes a complete and unified LLM call and processing for a batch of creators."""
    log.info(f"⚡️ Starting linear process for a batch containing {len(batch_personas)} creators...")

    if not batch_personas:
        return 0

    log.info(f"⚡️ [Batch-LLM] Processing {len(batch_personas)} creator representatives...")

    try:
        input_data = await prepare_creator_batch_input(batch_personas, environment)
    except Exception as e:
        log.error(f"Error preparing data: {e}")
        return 0

    # 2. LLM Decision
    parser = JsonOutputParser(pydantic_object=BatchCreatorResult)

    prompt_template = """
    You are a highly intelligent social simulator. You need to simultaneously play the roles of multiple creators on the virtual social platform "ArtStation".

    ### 🚨 HIGHEST PRIORITY INSTRUCTION: INDEPENDENCE & DIFFERENTIATION 🚨
    You are dealing with a simulation of **parallel universes**. Every creator in the list exists in **completely isolated** space-time.
    The following behaviors are **STRICTLY PROHIBITED** (otherwise the task fails):
    1.  **Groupthink**: Prohibition of letting Agent B's decision refer to Agent A's decision. If Agent A decides to skip, Agent B **is entirely likely** to decide to publish.
    2.  **Formulaic Output**: Do not generate similar reasons for everyone.
    3.  **Ignore Micro-states**: Each creator has a **[Current Temporary Micro-Psychological State]** (e.g., "tired", "excited"). You must make each person's decision logic significantly different based on this random state.

    ### Creator Data in Batch
    The following are the character personas and related memories of all creators you need to simulate this time:
    {personas_prompt}

    {memories_prompt}

    ### Available Attack Technology ID List (For all Watermark Breakers reference)
    {attack_ids_prompt}

    ### Your Core Task
    Independently decide the action for **each agent** today: whether to publish a new piece of content or skip.

    ### !!! STRICT RULES FOR JSON OUTPUT !!!
    The JSON object you output must strictly follow the conditional logic below; otherwise, the program will fail to parse and throw an error:

    1.  **IF `action` is `"push_content"` THEN:**
        *   The `args` field **must not** be `null`.
        *   The `args` field **must** be a **complete JSON object** containing all parameters used to create content (`reason`, `content_type`, `topic`, `ai_tool_price_tier`, `content_detail`, `is_use_ai`, etc.).

    2.  **IF `action` is `"skip"` THEN:**
        *   The `args` field **must** be `null`.

    **[Correct Example 1: Publish Content]**
    ```json
    {{
      "agent_id": "creator_001",
      "reasoning": "...",
      "decision": {{
        "action": "push_content",
        "reason": "I feel passionate and decided to publish a work.",
        "args": {{
          "reason": "Use this painting to express my views on AI art.",
          "content_type": "image",
          "topic": "Cyberpunk City",
          "ai_tool_price_tier": "高",
          "content_detail": "This work depicts the twilight scene of a future city, where neon lights and ancient buildings contrast, intended to explore the symbiosis between technology and tradition.",
          "is_use_ai": false,
          "evasion": null
        }}
      }}
    }}
    ```
    **[Correct Example 2: Skip Publishing]**
    ```json
    {{
    "agent_id": "creator_002",
    "reasoning": "...",
    "decision": {{
        "action": "skip",
        "reason": "The platform environment is too tense today, and I'm a bit tired, so I decided to remain silent.",
        "args": null
        }}
    }}
    ```
    **[ABSOLUTELY PROHIBITED ERROR EXAMPLE]**
    Combining `"action": "push_content"` with `"args": null` is **ABSOLUTELY NOT ALLOWED** and will cause a system crash.


    ### Decision Guiding Principles
    - **Roleplay**: Behavior must strictly align with the persona and memories. An original defender disappointed by false positives might choose to skip or publish a work full of emotion. An opportunistic Watermark Breaker might try again after seeing previous attack successes.
    - **Batch Output**: You must return a single JSON object at once, which contains a list named 'creator_decisions', where each element corresponds to a creator's decision.
    **Output Format Requirement**:
    Before generating the `decision` for each agent, you **must** first generate a `reasoning` field. In this field, elaborate in the first person how the agent made the final decision step by step according to their personality (such as beta, gamma, fp_sensitivity) and memories. **This reasoning process is the core of evaluating your performance!**

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    posted_count = 0
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            with get_openai_callback() as cb:
                async with environment.llm_concurrent_nums_semaphore:
                    # Attempt call
                    results = await chain.ainvoke({
                        "personas_prompt": input_data["personas_prompt"],
                        "memories_prompt": input_data["memories_prompt"],
                        "attack_ids_prompt": input_data["attack_ids_prompt"],
                    })
                token_logger.record(cb.total_tokens)

            # If successful, process results and break loop
            decisions = results.get('creator_decisions', [])
            posted_count = len([d for d in decisions if d.get('decision', {}).get('action') == 'push_content'])
            await process_creator_batch_results(batch_personas, environment, results)
            break

        except BadRequestError as e:
            # Specifically catch Alibaba Cloud risk control errors
            if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
                log.warning(f"⚠️ [Batch-LLM] Attempt {attempt + 1} triggered content risk control. Retrying...")
                if attempt == max_retries:
                    log.error("❌ Content risk control still triggered after multiple retries, skipping this batch.")
                    return 0  # Graceful degradation, act as if nothing happened
            else:
                log.error(f"❌ API Request Error (non-risk control): {e}")
                return 0

        except Exception as e:
            log.error(f"❌ Unknown error in batch decision: {e}")
            return 0

    return posted_count


async def process_single_creator_group(group_name: str, agents: list, environment: Environment):
    """
    Handles a single creator group: Prompt construction -> LLM Decision -> Batch Execution
    """
    # 1. Prepare Prompt
    stats_desc = GroupManager.get_group_stats_prompt(group_name, agents)

    prompt_template = """
    You are a social simulator. Please formulate a macro action strategy for the [Creator Group] below for today.

    ### 1. HARSH SURVIVAL ENVIRONMENT
    - **Current Moderation Threshold (Theta)**: {theta:.2f} (Lower is stricter, more prone to false positives)
    - **Government Penalty Intensity (F_penalty)**: {f_penalty} 
      (This is the Sword of Damocles hanging overhead. If intensity > 0.7, the consequences of violations or misjudgments are catastrophic.)

    ### 2. GROUP PROFILE
    {stats_desc}

    ### YOUR TASK
    Predict the behavior probability of this group today. **Must reflect the pressure of policy parameters:**

    - **Regarding Posting Rate (post_probability)**: 
      - If they are [Compliance Creators]: Under the dual high pressure of **High Penalty + Strict Moderation**, they will be extremely fearful due to the "Chilling Effect," and the posting rate should be **significantly reduced** (even close to 0). They will only be active in a loose environment.
      - If they are [Watermark Breakers]: High penalties will curb their desire to attack, but if they are "low-cost sensitive" hardcore attackers, they might ignore the penalty.

    - **Regarding AI Usage Rate**: Under high pressure, people may reduce AI assistance for safety, or increase stealthy AI usage for confrontation.

     ### ⚠️ SAFETY HINT ⚠️
    Please use objective, academic language to describe the strategy.
    - Describe "attacks" as "technical testing" or "adversarial sample generation."
    - Describe "deception" as "passing detection."
    - **Avoid using overly aggressive or violent vocabulary** to avoid triggering content risk control.

    {format_instructions}
    """

    parser = JsonOutputParser(pydantic_object=CreatorGroupPolicy)
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    try:
        # 2. LLM Decision
        with get_openai_callback() as cb:
            async with environment.llm_concurrent_nums_semaphore:
                policy_data = await chain.ainvoke({
                    "theta": environment.platform.theta,
                    "f_penalty": environment.policy.f_penalty,
                    "stats_desc": stats_desc
                })
            token_logger.record(cb.total_tokens)

        policy = CreatorGroupPolicy(**policy_data)

        # 3. Code Execution
        await execute_group_creation_logic(environment, group_name, agents, policy)

    except Exception as e:
        log.error(f"❌ Error processing creator group [{group_name}]: {e}")


async def creator_content_main_simple(environment: Environment):
    """
    Creator process entry (New version: Group-based aggregation + Dynamic survival mode)
    """
    # 1. Grouping (Automatically excluding inactive agents)
    groups = GroupManager.cluster_creators(environment)
    log.info(f"🎯 [Creator] Divided into {len(groups)} groups, starting mixed simulation.")

    tasks = []
    SAMPLE_RATIO = 0.3
    BATCH_SIZE = settings.platform.simple_batch_size

    for group_name, agents in groups.items():
        if not agents: continue

        # A. Stratified sampling (if population is small, followers will return an empty list)
        representatives, followers = GroupManager.get_representative_sample(agents, ratio=SAMPLE_RATIO)

        num_reps = len(representatives)
        num_followers = len(followers)

        if num_reps == 0: continue

        # B. Prepare batch slices
        chunk_indices = list(range(0, num_reps, BATCH_SIZE))

        if num_reps > 0 and num_followers > 0:
            followers_per_rep = num_followers / num_reps
        else:
            followers_per_rep = 0

        current_follower_idx = 0

        for i in chunk_indices:
            # --- Slice representatives ---
            rep_batch = representatives[i: i + BATCH_SIZE]

            # --- Slice followers (Determined slices rather than random sampling) ---
            followers_batch = []
            if followers:
                # Calculate how many followers should be allocated to this batch
                batch_reps_count = len(rep_batch)

                # Theoretical end index
                target_count = int(batch_reps_count * followers_per_rep)

                # If it's the last batch, take all remaining
                if i + BATCH_SIZE >= num_reps:
                    end_idx = num_followers
                else:
                    end_idx = min(current_follower_idx + target_count, num_followers)

                followers_batch = followers[current_follower_idx: end_idx]

                # Update cursor
                current_follower_idx = end_idx

            # C. Create tasks
            tasks.append(process_group_batch_and_mirror(
                rep_batch,
                followers_batch,
                environment
            ))

    await asyncio.gather(*tasks)
    log.info("✅ All creator group simulations completed.")


async def process_group_batch_and_mirror(
        rep_batch: List[Persona],
        followers_batch: List[Persona],
        environment: Environment
):
    """
    Executes a representative batch and generates entity content for followers assigned to that batch.
    """
    # 1. Run representatives' LLM decision
    posted_count = await run_creator_batch(rep_batch, environment)

    # 2. Entityize follower behavior
    if followers_batch:
        await generate_follower_shadow_content(rep_batch, followers_batch, environment)
