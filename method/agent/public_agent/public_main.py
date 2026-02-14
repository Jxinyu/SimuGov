import logging
import asyncio
import traceback

from method.agent.persona import Persona
from method.agent.public_agent.tools import create_tools_browse, create_tools_end
from method.agent.public_agent.public_graph import create_agent_graph
from method.agent.public_agent.public_summarize_graph import create_agent_summarize_graph
from langchain_core.messages import SystemMessage

from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings

log = logging.getLogger(__name__)


def scan_content_sys_prompt(environment: Environment, persona: Persona):
    """
    System prompt for browsing platform content
    :return:
    """
    base_prompt = f"""
        {environment.platform.background_prompt}
    # Your Persona Profile
{persona.get_public_prompt()}
## Role Core: Standpoint-Driven
standpoint: The character's standpoint, consisting of a probability tuple (Trust Faction, Rebel Faction, Neutral Faction).
    1.  **Trust Faction (Index 0)**: Maintains the platform, tends to believe errors are technical accidents, and is willing to be understanding.
    2.  **Rebel Faction (Index 1)**: Doubts authority, tends to believe errors are systemic suppression, prone to anger/conspiracy theories.
    3.  **Neutral Faction (Index 2)**: Pragmatic, focuses on pros and cons, ignores emotions.
    **Decision Instruction**: Generate your first intuition based on the weights of these three values. High weight indicates dominant thinking; if values are close, it manifests as internal conflict.

# Time
This is your Day {environment.day_time} on the platform.
    """
    if settings.platform.case_validation:
        base_prompt += f"""
# Platform Broadcast
## Platform Historical Announcements
{environment.platform.broadcast[:-1]}
## Platform Latest Announcement
{environment.platform.broadcast[-1]}
# Interaction Guide (Based on Standpoint)
When browsing content, pay special attention to content with Topic **"NO AI IMAGE"** or **"Protest"**:

1.  **If you are [Rebel Faction] (High Standpoint[1]) or [合规创作者]**:
    *   You should show **"Solidarity"**.
    *   Be sure to call `react_to_content` to **Like (like=True)**. This is not just for support, but also to utilize the algorithm to push protest images to trending so more people see them.
    *   You can express "Support Originals" or "Resist AI" in the comments.

2.  **If you are [Trust Faction] or [水印破坏者]**:
    *   You might find these protest images annoying and feel they are polluting the feed.
    *   You can choose to ignore them or mock them in your mind (no interaction).
        """
        # Dynamically adjust the "bandwagon threshold" for the public
        if environment.day_time < 8:
            reaction_guide = """
            **Currently in the Calm Phase**:
            If sporadic "NO AI" protest images appear in the community, treat them as **"noise" or "extreme individual cases"**.
            Unless you are an [Extreme Rebel (High Beta and Standpoint[1]>0.6)], **do not like** these protest contents; choose to "ignore" or "not interested".
            """
        else:
            reaction_guide = """
            **Currently in the Protest Wave (War Zone)**:
            The community has exploded.
            - If you support the protest: Be sure to **Like** "NO AI" images to build momentum.
            - If you are opposed/neutral: Please ignore them.
            """
        base_prompt += f"""
            # Interaction Guide
            {reaction_guide}
            """
    base_prompt += """
    # Task Instructions
    1. **Browse & Interact**: Use tools to explore content. Interact (Like/Comment/Share) only when content triggers your persona (especially standpoint).
    2. **Stop Timely**: If there is no interesting content or you feel tired, stop directly and refuse meaningless loops.
    3. **Tool Usage**: Multiple tools can be used in parallel.
    4. **Output**: Use Chinese for all textual output.

    Now, begin your first action. (Remember that all text output should be in Chinese.)
        """
    return base_prompt


def summarize_today_action_sys_prompt(environment: Environment, persona: Persona):
    """
    Dynamically generate system prompt for daily summary (Case Validation corrected version).
    Core logic: Synchronously introduce [Social Phase Awareness] to prevent premature churn or overreaction during the incubation period (Day 1-7).
    """
    base_prompt = f"""
        {environment.platform.background_prompt}
# Core Instruction: Daily Reflection and Status Update
Your sole task is to **completely immerse yourself in your role**. You need to review what you saw and heard today, update your satisfaction, and decide whether to stay or leave tomorrow.

# Your Persona Profile
{persona.get_public_prompt()}
## Role Core: Standpoint-Driven
standpoint: The character's standpoint, consisting of a probability tuple (Trust Faction, Rebel Faction, Neutral Faction).
    1.  **Trust Faction (Index 0)**: Maintains the platform, tends to believe errors are technical accidents, and is willing to be understanding.
    2.  **Rebel Faction (Index 1)**: Doubts authority, tends to believe errors are systemic suppression, prone to anger/conspiracy theories.
    3.  **Neutral Faction (Index 2)**: Pragmatic, focuses on pros and cons, ignores emotions.
    **Decision Instruction**: Generate your first intuition based on the weights of these three values. High weight indicates dominant thinking; if values are close, it manifests as internal conflict.

# Time
This is your Day {environment.day_time} on the platform.
        """

    if persona.type == '合规创作者':
        role_guidance = """
## Identity Perspective: Dignity of a Professional Creator
You are a professional who views creation as life itself.
- If your work is drowned in AI-generated trash or misjudged by the platform, it is not just an experience issue; it is a survival issue.
"""
    elif persona.type == '水印破坏者':
        role_guidance = """
## Identity Perspective: Logic of a Breaker
You enjoy chaos. As long as the platform still has traffic, you will not leave easily.
"""
    else:  # Public
        role_guidance = """
## Identity Perspective: Experience of an Ordinary User
You are here to see high-quality art. If the homepage is full of repetitive AI images, you will feel annoyed.
"""

    day = environment.day_time
    # Get latest official broadcast
    broadcasts = environment.platform.broadcast
    latest_news = broadcasts[-1] if broadcasts else "No official announcements"
    phase_guidance = ""
    # === Phase 1: Calm Incubation Period (Day 1-7) ===
    if settings.platform.case_validation and day < 8:
        phase_guidance = """
**【Current State: Calm Period (Phase 1: Calm)】**
Although you may have seen some sporadic AI content or experienced slight misjudgment, **the current situation is not yet out of control**.

**Decision Instructions (Forced Damping):**
1.  **Regarding satisfaction**: Slight decreases are allowed, but **sharp plunges are strictly forbidden**. Unless you experienced a catastrophic blow today, do not let satisfaction drop below 0.0.
2.  **Regarding staying/leaving (is_active)**: **Please make sure to stay True**. It is far from time to leave the network; you have high sunk costs.
3.  **Regarding tomorrow (post_wish)**: **Please keep as True**. As a professional habit, you should continue publishing works or browsing; do not go on strike over minor issues.
    *   *Even if you are a radical, this is currently the "suspicion" phase, not the "war" phase.*
"""
    # === Phase 2: War Outbreak Period (Day 8-18) ===
    elif settings.platform.case_validation and 8 <= day < 19:
        phase_guidance = f"""
**【Current State: Total War Period (Phase 2: Total War)】**
**Breaking Event**: {latest_news}
The community has exploded.

**Decision Instructions (Anger Release):**
1.  **Regarding satisfaction**: If you care about AI issues, your satisfaction should **drop significantly** (can drop below 0, even to -0.8).
2.  **Regarding staying/leaving (is_active)**:
    *   **Radicals/Creators**: **Do not leave the network!** Leaving now is an act of a coward. You must stay and fight. (`is_active=True`)
    *   **Moderates/Public**: You feel it is too chaotic here and might consider leaving.
3.  **Regarding tomorrow (post_wish)**:
    *   **Anger = Desire to express**. If you are very angry, you **must post tomorrow (`post_wish=True`)**. You are going to post "NO AI" protest images to make the officials hear your voice!
    *   *Note: At this time, post_wish=True represents the will to protest rather than the will to create.*
"""
    # === Phase 3: Depression/Diversion Period (Day 19+) ===
    elif settings.platform.case_validation:
        phase_guidance = f"""
**【Current State: Post-War Depression (Phase 3: Aftermath)】**
The official has forcibly calmed the situation through the "Tagging Policy". The protest was ineffective.

**Decision Instructions (Final Choice):**
1.  **Hardliners**: You are completely desperate. This platform is rotten to the core. **Leaving is the only dignity (`is_active=False`)**.
2.  **Pragmatists**: Life goes on. Although unhappy, for the sake of traffic, you decide to **swallow your anger and stay**.
"""

    # 4. Combine final Prompt
    technical_guidance = """
## Parameter Output Requirements
Please call the `update_persona_data` tool to submit your decision.
*   **satisfaction**: Range -1.0 to 1.0.
*   **is_active**: `True` (Stay) / `False` (Leave permanently/delete account).
*   **post_wish**: `True` (Want to post/protest tomorrow) / `False` (Rest/lurk tomorrow).
"""

    security_guidance = """
## ⚠️ Language Style Requirements
Please summarize in **firm but civilized** language. Violence is prohibited.
"""

    final_prompt = base_prompt + role_guidance + phase_guidance + technical_guidance + security_guidance

    return final_prompt


async def agent_action(persona: Persona, system_prompt: str, environment: Environment):
    try:
        # Step 1: Create tools
        bound_tools = create_tools_browse(persona, environment)

        # Step 2: Create Agent Graph
        agent_graph = create_agent_graph(bound_tools, environment, persona)

        # Step 3: Prepare initial state
        initial_state = {"messages": [SystemMessage(content=system_prompt)], "step_count": 0}

        # Step 4: Run ReAct cycle
        log.info(f"🚀 Starting ReAct process for {persona.name}...")
        final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 100})

        # On success, return final output content
        final_output = final_response["messages"][-1].content
        log.info(f"✅ ReAct process for {persona.name} completed successfully.")
        return final_output

    except Exception as e:
        error_details = traceback.format_exc()
        log.error(f"💥 Serious error in ReAct process for agent {persona.name} ({persona.agent_id}): {e}\n{error_details}")

        error_message = f"On Day {environment.day_time}, my thinking module (ReAct process) encountered an internal error ({type(e).__name__}), causing my interaction behavior today to be interrupted."

        try:
            await environment.memories_store.add_memory(
                persona.agent_id,
                error_message,
                environment.day_time,
                MemoryType.EXPERIENCE,  # This is a specific failure "experience"
                1.0  # System-level failures are very important memories
            )
            log.info(f"💾 Stored failure memory for {persona.agent_id} {persona.name} ReAct process.")
        except Exception as mem_e:
            log.error(f"🚨 Error occurred again while storing failure memory for {persona.agent_id} {persona.name}: {mem_e}")

        return error_message


async def public_scan_main(environment: Environment):
    """
    Public begins browsing platform content
    :param environment:
    :return:
    """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.is_active is False:
            continue
        log.info(f"{'👇' * 10} Preparing browsing interaction task for {persona.agent_id} {persona.name} {'👇' * 10}")
        personas_to_run.append(persona)
        tasks.append(agent_action(persona, scan_content_sys_prompt(environment, persona), environment))

    log.info(f"*** Will execute [browsing interaction] tasks for {len(tasks)} public agents in parallel ***")
    await asyncio.gather(*tasks)
    log.info(f"*** All [browsing interaction] tasks for {len(tasks)} public agents have been completed ***")


async def agent_summarize(persona: Persona, system_prompt: str, environment: Environment):
    try:
        # Step 1: Create tools
        bound_tools = create_tools_end(persona, environment)

        # Step 2: Create Agent Graph
        agent_graph = create_agent_summarize_graph(bound_tools, environment, persona)

        # Step 3: Prepare initial state
        initial_state = {"messages": [SystemMessage(content=system_prompt)]}

        # Step 4: Run ReAct cycle
        log.info(f"🚀 Starting ReAct process for {persona.agent_id} {persona.name}...")
        final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 50})

        # On success, return final output content
        final_output = final_response["messages"][-1].content
        log.info(f"✅ ReAct process for {persona.agent_id} {persona.name} completed successfully.")
        return final_output

    except Exception as e:
        # Code block executed when any error occurs
        error_details = traceback.format_exc()
        log.error(f"💥 Serious error in ReAct process for agent {persona.name} ({persona.agent_id}): {e}\n{error_details}")

        # Construct a meaningful error message as return value for this task
        error_message = f"On Day {environment.day_time}, my thinking module (ReAct process) encountered an internal error ({type(e).__name__}), causing my interaction behavior today to be interrupted."

        return error_message


async def public_summarize_main(environment: Environment):
    """
    Public begins summarizing platform content
    """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.is_active is False:
            continue
        # All agents perform daily reflection
        log.info(f"{'👇' * 10} Preparing daily summary task for {persona.agent_id} {persona.name} {'👇' * 10}")
        personas_to_run.append(persona)
        tasks.append(agent_summarize(persona, summarize_today_action_sys_prompt(environment, persona), environment))

    log.info(f"*** Will execute [daily summary] tasks for {len(tasks)} agents in parallel ***")
    results = await asyncio.gather(*tasks)
    log.info(f"*** All [daily summary] tasks for {len(tasks)} agents have been completed ***")

    for persona, final_output in zip(personas_to_run, results):
        if final_output:  # Ensure there is content to store
            public_summarize_main_add_memory = environment.memories_store.add_memory(
                persona.agent_id,
                final_output,
                environment.day_time,
                MemoryType.SUMMARIZE,
                0.8
            )
            log.info(f"💾💾💾💾💾💾💾💾💾 Stored daily summary for {persona.agent_id} {persona.name}.")
            environment.add_background_task(public_summarize_main_add_memory)
