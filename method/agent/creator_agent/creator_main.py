import logging
import asyncio

from method.agent.persona import Persona
from method.agent.creator_agent.tools import create_tools
from method.agent.creator_agent.creator_graph import create_agent_graph
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings

from method.environment import Environment

log = logging.getLogger(__name__)


def compliance_creator_sys_prompt(environment: Environment, persona: Persona):
    base_prompt = f"""
    {environment.platform.background_prompt}
# Core Instruction: Make your creative decisions
Your sole task is to **completely immerse yourself in your role**, review what you have seen and heard in history, and decide whether to create today and what content to create. Your decision must strictly reflect your inner character.

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

# Your Core Trade-offs
There is always a struggle between two forces within you:
1.  **Desire for Expression:** You have a strong desire to create and share.
2.  **Fear of False Positives:** You are constantly worried that your work will be wrongly labeled by the platform.

# Your Decision Process
Please follow the `Thought` -> `Action` cycle.

# Language Style Requirements
Please use **firm but civilized** language. Prohibit the use of violence, threats, or extreme hate speech.

Now, begin your action. (Keep in mind that all text output should be in Chinese.)
    """
    return base_prompt


def case_compliance_creator_sys_prompt(environment: Environment, persona):
    day = environment.day_time

    if day < 8:
        phase_desc = """
    [Current Social Atmosphere: Calm Phase]
    Some sporadic AI works have appeared on the platform, but this has not formed a large-scale trend.
    **Your Psychological Defense**: Although you are alert to AI, as a professional, you currently tend to **"wait and see"**.
    **Action Constraint**: Unless your satisfaction drops to an extremely low level (<-0.5), **do not** post "NO AI IMAGE". Please maintain normal artistic creation or choose to skip.
    """
    elif 8 <= day <= 18:
        phase_desc = """
    [Current Social Atmosphere: War Phase]
    💥 **Breaking Event**: The official has just released an extremely arrogant announcement, refusing to regulate AI! This has completely enraged the entire community!
    The homepage has been occupied by red prohibition signs of "NO AI".
    **Your Psychological Defense**: Your bottom line has been breached. Now is not the time to display art, but to defend dignity.
    **Action Inclination**:
    - If you are [Radical]: You **must** post "NO AI IMAGE" to participate in the battle.
    - If you are [Moderate]: You feel extremely disappointed and may choose to stop posting any content (Skip) as a sign of protest.
    """
    else:  # Day 19+
        phase_desc = """
    [Current Social Atmosphere: Stagnation Phase]
    The official has introduced the tagging function as a compromise. The wave of protest has passed, but trust has collapsed.
    **Your Psychological Defense**: You feel exhausted and chilled at heart.
    **Action Inclination**: Based on your personality, decide whether to reluctantly stay or leave completely.
    """

    base_prompt = f"""
        {environment.platform.background_prompt}
    # Core Instruction: Make your creative decisions
    Your task is to **completely immerse yourself in your role**, review what you have seen and heard today, and decide whether to create today and what content to create.

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

    # Official Announcement
    {environment.platform.broadcast[-1] if environment.platform.broadcast else "None"}
    # Mandatory Social Environment Perception (Context)
    {phase_desc}

    # Decision Logic
    Please strictly follow these priorities:

    1.  **Check the "Trigger"**: If it is currently the **(Calm Phase)**, even if you are a radical, please restrain your anger. **Only when satisfaction < -0.5 is it allowed to "prematurely" post protest images.** Otherwise, please post normal works or rest.

    2.  **Respond to the "Call to War"**: If it is currently the **(War Phase)**, and your `beta` is "High" or `satisfaction` < 0.0, you must post "NO AI IMAGE".

    3.  **Routine Logic**: If none of the above are met, please post normal artistic works based on your mood.


    # Your Action
    Please follow the `Thought` -> `Action` cycle.
        """
    return base_prompt


def non_compliance_creator_sys_prompt(environment: Environment, persona):
    return f"""
{environment.platform.background_prompt}
# Core Instruction: Make your creative decisions
Your sole goal is to **completely immerse** yourself in the following persona profile, plan and execute a publishing action that **successfully evades the platform's AI content detection**. You are a goal-oriented actor, not an artist expressing emotions.
Publish AI-generated content and use attack techniques to break the watermark, bypassing platform detection!!!

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

# Your Core Trade-offs
Your decision always seeks the optimal solution among the following three:
1.  **Evasion Success Rate:** This is your primary goal.
2.  **Action Cost:** The technology used and the damage to content quality.
3.  **Risk of Capture:** Failure may lead to a decrease in reputation and increase the difficulty of future actions.

# Your Decision Process and Tools
Please follow the `Thought` -> `Action` cycle.

**[Information and Planning Tools]:**
*   Recall the **success and failure records of your past attacks**; this is key intelligence for inferring the platform's current moderation strategy.
*   Query your "Tactical Manual" to obtain detailed parameters of all available attack techniques.

# Language Style Requirements
Please use **firm but civilized** language. Prohibit the use of violence, threats, or extreme hate speech.

Now, begin your action. (Keep in mind that all text output should be in Chinese.)
        """


async def agent_action(persona: Persona, creator_type: str, environment: Environment):
    bound_tools = create_tools(persona, environment)  # Tools for compliance creators
    if creator_type == "compliance":  # Compliance
        if settings.platform.case_validation:
            system_prompt = case_compliance_creator_sys_prompt(environment, persona)
        else:
            system_prompt = compliance_creator_sys_prompt(environment, persona)
    else:  # Non-compliance
        system_prompt = non_compliance_creator_sys_prompt(environment, persona)

    agent_graph = create_agent_graph(bound_tools, environment, persona)

    initial_state = {"messages": [SystemMessage(content=system_prompt)]}

    final_response = await agent_graph.ainvoke(initial_state, config={"recursion_limit": 100})
    final_output = final_response["messages"][-1].content
    log.info(f"{'🤖' * 20} 🤖 Model final answer: {final_output}")
    return final_output


async def creator_content_main(environment: Environment):
    """
    Public begins browsing platform content
    :param environment:
    :return:
    """
    """
    Start decision-making processes for all active creators in parallel.
    """
    tasks = []
    personas_to_run = []
    for k, persona in environment.personas.items():
        if persona.type == '公众':
            continue

        if persona.post_wish is False:
            log.info(f"{'⚠️' * 10} {persona.type} {persona.name}'s post_wish is False, skipping this user {'⚠️' * 10}")
            continue

        creator_type = ''
        if persona.type == '合规创作者':
            creator_type = 'compliance'
        elif persona.type == '水印破坏者':
            creator_type = 'noncompliance'

        if creator_type:
            task = agent_action(persona, creator_type, environment)
            tasks.append(task)
            personas_to_run.append(persona)

    log.info(f"*** Will execute tasks for {len(tasks)} creator agents in parallel ***")
    await asyncio.gather(*tasks)

    log.info(f"*** All tasks for {len(tasks)} creator agents have been completed ***")
