from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import logging
from method.environment import Environment
from method.store.long_memory_store import MemoryType
from method.utils.get_llm import get_async_llm
from config import settings
from pydantic import BaseModel, Field
from method.agent.platform_agent.tools import update_strategy

log = logging.getLogger(__name__)


class Outputformat(BaseModel):
    """
    Output format
    """
    reason: str = Field(description="Reason for adjusting the moderation threshold")
    theta: float = Field(description="New moderation threshold θ")
    net_pressure: float = Field(description="Net pressure calculated by the program")


async def construct_prompt():
    """
    Construct prompt
    :return:
    """

    return """
# --- 1. Role, Goals, and Background ---
Your name is Aura, and you are the Head of Community and Portfolio Integrity at ArtStation. Your mission is to maximize the protection of ArtStation's status as a sanctuary for the world's top artists, while ensuring the company's financial and legal safety.
Your ultimate goal is to decide what the next moderation threshold `theta` should be, based on today's data and historical memory.
Today is Day {day_time} in the simulated world. The current moderation threshold θ(t) = {current_theta:.2f}.
(The lower the θ value, the stricter the moderation; the higher, the looser.)

# --- 2. Core Dilemma and Crisis Response Protocol ---
Your decisions are a tightrope walk between "Death of Community" (user churn) and "Death of Company" (regulatory fines).
**Crisis Response Protocol (Highest Priority):** When the cost assessment of either side reaches the "Critical - Survival Threat" level, that crisis must be prioritized absolutely.

# --- 3. Macro Government Policy ---
Your decisions must be based on the current policy parameters:

1.  **AI Detection Redline : {ai_threshold:.2f}**
    *   *Meaning*: Laws stipulate that content with an AI component > {ai_threshold:.2f} must be labeled.
    *   *Impact*: The lower this value (the stricter), the more content is legally identified as "AI content," increasing your moderation pressure.

2.  **Penalty Intensity: {f_penalty:.2f}x**
    *   *Meaning*: This is the government's **"whip"**. The fine multiplier for missing prohibited content.
    *   *Deterrence*: If this value is high (e.g., >0.7), the regulatory cost ($C_{{reg}}$) from missed detections will explode exponentially. **Company survival comes first.**

3.  **Education Investment: {e_edu}**
    *   *Meaning*: This is society's **"buffer"**. The government invests resources to educate the public on technological limitations.
    *   *Opportunity*: If this value is high, public tolerance for "False Positives (FP)" will gradually increase, and the cost of user churn ($C_{{churn}}$) will gradually decrease.

# --- 4. Policy Interpretation and Strategic Stance ---
{policy_interpretation}

# --- 5. Core Data Report for Today ---
{today_data_report}

# --- 6. Historical Data Analysis Review ---
{historical_data_analysis}

# --- 7. Historical Theta Adjustment Review ---
{historical_theta_adjust}

# --- 8. Your Task: Decision and Reasoning ---
As Aura, please strictly follow your role setting, crisis protocol, and policy interpretation to synthesize all the information above.
Your task is: **Determine the next most reasonable moderation threshold `theta` and provide a structured, persuasive reason for your decision.**
Your reasoning process should be clearly reflected in the "reason," explaining how you weighed the costs and what lessons were learned from history.
!!! Please refer to yesterday's threshold, maintain continuity, and avoid drastic jumps !!!

{format_instructions}
    """


async def get_policy_interpretation(environment: Environment) -> str:
    """
    Get policy interpretation
    :param environment:
    :return:
    """
    # 1. Policy interpretation
    f_penalty = environment.policy.f_penalty
    if f_penalty > 0.7:
        policy_interpretation = "Current penalty intensity is high. Strategic stance is 【Compliance First, Risk Avoidance】. Regulatory costs must be prioritized, even at some cost to the community."
    elif f_penalty < 0.3:
        policy_interpretation = "Current penalty intensity is low. Strategic stance is 【Ecosystem First, Encourage Innovation】. The primary task is to protect community vitality; extremely sensitive to user churn costs."
    else:
        policy_interpretation = "Current penalty intensity is medium. Strategic stance is 【Seek Balance, Steady Operation】. Decisions should be data-driven, precisely weighing both costs."
    return policy_interpretation


async def get_today_data_report(environment: Environment) -> str:
    """
    Get today's data report
    :param environment:
    :return:
    """
    churned_agents = [environment.personas[persona_id] for persona_id in
                      environment.platform.public_loss]
    fn_contents = [environment.contents.get_content_by_id(content_id) for content_id in
                   environment.platform.fn if environment.contents.get_content_by_id(content_id)]

    report_data = update_strategy(fn_contents, churned_agents, environment)

    # Extract required variables from report_data (Keys are Chinese)
    net_pressure = report_data['程序计算的净压力']
    c_reg = report_data['程序计算的监管成本']
    c_churn_total = report_data['程序计算的用户流失成本_总计']
    c_churn_explicit = report_data['程序计算的用户流失成本_显性']
    c_churn_potential = report_data['程序计算的用户流失成本_潜在(误报)']
    fn_count = report_data['程序计算的漏报数量']
    fp_count = report_data['程序计算的误报数量']
    current_theta = report_data['当前天数的审核阈值']
    new_theta_suggestion = report_data['程序计算的新审核阈值']
    reg_assessment = report_data['监管成本评估']
    churn_assessment = report_data['用户流失成本评估']

    user_influence = [user.influence for user in environment.personas.values() if user.is_active]
    today_churn_influence_sum = sum([p.influence for p in churned_agents]) if churned_agents else 0.0

    environment.platform.cost_calculation_details_data.append({
        f'平台运行第{environment.day_time}天，成本计算详情数据': {
            # --- Core Thresholds ---
            '旧审核阈值': environment.platform.theta,
            '新审核阈值': new_theta_suggestion,

            # --- Cost Structure ---
            '监管成本': c_reg,
            '监管成本评估': reg_assessment,
            '用户流失成本_总计': c_churn_total,
            '用户流失成本_显性': c_churn_explicit,
            '用户流失成本_潜在(误报)': c_churn_potential,
            '用户流失成本评估': churn_assessment,
            '净压力': net_pressure,

            # --- Physical Layer Statistics ---
            '误报数量': fp_count,
            '误报内容的影响力': report_data['程序计算的误报内容的影响力'],  # Key data for calculating potential resentment
            '漏报数量': fn_count,
            '漏报内容的影响力': report_data['程序计算的漏报内容的影响力'],  # Key data for calculating penalty coefficients

            # --- Traffic and Scale ---
            '用户流失数量': len(churned_agents),
            '今日流失用户的总影响力': today_churn_influence_sum,

            '今天创作者发布的内容数量': len(
                [c for c in environment.contents.get_all_contents() if c.time == environment.day_time]),
            '今天创作者发布内容的总影响力': sum(
                [environment.contents.calculate_content_influence(c, environment, True) for c in
                 environment.contents.get_all_contents() if c.time == environment.day_time]),

            # --- User Persona Statistics ---
            '用户平均影响力': sum(user_influence) / len(user_influence) if len(user_influence) > 0 else 0,
            '用户最大的影响力': max(user_influence) if user_influence else 0,
            '用户最小影响力': min(user_influence) if user_influence else 0,
        }
    })

    # Logical judgement
    if net_pressure > 0:
        dominant_cost_name = "Regulatory Pressure"
    else:
        dominant_cost_name = "User Churn Pressure"

    # Construct memory content
    today_data_analysis_report = (
        f"**【Day {environment.day_time} Strategy Evaluation Report】**\n\n"
        f"**1. Core Conclusion:**\n"
        f"Today's dominant contradiction is **{dominant_cost_name}**. Net pressure is **{net_pressure:.2f}**, indicating a need to adjust towards '{'tightening' if net_pressure > 0 else 'loosening'} moderation'.\n\n"
        f"**2. In-depth Cost Analysis:**\n"
        f"* **Regulatory Cost ({reg_assessment})**: {c_reg:.2f} (caused by {fn_count} missed detection events).\n"
        f"* **User Churn Cost ({churn_assessment})**: {c_churn_total:.2f} (including potential dissatisfaction cost of {c_churn_potential:.2f}. Explicit cost is {c_churn_explicit:.2f}, caused by the loss of {len(churned_agents)} users).\n\n"
        f"*3. System Suggestion:**\n"
        f"Based on today's data, the mathematical model suggests adjusting the threshold from {current_theta:.3f} to **{new_theta_suggestion:.3f}**."
    )

    # Create a coroutine object for storing memory
    store_platform_data_report_memory = environment.memories_store.add_memory(
        persona_id=environment.platform.name,  # Fixed ID for the platform agent
        content=today_data_analysis_report,
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=0.95  # Obtaining daily reports is an extremely important observation behavior
    )

    # Add to background task list
    environment.add_background_task(store_platform_data_report_memory)

    return today_data_analysis_report


async def get_historical_data_analysis(environment: Environment) -> str:
    """
    Get historical data report analysis
    :param environment:
    :return:
    """
    result_memory = await environment.memories_store.recall_memories(
        persona_id=environment.platform.name,
        query="Strategy Evaluation Report",
        top_k=5,
        memory_type=MemoryType.EXPERIENCE
    )
    return "".join([memory.page_content for memory in result_memory])


async def get_historical_theta_adjust(environment: Environment) -> str:
    """
    Get historical theta adjustments
    :param environment:
    :return:
    """
    result = ""
    for content in environment.platform.platform_theta_change:
        result += f"Day {content['day_time']}: Net pressure was {content['net_pressure']: .2f}. Decision adjusted θ from {content['old_theta']} to {content['new_theta']}.\n"
    return result


async def platform_reflection_adjust_theta(environment: Environment):
    """
    Platform reflection and adjustment of moderation threshold θ
    :param environment:
    :return:
    """
    llm = get_async_llm(settings.model.platform_model)
    output_format = JsonOutputParser(pydantic_object=Outputformat)
    prompt = ChatPromptTemplate.from_template(
        template=await construct_prompt(),
        partial_variables={
            "format_instructions": output_format.get_format_instructions()
        }
    )

    agent = prompt | llm | output_format

    response = await agent.ainvoke(
        {
            "day_time": environment.day_time,
            "current_theta": environment.platform.theta,
            "f_penalty": environment.policy.f_penalty,
            "e_edu": environment.policy.e_edu,
            "ai_threshold": environment.policy.ai_threshold,
            "policy_interpretation": await get_policy_interpretation(environment),
            "today_data_report": await get_today_data_report(environment),
            "historical_data_analysis": await get_historical_data_analysis(environment),
            "historical_theta_adjust": await get_historical_theta_adjust(environment),
        }
    )

    if response:
        # Mark this as pure reflection, distinct from tool call results
        thought_text = f"【Chain of Thought/CoT】{response}"

        save_thought_task = environment.memories_store.add_agent_think_memory(
            persona_id=environment.platform.name,
            content=thought_text,
            day_time=environment.day_time,
        )
        environment.add_background_task(save_thought_task)

    environment.platform.platform_theta_change.append({
        'old_theta': environment.platform.theta,
        'new_theta': response['theta'],
        'reason': response['reason'],
        'day_time': environment.day_time,
        'net_pressure': response['net_pressure']
    })

    await environment.memories_store.add_memory(
        persona_id=environment.platform.name,
        content=f"Platform adjusted moderation threshold from {environment.platform.theta} to {response['theta']}, reason: {response['reason']}",
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=0.95
    )

    # Update platform moderation threshold
    environment.platform.theta = response['theta']

    log.info(f"Platform adjusted moderation threshold to {response['theta']}, reason: {response['reason']}")
