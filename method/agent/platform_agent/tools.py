import traceback
import logging

import numpy as np
from langchain_core.tools import tool

from typing import List
from config import settings

from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


def calculate_regulatory_cost(fn_contents, environment):
    """
    Calculate the regulatory cost C_reg(t).
    Args:
        fn_contents (list): List of all missed detection (FN) content objects for the day.
        environment: Environment object.

    Returns:
        float: Calculated total regulatory cost.
        float: Total influence of missed detection content.
    """
    if not fn_contents:
        return 0.0, 0.0

    # Calculate the total influence of all missed detection content
    total_fn_content_influence = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True) for content in
         fn_contents]
    )

    c_reg = (0 + environment.policy.f_penalty) * total_fn_content_influence
    return c_reg, total_fn_content_influence


def calculate_churn_cost(churned_agents, environment) -> dict:
    """
   Calculate the user churn cost C_churn(t), including explicit churn cost and potential false positive dissatisfaction cost.

   Args:
       churned_agents (list): List of all agent objects churned on the current day.
       environment: Environment object used to obtain platform parameters and update cumulative dissatisfaction.

   Returns:
       dict: Calculated user churn cost details.
   """
    # --- Step 1: Explicit churn cost (Unit: User influence) ---
    explicit_churn_influence = sum([agent.influence for agent in churned_agents]) if churned_agents else 0.0

    # --- Step 2: Potential churn cost (Unit: Sum of raw influence of all FP content today) ---
    # 2.1 Get all content objects flagged as false positives (FP) today
    fp_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fp if
                   environment.contents.get_content_by_id(cid)]

    # 2.2 Calculate the real raw influence accumulated by these FP contents at the end of the day
    new_fp_influence_today = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True) for content in
         fp_contents]
    ) * 0.3

    # 2.3 Update the platform's cumulative dissatisfaction influence (it decays automatically daily)
    environment.platform.total_fp_creator_influence += new_fp_influence_today
    potential_churn_influence = environment.platform.total_fp_creator_influence

    # --- Step 3: Total "equivalent churn influence" ---
    total_churn_influence = explicit_churn_influence * settings.platform.mu + potential_churn_influence

    if total_churn_influence > 10:
        log.warning(f"User churn cost exceeds 10, please check!")

    return {
        'total': total_churn_influence,
        'explicit_churn': explicit_churn_influence,
        'potential_from_fp': potential_churn_influence,
        'fp_influence_today': new_fp_influence_today,
        'current_grievance': environment.platform.total_fp_creator_influence
    }


def update_strategy(fn_contents, churned_agents, environment):
    """
    Calculate cost, net pressure, and update the moderation threshold θ based on daily operational data.
    Args:
        fn_contents (list): List of all missed detection (FN) content today.
        churned_agents (list): List of all agents churned today.
        environment: Environment object.
    Returns:
        dict: A dictionary containing detailed cost, net pressure, and recommended new threshold.
    """
    # Step 1: Calculate core costs
    c_reg, total_fn_content_influence = calculate_regulatory_cost(fn_contents, environment)
    churn_cost_details = calculate_churn_cost(churned_agents, environment)
    c_churn_total = churn_cost_details['total']  # Get total user churn cost

    if c_churn_total > 10:
        log.warning(f"User churn cost exceeds 10, please check!")

    # Step 2: Calculate net pressure (using total user churn cost)
    net_pressure = c_reg - environment.platform.w * c_churn_total

    # Step 3: Calculate the adjustment amount Δθ
    # Use the tanh function to map net pressure to the [-1, 1] interval
    tanh_value = np.tanh(net_pressure / environment.platform.steep)
    delta_theta = -environment.platform.eta * tanh_value

    # Step 4: Update threshold θ(t+1) and use clip to ensure it stays within a valid range
    new_theta = environment.platform.theta + delta_theta
    new_theta = float(np.clip(new_theta, 0.05, 0.95))

    # Step 5: Return a more detailed report (Note: Keys are preserved in Chinese as requested)
    return {
        '当前天数的审核阈值': environment.platform.theta,
        '程序计算的新审核阈值': new_theta,
        '程序计算的监管成本': c_reg,
        '监管成本评估': environment.platform.get_severity_level(c_reg, 'reg'),
        '程序计算的用户流失成本_总计': c_churn_total,
        '程序计算的用户流失成本_显性': churn_cost_details['explicit_churn'],
        '程序计算的用户流失成本_潜在(误报)': churn_cost_details['potential_from_fp'],
        '用户流失成本评估': environment.platform.get_severity_level(c_churn_total, 'churn'),
        '程序计算的净压力': net_pressure,
        '程序计算的误报数量': len(environment.platform.fp),
        '程序计算的误报内容的影响力': churn_cost_details['fp_influence_today'],
        '程序计算的漏报数量': len(environment.platform.fn),
        '程序计算的漏报内容的影响力': total_fn_content_influence,
        '程序计算的用户流失数量': len(churned_agents),
    }


def create_platform_tools(environment: Environment) -> List[tool]:
    """
    Factory function: Create and return tools bound to a specific ContentStore instance.
    This is an implementation of dependency injection.

    Args:
        :param environment: Environment object.

    Returns:
        A list containing the configured tools.
    """

    @tool
    async def get_today_platform_data() -> dict | str:
        """
        Obtain data analysis reports, including core costs, net pressure, and preliminary system threshold adjustment suggestions. Preferred for decision-making.
        """
        try:
            # --- 1. Execute raw operations ---
            churned_agents = [environment.personas[persona_id] for persona_id in
                              environment.platform.public_loss]
            fn_contents = [environment.contents.get_content_by_id(content_id) for content_id in
                           environment.platform.fn if environment.contents.get_content_by_id(content_id)]

            report_data = update_strategy(fn_contents, churned_agents, environment)

            # Extract variables from report_data (Using Chinese keys as mapped above)
            net_pressure = report_data['程序计算的净压力']
            c_reg = report_data['程序计算的监管成本']
            c_churn_total = report_data['程序计算的用户流失成本_总计']
            c_churn_potential = report_data['程序计算的用户流失成本_潜在(误报)']
            fn_count = report_data['程序计算的漏报数量']
            fp_count = report_data['程序计算的误报数量']
            current_theta = report_data['当前天数的审核阈值']
            new_theta = report_data['程序计算的新审核阈值']
            reg_assessment = report_data['监管成本评估']
            churn_assessment = report_data['用户流失成本评估']

            # Logical judgment for report generation
            if net_pressure > 0:
                dominant_cost_name = "Regulatory Pressure"
            else:
                dominant_cost_name = "User Churn Pressure"

            # Construct memory content
            memory_content = (
                f"**Day {environment.day_time} Strategy Evaluation Report**\n\n"
                f"**1. Core Conclusion:**\n"
                f"Today, {dominant_cost_name} has become the dominant contradiction. Net pressure is {net_pressure:.2f}, indicating an imbalance between current moderation strategies and the market environment.\n\n"
                f"**2. In-depth Cost Analysis:**\n"
                f"* **Regulatory Cost ({reg_assessment})**: {c_reg:.2f}, primarily caused by {fn_count} missed detection events.\n"
                f"* **User Churn Cost ({churn_assessment})**: {c_churn_total:.2f}, where potential 'creator dissatisfaction' cost ({c_churn_potential:.2f}) is accumulating, with {fp_count} new false positive events today.\n\n"
                f"**3. Trends and Suggestions:**\n"
                f"To address today's {dominant_cost_name}, the system suggests adjusting the threshold from {current_theta:.3f} to {new_theta:.3f}.\n"
            )

            await environment.memories_store.add_memory(
                persona_id=environment.platform.name,  # Fixed ID for the platform agent
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.95  # Obtaining daily reports is an extremely important observation behavior
            )

            return memory_content
        except:
            error_traceback = traceback.format_exc()
            log.error("Full stack trace is as follows:\n" + error_traceback)
            return "Data acquisition failed"

    @tool
    async def update_platform_theta(new_theta: float, reason: str, net_pressure: float) -> bool | str:
        """
        [Final Decision] Execute the new moderation threshold (new_theta). A decision reason must be provided. net_pressure (today's net pressure).
        """
        try:
            # --- 1. Execute raw operations ---
            old_theta = environment.platform.theta
            environment.platform.theta = new_theta

            environment.platform.platform_theta_change.append(
                {'day_time': environment.day_time, 'old_theta': old_theta, 'new_theta': new_theta, 'reason': reason,
                 'net_pressure': net_pressure})

            # --- 2. Automatically record memory ---
            memory_content = (
                f"On Day {environment.day_time}, I made the final decision and updated the moderation threshold."
                f" My decision reason is: '{reason}'."
                f" Threshold adjusted from {old_theta:.2f} to {new_theta:.2f}."
            )

            await environment.memories_store.add_memory(
                persona_id=environment.platform.name,  # Fixed ID for platform agent
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=1.0  # Updating threshold is a top-priority action
            )

            return True
        except:
            error_traceback = traceback.format_exc()
            log.error("Full stack trace is as follows:\n" + error_traceback)
            return "Operation failed"

    @tool
    async def get_memories(
            query: str,
            top_k: int = 3
    ) -> List[str] | str:
        """
        [Recall] Search memories based on a query.
        """
        try:
            # Get current agent and time from environment
            memories_docs = await environment.memories_store.recall_memories(
                persona_id=environment.platform.name,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if not memories_docs:
                return [f"No memories related to '{query}' were found."]

            # Format the returned Document objects into a string list friendly to the LLM
            formatted_memories = [
                f"Memory (from Day {doc.metadata.get('day_time', 'Unknown')}): {doc.page_content}"
                for doc in memories_docs
            ]

            return formatted_memories
        except:
            error_traceback = traceback.format_exc()
            log.error("Full stack trace is as follows:\n" + error_traceback)
            return "Data acquisition failed"

    return [get_today_platform_data, update_platform_theta, get_memories]
