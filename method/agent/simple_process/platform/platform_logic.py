import logging
import numpy as np

from method.agent.simple_process.platform.platform_models import PlatformDailyReport
from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings

log = logging.getLogger(__name__)


def _calculate_regulatory_cost(fn_contents: list, environment: Environment) -> float:
    """
    Calculate regulatory cost C_reg(t).
    Logic source: method/agent/platform_agent/tools.py -> calculate_regulatory_cost
    """
    if not fn_contents:
        return 0.0

    # 1. Calculate the total influence of all missed detection content (FN)
    total_fn_content_influence = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fn_contents]
    )

    # 2. Formula: C_reg = F_penalty * Total_Influence
    c_reg = environment.policy.f_penalty * total_fn_content_influence

    return c_reg


def _calculate_churn_cost_details(churned_agents: list, fp_contents: list, environment: Environment) -> dict:
    """
    Calculate user churn cost C_churn(t).
    Logic source: method/agent/platform_agent/tools.py -> calculate_churn_cost
    Includes explicit churn cost and potential false positive dissatisfaction cost.
    """
    # --- Step 1: Explicit churn cost (Unit: User influence) ---
    explicit_churn_influence = sum([agent.influence for agent in churned_agents]) if churned_agents else 0.0

    # --- Step 2: Potential churn cost (Unit: Sum of raw influence of all FP content today) ---
    new_fp_influence_today = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fp_contents]
    ) * 0.3

    # 2.2 Update the platform's cumulative dissatisfaction influence (it decays automatically in environment.start_new_day(), this accumulates here)
    environment.platform.total_fp_creator_influence += new_fp_influence_today
    potential_churn_influence = environment.platform.total_fp_creator_influence

    # --- Step 3: Total cost ---
    total_churn_influence = explicit_churn_influence * settings.platform.mu + potential_churn_influence

    return {
        "explicit": explicit_churn_influence,
        "potential": potential_churn_influence,
        "total": total_churn_influence,
        "grievance": environment.platform.total_fp_creator_influence,
        "new_fp_influence": new_fp_influence_today
    }


async def get_platform_daily_report(environment: Environment) -> PlatformDailyReport:
    """
    Generate a structured daily platform operation report.
    This function corresponds to the calculation part of the update_strategy tool in the complete framework.
    """
    log.info("Generating platform daily operation report (logic-aligned version)...")

    # --- 1. Data preparation (Based on scheme 2, retrieving full real entities here) ---
    churned_agents = [environment.personas[pid] for pid in environment.platform.public_loss if
                      pid in environment.personas]

    fn_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fn if
                   environment.contents.get_content_by_id(cid)]

    fp_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fp if
                   environment.contents.get_content_by_id(cid)]

    fp_count = len(environment.platform.fp)  # Number of false positives

    # --- 2. Cost calculation (calling aligned functions) ---
    c_reg = _calculate_regulatory_cost(fn_contents, environment)
    churn_cost_details = _calculate_churn_cost_details(churned_agents, fp_contents, environment)
    c_churn_total = churn_cost_details['total']

    # --- 3. Net pressure and system recommendation ---
    net_pressure = c_reg - environment.platform.w * c_churn_total

    # Formula: Delta_Theta calculation
    tanh_value = np.tanh(net_pressure / environment.platform.steep)
    delta_theta = -environment.platform.eta * tanh_value

    # Calculate new threshold and clip
    recommended_theta = float(np.clip(environment.platform.theta + delta_theta, 0.05, 0.95))

    # --- 4. Encapsulate into Pydantic model ---
    report = PlatformDailyReport(
        day=environment.day_time,
        current_theta=environment.platform.theta,
        regulatory_cost=c_reg,
        regulatory_cost_severity=environment.platform.get_severity_level(c_reg, 'reg'),
        total_churn_cost=c_churn_total,
        explicit_churn_cost=churn_cost_details['explicit'],
        potential_churn_cost=churn_cost_details['potential'],
        churn_cost_severity=environment.platform.get_severity_level(c_churn_total, 'churn'),
        fp_today=fp_count,
        grievance_total=churn_cost_details['grievance'],
        net_pressure=net_pressure,
        system_recommendation=recommended_theta
    )

    log.info(
        f"Daily report: C_reg={c_reg:.2f}, C_churn={c_churn_total:.2f}, NetP={net_pressure:.2f}, RecTheta={recommended_theta:.3f}")
    return report


async def apply_platform_decision(environment: Environment, decision: dict):
    """
    Execute platform decisions returned by LLM.
    """
    reason = decision.get('reason', 'No reason provided.')

    memory_content = f"On day {environment.day_time}, my decision is: {reason}."
    importance = 0.8

    new_theta = decision.get('new_theta', environment.platform.theta)
    if new_theta is not None:
        old_theta = environment.platform.theta
        environment.platform.theta = new_theta

        # Record detailed change logs for plotting
        environment.platform.platform_theta_change.append({
            'day_time': environment.day_time,
            'old_theta': old_theta,
            'new_theta': new_theta,
            'reason': reason,
            'net_pressure': 0.0
        })

        log.info(f"Platform moderation threshold updated: from {old_theta:.3f} to {new_theta:.3f}.")
        memory_content += f" I adjusted the moderation threshold from {old_theta:.3f} to {new_theta:.3f}."
        importance = 1.0
    else:
        log.warning("Decision is 'update_theta' but 'new_theta' value not provided, skipping update.")
        memory_content += " I intended to update the threshold, but decision data is incomplete."

    # Record decision to memory store
    await environment.memories_store.add_memory(
        persona_id=environment.platform.name,
        content=memory_content,
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=importance
    )
