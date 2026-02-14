import logging

from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from method.agent.simple_process.platform.platform_logic import get_platform_daily_report, apply_platform_decision
from method.agent.simple_process.platform.platform_models import PlatformDecision
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from method.utils.token_statistics import token_logger
from config import settings

log = logging.getLogger(__name__)


async def linear_platform_main(environment: Environment):
    """
    Platform agent's linear daily decision-making process.
    """
    log.info(f"⚡️ Starting fast linear decision-making process for platform agent 'Aura'...")

    report = await get_platform_daily_report(environment)
    decision_data = {
        "new_theta": report.system_recommendation,
        "reason": (f"[System Auto-Execution] Based on net pressure {report.net_pressure:.2f} "
                   f"(Regulatory={report.regulatory_cost:.2f}, Churn={report.total_churn_cost:.2f}) "
                   f"automatically adjusting threshold.")
    }
    log.info(f"🤖 [Auto-Platform] Auto-adjustment: {report.current_theta:.3f} -> {report.system_recommendation:.3f}")
    # 3. Execute decision (Reuse existing logic)
    await apply_platform_decision(environment, decision_data)


async def platform_main_simple(environment: Environment):
    """ Main entry point for the platform agent process """
    log.info("=" * 60)
    log.info("==========   Starting Platform Agent [Fast Linear] Process   ==========")
    log.info("=" * 60)

    await linear_platform_main(environment)

    log.info("=" * 60)
    log.info("==========   Platform Agent [Fast Linear] Process Completed   ==========")
    log.info("=" * 60)