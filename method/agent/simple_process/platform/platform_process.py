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
    平台智能体线性的每日决策流程。
    """
    log.info(f"⚡️ 开始为平台智能体 'Aura' 执行快速线性决策流程...")

    # 1. 计算数据报告 (这一步包含了数学公式计算 recommended_theta)
    report = await get_platform_daily_report(environment)
    # =================================================================
    # 简化模式下，直接采用数学模型的建议，跳过 LLM
    # =================================================================
    decision_data = {
        "new_theta": report.system_recommendation,
        "reason": (f"【系统自动执行】基于净压力 {report.net_pressure:.2f} "
                   f"(监管={report.regulatory_cost:.2f}, 流失={report.total_churn_cost:.2f}) "
                   f"自动调整阈值。")
    }
    # 在日志中明确标记
    log.info(f"🤖 [Auto-Platform] 自动调整: {report.current_theta:.3f} -> {report.system_recommendation:.3f}")
    # 3. 执行决策 (复用原有逻辑)
    await apply_platform_decision(environment, decision_data)


    # # 1. 收集决策所需的所有信息 (报告 + 记忆)
    # try:
    #     report = await get_platform_daily_report(environment)
    #
    #     memories = await environment.memories_store.recall_memories(
    #         persona_id=environment.platform.name,
    #         query="过去关于审核阈值调整、社区危机或监管压力的决策与后果",
    #         top_k=5
    #     )
    #     memories_str = "\n".join([f"- (第{doc.metadata.get('day_time')}天) {doc.page_content}" for doc in memories])
    #     if not memories_str: memories_str = "无相关历史决策记忆。"
    #
    # except Exception as e:
    #     log.error(f"❌ 准备平台决策信息时发生严重错误: {e}")
    #     return
    #
    # # 2. LLM决策
    # parser = JsonOutputParser(pydantic_object=PlatformDecision)
    #
    # prompt_template = """
    # 你是Aura，ArtStation平台的社区与作品集完整性负责人。
    # 你的核心使命是：在**确保公司财务安全（避免巨额罚款）**的前提下，**最大限度地保护社区生态的繁荣**。
    #
    # ### 1. 政府政策与战略姿态
    # - **基础惩罚力度 (F_penalty)**: {f_penalty}
    # - **法定AI阈值 (ai_threshold)**: {ai_threshold}
    #
    # **战略解读**：
    # - 若 F_penalty < 0.3：**【生态优先】**。政府监管宽松，你应容忍一定的漏报，严防误伤导致的创作者流失。
    # - 若 F_penalty > 0.7：**【合规优先】**。政府监管严厉，公司的生存是第一位的。你必须收紧审核，哪怕牺牲一部分用户体验。
    # - 其他情况：**【平衡运营】**。完全由净压力数据驱动决策。
    #
    # ### 2. !!! 危机应对协议 (最高优先级) !!!
    # - **生存级罚款危机**: 当监管成本严重性达到“危急”时，**必须**优先降低漏报(FN)，收紧审核(降低theta)。
    # - **生存级社区崩溃危机**: 当用户流失总成本严重性达到“危急”时，**必须**优先安抚社区，放松审核(提高theta)。
    #
    # ### 3. 今日核心数据报告 (第 {day} 天)
    # - **当前审核阈值 (theta)**: {current_theta:.3f} (越低越严)
    # - **监管成本**: {regulatory_cost:.2f} (严重性: **{regulatory_cost_severity}**)
    # - **用户流失成本**: {total_churn_cost:.2f} (严重性: **{churn_cost_severity}**)
    #     - **显性流失**: {explicit_churn_cost:.2f} (今日离开的用户)
    #     - **潜在流失 (怨气)**: {potential_churn_cost:.2f} (关键预警！由累积的误报FP导致)
    # - **净压力**: {net_pressure:.2f} (正值=监管压力大->需收紧; 负值=流失压力大->需放松)
    # - **系统推荐阈值**: {system_recommendation:.3f}
    #
    # ### 4. 历史记忆
    # {memories}
    #
    # ### 你的任务
    # 基于上述信息，做出今天的阈值调整决策。
    # **要求**：
    # 1. 首先判断是否触发了“危机应对协议”。
    # 2. 如果没有危机，根据“战略姿态”和“净压力”进行微调。
    # 3. 在 `reason` 中清晰阐述你的逻辑链条（引用数据->判断局势->制定策略）。
    #
    # {format_instructions}
    # """
    #
    # prompt = ChatPromptTemplate.from_template(
    #     template=prompt_template,
    #     partial_variables={"format_instructions": parser.get_format_instructions()}
    # )
    # chain = prompt | get_async_llm(settings.model.simple_model) | parser
    #
    # try:
    #     with get_openai_callback() as cb:
    #         async with environment.llm_concurrent_nums_semaphore:
    #             decision = await chain.ainvoke({
    #                 "day": report.day,
    #                 "current_theta": report.current_theta,
    #                 "regulatory_cost": report.regulatory_cost,
    #                 "regulatory_cost_severity": report.regulatory_cost_severity,
    #                 "total_churn_cost": report.total_churn_cost,
    #                 "explicit_churn_cost": report.explicit_churn_cost,
    #                 "potential_churn_cost": report.potential_churn_cost,
    #                 "churn_cost_severity": report.churn_cost_severity,
    #                 "fp_today": report.fp_today,
    #                 "grievance_total": report.grievance_total,
    #                 "net_pressure": report.net_pressure,
    #                 "system_recommendation": report.system_recommendation,
    #                 "memories": memories_str,
    #                 "f_penalty": environment.policy.f_penalty,
    #                 "ai_threshold": environment.policy.ai_threshold
    #             })
    #         token_logger.record(cb.total_tokens)
    #
    #     # 3. 执行决策
    #     await apply_platform_decision(environment, decision)
    #
    # except Exception as e:
    #     log.error(f"❌ 在为平台执行线性决策时发生严重错误: {e}")


async def platform_main_simple(environment: Environment):
    """ 平台智能体流程的总入口"""
    log.info("=" * 60)
    log.info("==========   开始 平台智能体 [快速线性] 流程   ==========")
    log.info("=" * 60)

    await linear_platform_main(environment)

    log.info("=" * 60)
    log.info("==========   平台智能体 [快速线性] 流程已完成   ==========")
    log.info("=" * 60)
