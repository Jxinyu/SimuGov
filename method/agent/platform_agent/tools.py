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
    计算监管成本 C_reg(t)。
    Args:
        fn_contents (list): 当天所有被漏报(FN)的内容对象列表。
                            每个内容对象应有一个 属性。
        f_penalty (float): 政府设定的基础罚款单位。

    Returns:
        float: 计算出的总监管成本。
    """
    if not fn_contents:
        return 0.0, 0.0

    # 计算所有漏报内容的总影响力
    total_fn_content_influence = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True) for content in
         fn_contents]
    )

    c_reg = (0 + environment.policy.f_penalty) * total_fn_content_influence
    return c_reg, total_fn_content_influence


def calculate_churn_cost(churned_agents, environment) -> dict:
    """
   计算用户流失成本 C_churn(t)，包含显性流失成本和潜在误报不满成本。

   Args:
       churned_agents (list): 当天所有流失的智能体对象列表。
       environment: 环境对象，用于获取平台参数和更新累积不满值。

   Returns:
       float: 计算出的总用户流失成本。
   """
    # --- 步骤 1: 显性流失成本 (单位: 用户的影响力) ---
    explicit_churn_influence = sum([agent.influence for agent in churned_agents]) if churned_agents else 0.0

    # --- 步骤 2: 潜在流失成本 (单位: 当天所有FP内容的原始影响力之和) ---
    # 2.1 获取当天所有被误报(FP)的内容对象
    fp_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fp if
                   environment.contents.get_content_by_id(cid)]

    # 2.2 计算这些FP内容在当天结束时积累的真实原始影响力
    ls = [environment.contents.calculate_content_influence(content, environment, initial_score=True) for content in
          fp_contents]
    new_fp_influence_today = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True) for content in
         fp_contents]
    ) * 0.3

    # 2.3 更新平台的累积不满影响力 (它每日会自动衰减)
    environment.platform.total_fp_creator_influence += new_fp_influence_today
    potential_churn_influence = environment.platform.total_fp_creator_influence

    # --- 步骤 3: 总“等效流失影响力” ---
    total_churn_influence = explicit_churn_influence * settings.platform.mu + potential_churn_influence

    if total_churn_influence > 10:
        log.warning(f"用户流失成本超过10，请检查！")

    return {
        'total': total_churn_influence,
        'explicit_churn': explicit_churn_influence,
        'potential_from_fp': potential_churn_influence,
        'fp_influence_today': new_fp_influence_today,
        'current_grievance': environment.platform.total_fp_creator_influence
    }


def update_strategy(fn_contents, churned_agents, environment):
    """
    根据当天的运营数据，计算成本、净压力，并更新审核阈值 θ。
    Args:
        fn_contents (list): 当天所有被漏报(FN)的内容列表。
        churned_agents (list): 当天所有流失的智能体列表。
        environment: 环境对象。
    Returns:
        dict: 包含详细成本、净压力和推荐新阈值的字典。
    """
    # 步骤 1: 计算核心成本
    c_reg, total_fn_content_influence = calculate_regulatory_cost(fn_contents, environment)
    churn_cost_details = calculate_churn_cost(churned_agents, environment)
    c_churn_total = churn_cost_details['total']  # 获取总的用户流失成本

    if c_churn_total > 10:
        log.warning(f"用户流失成本超过10，请检查！")

    # 步骤 2: 计算净压力 (使用总用户流失成本)
    net_pressure = c_reg - environment.platform.w * c_churn_total

    # 步骤 3: 计算调整量 Δθ
    # 使用 tanh 函数将净压力映射到 [-1, 1] 区间
    tanh_value = np.tanh(net_pressure / environment.platform.steep)
    delta_theta = -environment.platform.eta * tanh_value

    # 步骤 4: 更新阈值 θ(t+1)，并使用 clip 保证其在有效范围内
    new_theta = environment.platform.theta + delta_theta
    new_theta = float(np.clip(new_theta, 0.05, 0.95))

    # 步骤 5: 返回更详细的报告
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
    工厂函数：创建并返回与特定 ContentStore 实例绑定的工具。
    这是一种依赖注入的实现方式。

    Args:
        :param environment:

    Returns:
        一个包含配置好的工具的列表。
    """

    @tool
    async def get_today_platform_data() -> dict | str:
        """
        获取数据分析报告,包含核心成本、净压力和系统初步的阈值调整建议。决策首选。
        """
        try:
            # --- 1. 执行原始操作 ---
            churned_agents = [environment.personas[persona_id] for persona_id in
                              environment.platform.public_loss]
            fn_contents = [environment.contents.get_content_by_id(content_id) for content_id in
                           environment.platform.fn if environment.contents.get_content_by_id(content_id)]

            report_data = update_strategy(fn_contents, churned_agents, environment)

            # 从 report_data 中提取所需变量，让后续代码更清晰
            net_pressure = report_data['程序计算的净压力']
            c_reg = report_data['程序计算的监管成本']
            c_churn_total = report_data['程序计算的用户流失成本_总计']
            c_churn_explicit = report_data['程序计算的用户流失成本_显性']
            c_churn_potential = report_data['程序计算的用户流失成本_潜在(误报)']
            fn_count = report_data['程序计算的漏报数量']
            fp_count = report_data['程序计算的误报数量']
            current_theta = report_data['当前天数的审核阈值']
            new_theta = report_data['程序计算的新审核阈值']
            reg_assessment = report_data['监管成本评估']
            churn_assessment = report_data['用户流失成本评估']
            platform_adjust_history = report_data['近几天平台的调整数据'][:-1]

            # 进行逻辑判断
            if net_pressure > 0:
                dominant_cost_name = "监管压力"
            else:
                dominant_cost_name = "用户流失压力"

            # 构建记忆内容
            memory_content = (
                f"**第 {environment.day_time} 天 策略评估报告**\n\n"
                f"**1. 核心结论：**\n"
                f"今日，{dominant_cost_name} 成为主导矛盾。净压力为 {net_pressure:.2f}，表明当前审核策略与市场环境失衡。\n\n"
                f"**2. 成本深度分析：**\n"
                f"* **监管成本 ({reg_assessment})**: {c_reg:.2f}，主要由 {fn_count} 次漏报事件引起。\n"
                f"* **用户流失成本 ({churn_assessment})**: {c_churn_total:.2f}，其中潜在的“创作者不满”成本 ({c_churn_potential:.2f}) 正在累积，当日新增 {fp_count} 次误报事件。\n\n"
                f"**3. 趋势与建议：**\n"
                f"为应对今日的 {dominant_cost_name}，系统建议将阈值从 {current_theta:.3f} 调整至 {new_theta:.3f}。\n"
            )

            await environment.memories_store.add_memory(
                persona_id=environment.platform.name,  # 平台智能体的固定ID
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=0.95  # 获取每日报告是极其重要的观察行为
            )

            return memory_content
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "数据获取失败"

    @tool
    async def update_platform_theta(new_theta: float, reason: str, net_pressure: float) -> bool | str:
        """
        【最终决策】执行新的审核阈值(new_theta)。必须提供决策理由(reason)。net_pressure(今天的净压力)
        """
        try:
            # --- 1. 执行原始操作 ---
            old_theta = environment.platform.theta
            environment.platform.theta = new_theta

            environment.platform.platform_theta_change.append(
                {'day_time': environment.day_time, 'old_theta': old_theta, 'new_theta': new_theta, 'reason': reason,
                 'net_pressure': net_pressure})

            # --- 2. 自动记录记忆 ---
            memory_content = (
                f"在第 {environment.day_time} 天，我做出了最终决策并更新了审核阈值。"
                f" 我的决策理由是: '{reason}'。"
                f" 阈值从 {old_theta:.2f} 调整为 {new_theta:.2f}。"
            )

            await environment.memories_store.add_memory(
                persona_id=environment.platform.name,  # 平台智能体的固定ID
                content=memory_content,
                day_time=environment.day_time,
                memory_type=MemoryType.EXPERIENCE,
                important_score=1.0  # 更新阈值是最高重要性的行动
            )

            return True
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "操作失败"

    @tool
    async def get_memories(
            query: str,
            top_k: int = 3
    ) -> List[str] | str:
        """
        【回忆】根据query搜索记忆。
        """
        try:
            # 从环境中获取当前智能体和时间
            memories_docs = await environment.memories_store.recall_memories(
                persona_id=environment.platform.name,
                query=query,
                top_k=top_k,
                memory_type=MemoryType.EXPERIENCE,
            )

            if not memories_docs:
                return [f"没有找到与 '{query}' 相关的记忆。"]

            # 将返回的Document对象格式化为对LLM更友好的字符串列表
            formatted_memories = [
                f"记忆 (来自第 {doc.metadata.get('day_time', '未知')} 天): {doc.page_content}"
                for doc in memories_docs
            ]

            return formatted_memories
        except:
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
            return "数据获取失败"

    return [get_today_platform_data, update_platform_theta, get_memories]
