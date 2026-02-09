import logging
import numpy as np

from method.agent.simple_process.platform.platform_models import PlatformDailyReport
from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings  # 必须导入配置，以获取 mu, w, steep 等参数

log = logging.getLogger(__name__)


def _calculate_regulatory_cost(fn_contents: list, environment: Environment) -> float:
    """
    计算监管成本 C_reg(t)。
    逻辑来源: method/agent/platform_agent/tools.py -> calculate_regulatory_cost
    """
    if not fn_contents:
        return 0.0

    # 1. 计算所有漏报内容的总影响力
    # initial_score=True 表示只计算基础影响力，不经过sigmoid
    total_fn_content_influence = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fn_contents]
    )

    # 2. 公式: C_reg = F_penalty * Total_Influence
    # 注意：完整框架中是 (0 + f_penalty)，这里保持一致
    c_reg = environment.policy.f_penalty * total_fn_content_influence

    return c_reg


def _calculate_churn_cost_details(churned_agents: list, fp_contents: list, environment: Environment) -> dict:
    """
    计算用户流失成本 C_churn(t)。
    逻辑来源: method/agent/platform_agent/tools.py -> calculate_churn_cost
    包含显性流失成本和潜在误报不满成本。
    """
    # --- 步骤 1: 显性流失成本 (单位: 用户的影响力) ---
    explicit_churn_influence = sum([agent.influence for agent in churned_agents]) if churned_agents else 0.0

    # --- 步骤 2: 潜在流失成本 (单位: 当天所有FP内容的原始影响力之和) ---
    # 2.1 计算这些FP内容在当天积累的真实原始影响力
    # 完整框架逻辑：new_fp_influence_today = sum(...) * 0.3
    # 系数 0.3 代表误报带来的不满转化率
    new_fp_influence_today = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fp_contents]
    ) * 0.3

    # 2.2 更新平台的累积不满影响力 (它每日在 environment.start_new_day() 会自动衰减，这里负责累加)
    # 【重要】这一步会修改环境状态，这与完整框架中 update_strategy 工具的行为一致
    environment.platform.total_fp_creator_influence += new_fp_influence_today
    potential_churn_influence = environment.platform.total_fp_creator_influence

    # --- 步骤 3: 总成本 ---
    # 公式: C_churn = Explicit * mu + Potential
    # mu 是基础影响力单价
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
    生成一份结构化的每日平台运营报告。
    该函数对应完整框架中 update_strategy 工具的计算部分。
    """
    log.info("正在生成平台每日运营报告 (逻辑对齐版)...")

    # --- 1. 数据准备 (基于方案二，这里获取的是全量真实实体) ---
    churned_agents = [environment.personas[pid] for pid in environment.platform.public_loss if
                      pid in environment.personas]

    fn_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fn if
                   environment.contents.get_content_by_id(cid)]

    fp_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fp if
                   environment.contents.get_content_by_id(cid)]

    fp_count = len(environment.platform.fp)  # 误报数量

    # --- 2. 成本计算 (调用对齐后的函数) ---
    c_reg = _calculate_regulatory_cost(fn_contents, environment)
    churn_cost_details = _calculate_churn_cost_details(churned_agents, fp_contents, environment)
    c_churn_total = churn_cost_details['total']

    # --- 3. 净压力和系统推荐 ---
    # 逻辑来源: method/agent/platform_agent/tools.py -> update_strategy

    # 公式: Net_Pressure = C_reg - w * C_churn
    # w 是用户流失厌恶系数
    net_pressure = c_reg - environment.platform.w * c_churn_total

    # 公式: Delta_Theta calculation
    # steep 是压力敏感度调节因子
    # eta 是策略调整速率
    tanh_value = np.tanh(net_pressure / environment.platform.steep)
    delta_theta = -environment.platform.eta * tanh_value

    # 计算新阈值并截断
    recommended_theta = float(np.clip(environment.platform.theta + delta_theta, 0.05, 0.95))

    # --- 4. 封装成Pydantic模型 ---
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
        f"每日报告: C_reg={c_reg:.2f}, C_churn={c_churn_total:.2f}, NetP={net_pressure:.2f}, RecTheta={recommended_theta:.3f}")
    return report


async def apply_platform_decision(environment: Environment, decision: dict):
    """
    执行LLM返回的平台决策。
    """
    reason = decision.get('reason', '无提供理由。')

    memory_content = f"在第{environment.day_time}天，我的决策是：{reason}。"
    importance = 0.8

    new_theta = decision.get('new_theta', environment.platform.theta)
    if new_theta is not None:
        old_theta = environment.platform.theta
        environment.platform.theta = new_theta

        # 记录详细的变动日志，方便绘图
        environment.platform.platform_theta_change.append({
            'day_time': environment.day_time,
            'old_theta': old_theta,
            'new_theta': new_theta,
            'reason': reason,
            # 注意：这里我们无法直接获取 net_pressure，除非从外部传入。
            # 但在简化流程中，通常不需要在这个列表里存 net_pressure，
            # 因为 cost_calculation_details_data 已经存了详细数据。
            'net_pressure': 0.0
        })

        log.info(f"平台审核阈值已更新: 从 {old_theta:.3f} -> {new_theta:.3f}。")
        memory_content += f" 我将审核阈值从 {old_theta:.3f} 调整到了 {new_theta:.3f}。"
        importance = 1.0
    else:
        log.warning("决策为'update_theta'但未提供'new_theta'值，跳过更新。")
        memory_content += " 我本想更新阈值，但决策数据不完整。"

    # 记录决策到记忆库
    await environment.memories_store.add_memory(
        persona_id=environment.platform.name,
        content=memory_content,
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=importance
    )

    # 【新增】确保 cost_calculation_details_data 被填充
    # 在完整框架中，update_strategy 会填充这个列表。
    # 在简化框架中，由于 get_platform_daily_report 是只读的（除了累加怨气），
    # 我们最好在这里或者 get_platform_daily_report 里记录一下详细数据，以便 draw_kpi.py 使用。
    # 为了保持简单，建议在 get_platform_daily_report 计算完后，直接写入 environment 的日志列表。
    # (上面的代码中为了保持函数纯净性未加入，但为了绘图一致性，您可能需要加上)
