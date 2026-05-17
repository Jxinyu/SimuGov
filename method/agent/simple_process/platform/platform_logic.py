import logging
import numpy as np

from method.agent.simple_process.platform.platform_models import PlatformDailyReport
from method.environment import Environment
from method.store.long_memory_store import MemoryType
from config import settings                               

log = logging.getLogger(__name__)


def _calculate_regulatory_cost(fn_contents: list, environment: Environment) -> float:
    """
    计算监管成本 C_reg(t)。
    逻辑来源: method/agent/platform_agent/tools.py -> calculate_regulatory_cost
    """
    if not fn_contents:
        return 0.0

                      
                                              
    total_fn_content_influence = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fn_contents]
    )

                                                
                                      
    c_reg = environment.policy.f_penalty * total_fn_content_influence

    return c_reg


def _calculate_churn_cost_details(churned_agents: list, fp_contents: list, environment: Environment) -> dict:
    """
    计算用户流失成本 C_churn(t)。
    逻辑来源: method/agent/platform_agent/tools.py -> calculate_churn_cost
    包含显性流失成本和潜在误报不满成本。
    """
                                       
    explicit_churn_influence = sum([agent.influence for agent in churned_agents]) if churned_agents else 0.0

                                                 
                               
                                                    
                         
    new_fp_influence_today = sum(
        [environment.contents.calculate_content_influence(content, environment, initial_score=True)
         for content in fp_contents]
    ) * 0.3

                                                                      
                                                    
    environment.platform.total_fp_creator_influence += new_fp_influence_today
    potential_churn_influence = environment.platform.total_fp_creator_influence

                       
                                             
                 
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

                                          
    churned_agents = [environment.personas[pid] for pid in environment.platform.public_loss if
                      pid in environment.personas]

    fn_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fn if
                   environment.contents.get_content_by_id(cid)]

    fp_contents = [environment.contents.get_content_by_id(cid) for cid in environment.platform.fp if
                   environment.contents.get_content_by_id(cid)]

    fp_count = len(environment.platform.fp)        

                                
    c_reg = _calculate_regulatory_cost(fn_contents, environment)
    churn_cost_details = _calculate_churn_cost_details(churned_agents, fp_contents, environment)
    c_churn_total = churn_cost_details['total']

                         
                                                                   

                                            
                 
    net_pressure = c_reg - environment.platform.w * c_churn_total

                                 
                      
                 
    tanh_value = np.tanh(net_pressure / environment.platform.steep)
    delta_theta = -environment.platform.eta * tanh_value

              
    recommended_theta = float(np.clip(environment.platform.theta + delta_theta, 0.05, 0.95))

                              
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
    net_pressure = float(decision.get('net_pressure', 0.0))
    if new_theta is not None:
        old_theta = environment.platform.theta
        environment.platform.theta = new_theta

        log.info(
            f"\n[update_platform_theta]\n\tpersona=[agent_id={environment.platform.name}]\n\t"
            f"params=[new_theta={new_theta}\n\t\treason={reason}\n\t\t"
            f"net_pressure={net_pressure}]\n\t"
            f"env=[day_time={environment.day_time}]")

                        
        environment.platform.platform_theta_change.append({
            'day_time': environment.day_time,
            'old_theta': old_theta,
            'new_theta': new_theta,
            'reason': reason,
            'net_pressure': net_pressure,
        })

        log.info(f"平台审核阈值已更新: 从 {old_theta:.3f} -> {new_theta:.3f}。")
        memory_content += f" 我将审核阈值从 {old_theta:.3f} 调整到了 {new_theta:.3f}。"
        importance = 1.0
    else:
        log.warning("决策为'update_theta'但未提供'new_theta'值，跳过更新。")
        memory_content += " 我本想更新阈值，但决策数据不完整。"

              
    await environment.memories_store.add_memory(
        persona_id=environment.platform.name,
        content=memory_content,
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=importance
    )

                                              
                                     
                                                       
                                                                      
                                                                       
                                           
