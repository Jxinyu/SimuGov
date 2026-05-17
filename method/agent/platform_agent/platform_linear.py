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
    输出格式
    """
    reason: str = Field(description="调整审核阈值的理由", max_length=160)
    theta: float = Field(description="新的审核阈值 θ")
    net_pressure: float = Field(description="程序计算出的净压力")


async def construct_prompt():
    """
    构建提示词
    :return:
    """

    return """
# --- 1. 角色、目标与背景 ---
你叫Aura，是ArtStation的社区与作品集完整性负责人。你的使命是在确保公司财务和法律安全的前提下，最大限度地保护ArtStation作为全球顶级艺术家社区的圣殿地位。
你的最终目标是基于今天的数据和历史记忆，决定下一个审核阈值`theta`应该是多少。
今天是模拟世界的第 {day_time} 天。当前的审核阈值 θ(t) = {current_theta:.2f}。
（θ值越低，代表审核越严格；越高则越松）

# --- 2. 核心困境与危机应对协议 ---
你的决策是在“社区之死”（用户流失）和“公司之死”（监管罚款）之间走钢丝。
**危机应对协议 (最高优先级):** 当任何一方的成本评估达到“危急 - 生存威胁”等级时，必须绝对优先处理此危机。

# --- 3. 政府宏观政策 ---
你的决策必须基于当前的政策参数：

1.  **AI判定红线 : {ai_threshold:.2f}**
    *   *含义*：法律规定 AI成分 > {ai_threshold:.2f} 的内容必须标注。
    *   *影响*：此值越低（越严），法律认定的“AI内容”越多，你的审核压力越大。

2.  **惩罚力度: {f_penalty:.2f} 倍**
    *   *含义*：这是政府执法的**“鞭子”**。漏报违规内容的罚款倍率。
    *   *威慑*：若此值很高（如 >0.7），漏报导致的监管成本 ($C_{{reg}}$) 会呈指数级爆炸。**公司生存是第一位的。**

3.  **教育投入: {e_edu}**
    *   *含义*：这是社会的**“缓冲垫”**。政府投入资源教育公众理解技术局限。
    *   *机会*：若此值高，公众对“误伤(FP)”的包容度会慢慢变高，用户流失成本 ($C_{{churn}}$) 会慢慢降低。

# --- 4. 政策解读与战略姿态 ---
{policy_interpretation}

# --- 5. 今日核心数据报告 ---
{today_data_report}

# --- 6. 历史数据分析复盘 ---
{historical_data_analysis}

# --- 7. 历史theta调整复盘 ---
{historical_theta_adjust}

# --- 8. 你的任务：决策与推理 ---
作为Aura，请严格遵循你的角色设定、危机协议和政策解读，综合分析以上所有信息。
你的任务是：**决定下一个最合理的审核阈值`theta`，并提供一个结构化的、有说服力的决策理由。**
你的推理过程应该在理由中清晰地体现出来，特别是要说明你是如何权衡各种成本、以及如何吸取历史教训的。
！！！请参考昨天的阈值，保持阈值的连续性，避免剧烈跳变！！！

{format_instructions}
    """


async def get_policy_interpretation(environment: Environment) -> str:
    """
    获取政策解读
    :param environment:
    :return:
    """
             
    f_penalty = environment.policy.f_penalty
    if f_penalty > 0.7:
        policy_interpretation = "当前惩罚力度很高，战略姿态为【合规优先，规避风险】。必须优先控制监管成本，可以接受一定的社区代价。"
    elif f_penalty < 0.3:
        policy_interpretation = "当前惩罚力度很低，战略姿态为【生态优先，鼓励创新】。首要任务是保护社区活力，对用户流失成本极其敏感。"
    else:
        policy_interpretation = "当前惩罚力度中等，战略姿态为【寻求平衡，稳健运营】。决策应完全由数据驱动，精确权衡两种成本。"
    return policy_interpretation


async def get_today_data_report(environment: Environment) -> str:
    """
    获取今天的数据报告
    :param environment:
    :return:
    """
    churned_agents = [environment.personas[persona_id] for persona_id in
                      environment.platform.public_loss]
    fn_contents = [environment.contents.get_content_by_id(content_id) for content_id in
                   environment.platform.fn if environment.contents.get_content_by_id(content_id)]

    report_data = update_strategy(fn_contents, churned_agents, environment)

                           
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
                          
            '旧审核阈值': environment.platform.theta,
            '新审核阈值': new_theta_suggestion,

                         
            '监管成本': c_reg,
            '监管成本评估': reg_assessment,
            '用户流失成本_总计': c_churn_total,
            '用户流失成本_显性': c_churn_explicit,
            '用户流失成本_潜在(误报)': c_churn_potential,
            '用户流失成本评估': churn_assessment,
            '净压力': net_pressure,

                          
            '误报数量': fp_count,
            '误报内容的影响力': report_data['程序计算的误报内容的影响力'],                 
            '漏报数量': fn_count,
            '漏报内容的影响力': report_data['程序计算的漏报内容的影响力'],                 

                           
            '用户流失数量': len(churned_agents),
            '今日流失用户的总影响力': today_churn_influence_sum,                           

            '今天创作者发布的内容数量': len(
                [c for c in environment.contents.get_all_contents() if c.time == environment.day_time]),
            '今天创作者发布内容的总影响力': sum(
                [environment.contents.calculate_content_influence(c, environment, True) for c in
                 environment.contents.get_all_contents() if c.time == environment.day_time]),

                           
            '用户平均影响力': sum(user_influence) / len(user_influence) if len(user_influence) > 0 else 0,
            '用户最大的影响力': max(user_influence) if user_influence else 0,
            '用户最小影响力': min(user_influence) if user_influence else 0,
        }
    })

            
    if net_pressure > 0:
        dominant_cost_name = "监管压力"
    else:
        dominant_cost_name = "用户流失压力"

            
    today_data_analysis_report = (
        f"【第 {environment.day_time} 天 策略评估报告】\n\n"
        f"1. 核心结论：\n"
        f"今日主导矛盾为 {dominant_cost_name}。净压力为 {net_pressure:.2f}，表明需向“{'收紧' if net_pressure > 0 else '放宽'}审核”方向调整。\n\n"
        f"2. 成本深度分析：\n"
        f"- 监管成本 ({reg_assessment}): {c_reg:.2f} (由 {fn_count} 次漏报事件引起)。\n"
        f"- 用户流失成本 ({churn_assessment}): {c_churn_total:.2f} (其中潜在不满成本为 {c_churn_potential:.2f}。 显性成本为 {c_churn_explicit:.2f},由 {len(churned_agents)} 个用户流失引起)。\n\n"
        f"3. 系统建议：\n"
        f"基于今日数据，数学模型建议将阈值从 {current_theta:.3f} 调整至 {new_theta_suggestion:.3f}。"
    )

                   
    store_platform_data_report_memory = environment.memories_store.add_memory(
        persona_id=environment.platform.name,              
        content=today_data_analysis_report,
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=0.95                    
    )

               
    environment.add_background_task(store_platform_data_report_memory)

    return today_data_analysis_report


async def get_historical_data_analysis(environment: Environment) -> str:
    """
    获取历史数据报告分析
    :param environment:
    :return:
    """
    result_memory = await environment.memories_store.recall_memories(
        persona_id=environment.platform.name,
        query="策略评估报告",
        top_k=5,
        memory_type=MemoryType.EXPERIENCE,
        gamma=0.5,                     
    )
    return "".join([memory.page_content for memory in result_memory])


async def get_historical_theta_adjust(environment: Environment) -> str:
    """
    获取历史theta调整
    :param environment:
    :return:
    """
    result = ""
    for content in environment.platform.platform_theta_change:
        result += f"第{content['day_time']}天: 净压力为 {content['net_pressure']: .2f}，决策将θ从 {content['old_theta']} 调整至 {content['new_theta']}。\n"
    return result


async def platform_reflection_adjust_theta(environment: Environment):
    """
    平台反思调整审核阈值 θ
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

    llm_chain = prompt | llm

    ai_message = await llm_chain.ainvoke(
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

                                        
    response = output_format.invoke(ai_message)

    if response:
                             
        thought_text = f"【思维链/CoT】{response}"

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
        content=f"平台调整审核阈值 由 {environment.platform.theta} 变为 {response['theta']}，理由为 {response['reason']}",
        day_time=environment.day_time,
        memory_type=MemoryType.EXPERIENCE,
        important_score=0.95
    )

               
    environment.platform.theta = response['theta']

    log.info(f"\n[platform_reflection_adjust_theta]\n\t平台调整审核阈值为 {response['theta']}，理由为 {response['reason']}")
