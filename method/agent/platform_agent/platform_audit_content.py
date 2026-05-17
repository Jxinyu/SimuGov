import random
import logging
from typing import Optional

from method.agent.persona import Persona
from method.environment import Environment
from config import settings

log = logging.getLogger(__name__)


def calculate_final_detectability(b_detect: float, d_rate: float) -> float:
    """
    步骤 2.1: 计算最终可检测性 (Detectability)

    Args:
        b_detect (float): 从知识库查询基础可检测性
        d_rate (float): 从攻防效果库查询水印破坏率

    Returns:
        float: 最终呈现给检测器的“AI特征暴露度”.
    """

          
    noise = random.uniform(-0.02, 0.02)
    detectability = b_detect * (1 - d_rate - noise)

                         
    return max(0.0, min(detectability, 1.0))


def calculate_ai_proportion(watermark_signal: float, visual_signal: float, tau_tech: float) -> float:
    """
    步骤 2.2: 平台检测模型输出概率
    核心逻辑：双通道竞争，取最强信号。 P = max(S_wm, S_vis) * tau + Noise

    Args:
        watermark_signal (float): 水印的可检测性 (S_wm). 0.0 ~ 1.0
        visual_signal (float): 视觉特征/AI强度 (S_vis). 0.0 ~ 1.0 (即 ai_proportion)
        tau_tech (float): 平台的检测器技术水平.

    Returns:
        float: 平台模型判断内容为AI的概率.
    """
                          
    base_signal = max(watermark_signal, visual_signal - 100)

                      
    noise = random.uniform(-0.03, 0.03) + 0.05

               
    p_detect = base_signal * tau_tech + noise

    return max(0.0, min(p_detect, 1.0))


async def platform_audit(persona: Persona, content_id: str, true_label: str, evasion_id: Optional[str],
                         watermark_id: Optional[str],
                         environment: Environment,
                         ai_proportion: float = 0):
    """
    完整的平台审核流程函数.
    Args:

    Returns:
        tuple: (更新后的 content 对象, 是否为 FP, 是否为 FN)
    """

            
                
    s_wm = 0.0        
    s_vis = 0.0        

    if true_label == 'HUMAN':
                       
        s_wm = 0.0

                            
                                              
                                          
        s_wm = random.betavariate(2, 2)

    else:
                                 
        try:
            watermark_base_detectability =\
                environment.watermark_technology_library['watermark_technology_library'][watermark_id][
                    '基础可检测性']
        except:
            if evasion_id is not None:
                watermark_base_detectability = random.uniform(0.6, 0.9)
            else:
                watermark_base_detectability = random.uniform(0.1, 0.5)

                 
        if evasion_id is None:
                                  
            watermark_destruction_rate = random.uniform(0.0, 0.05)
        else:
                 
            try:
                                              
                watermark_destruction_rate = environment.watermark_technology_library['attack_defense_effect_library'][
                    (watermark_id, evasion_id)]
            except KeyError:
                                                   
                attack_lib = environment.watermark_technology_library['attack_technology_library']

                                       
                if evasion_id in attack_lib:
                    attack_info = attack_lib[evasion_id]

                                                        
                                    
                    base_strength = attack_info.get('攻击强度', 0.5)

                                     
                    watermark_destruction_rate = random.gauss(base_strength, 0.1)

                    log.warning(
                        f"⚠️ 未知攻防组合: {watermark_id} vs {evasion_id}。基于基础强度({base_strength})推算破坏率: {watermark_destruction_rate:.2f}")
                else:
                                          
                    log.error(f"❌ 极其异常：攻击ID {evasion_id} 不在技术库中！")
                    watermark_destruction_rate = 0.5

            except Exception as e:
                log.error(f"❌ 攻防计算发生未知错误: {e}")
                watermark_destruction_rate = 0.5
                                          
            watermark_destruction_rate = max(0.0, min(1.0, watermark_destruction_rate))

                   
        s_wm = calculate_final_detectability(
            watermark_base_detectability,
            watermark_destruction_rate
        )

                                    
        s_vis = ai_proportion

                               
                          
    calculate_final_ai_proportion = calculate_ai_proportion(
        watermark_signal=s_wm,
        visual_signal=s_vis,
        tau_tech=environment.platform.tau_tech
    )

                
    if calculate_final_ai_proportion > environment.platform.theta:
        if (true_label == 'HUMAN') and (persona.type == '合规创作者'):      
            async with environment.state_lock:
                environment.platform.fp.append(content_id)
                environment.platform.fp_data.append({
                    "day_time": environment.day_time,
                    "agent_id": persona.agent_id,
                    "content_id": content_id,
                    "calculate": {
                        "水印信号": s_wm,
                        "视觉信号": s_vis,
                        "最终可检测性": calculate_final_ai_proportion,
                        "平台检测模型输出概率": calculate_final_ai_proportion,
                        "当前阈值": environment.platform.theta
                    }
                })
        return 'AI'
    else:
        if (true_label == 'AI') and (persona.type == '水印破坏者'):      
            async with environment.state_lock:
                environment.platform.fn.append(content_id)
                environment.platform.fn_data.append({
                    "day_time": environment.day_time,
                    "agent_id": persona.agent_id,
                    "content_id": content_id,
                    "calculate": {
                        "水印信号": s_wm,
                        "视觉信号": s_vis,
                        "最终可检测性": calculate_final_ai_proportion,
                        "平台检测模型输出概率": calculate_final_ai_proportion,
                        "当前阈值": environment.platform.theta
                    }
                })
        return 'HUMAN'
