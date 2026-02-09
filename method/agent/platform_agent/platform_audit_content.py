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

    # 应用公式
    noise = random.uniform(-0.02, 0.02)
    detectability = b_detect * (1 - d_rate - noise)

    # 保证结果在 [0, 1] 的合理范围内
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
    # 1. 信号竞争：平台捕捉最明显的那个特征
    base_signal = max(watermark_signal, visual_signal - 100)

    # 2. 引入噪声 (模拟环境干扰)
    noise = random.uniform(-0.03, 0.03) + 0.05

    # 3. 计算最终概率
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

    # 获取可检测性
    # 初始化两个通道的信号
    s_wm = 0.0  # 水印信号
    s_vis = 0.0  # 视觉信号

    if true_label == 'HUMAN':
        # 1. 水印信号: 绝无水印
        s_wm = 0.0

        # 2. 视觉信号: 主要是"误判噪音"
        # 使用 Beta 分布模拟人类作品偶尔会被误认为有AI特征 (如画风诡异)
        # s_vis = random.betavariate(2, 2)
        s_wm = random.betavariate(2, 2)

    else:
        # 1. 水印信号 (S_wm): 取决于技术对抗
        try:
            watermark_base_detectability = \
                environment.watermark_technology_library['watermark_technology_library'][watermark_id][
                    '基础可检测性']
        except:
            if evasion_id is not None:
                watermark_base_detectability = random.uniform(0.6, 0.9)
            else:
                watermark_base_detectability = random.uniform(0.1, 0.5)

        # 获取水印破坏率
        if evasion_id is None:
            # 无攻击，只有自然损耗 (0% ~ 5%)
            watermark_destruction_rate = random.uniform(0.0, 0.05)
        else:
            # 有攻击
            try:
                # 1. 优先查表：获取特定【水印-攻击】组合的精确对抗结果
                watermark_destruction_rate = environment.watermark_technology_library['attack_defense_effect_library'][
                    (watermark_id, evasion_id)]
            except KeyError:
                # 2. 【核心修改】查表失败（未知组合），读取攻击技术的【通用强度】
                attack_lib = environment.watermark_technology_library['attack_technology_library']

                # 检查 evasion_id 是否在技术库中
                if evasion_id in attack_lib:
                    attack_info = attack_lib[evasion_id]

                    # 直接读取 JSON 中的 "攻击强度" 字段 (0.0 ~ 1.0)
                    # 如果数据缺失，默认给 0.5
                    base_strength = attack_info.get('攻击强度', 0.5)

                    # 3. 引入高斯噪声模拟不确定性
                    watermark_destruction_rate = random.gauss(base_strength, 0.1)

                    log.warning(
                        f"⚠️ 未知攻防组合: {watermark_id} vs {evasion_id}。基于基础强度({base_strength})推算破坏率: {watermark_destruction_rate:.2f}")
                else:
                    # 3. 彻底未知的攻击ID (脏数据兜底)
                    log.error(f"❌ 极其异常：攻击ID {evasion_id} 不在技术库中！")
                    watermark_destruction_rate = 0.5

            except Exception as e:
                log.error(f"❌ 攻防计算发生未知错误: {e}")
                watermark_destruction_rate = 0.5
            # 4. 最终边界截断 (保证在 0.0 ~ 1.0 之间)
            watermark_destruction_rate = max(0.0, min(1.0, watermark_destruction_rate))

        # 计算残留的水印信号
        s_wm = calculate_final_detectability(
            watermark_base_detectability,
            watermark_destruction_rate
        )

        # 2. 视觉信号 (S_vis): 直接等于 AI强度
        s_vis = ai_proportion

    # 步骤 2: 计算最终的平台打分 (P_score)
    # 传入分离的两个信号，由函数内部取 max
    calculate_final_ai_proportion = calculate_ai_proportion(
        watermark_signal=s_wm,
        visual_signal=s_vis,
        tau_tech=environment.platform.tau_tech
    )

    # 步骤 3: 打标决策
    if calculate_final_ai_proportion > environment.platform.theta:
        if (true_label == 'HUMAN') and (persona.type == '合规创作者'):  # 误报
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
        if (true_label == 'AI') and (persona.type == '水印破坏者'):  # 漏报
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
