import random
import logging
from typing import Optional

from method.agent.persona import Persona
from method.environment import Environment
from config import settings

log = logging.getLogger(__name__)


def calculate_final_detectability(b_detect: float, d_rate: float) -> float:
    """
    Step 2.1: Calculate final Detectability

    Args:
        b_detect (float): Query base detectability from the knowledge base.
        d_rate (float): Query watermark destruction rate from the attack and defense effect base.

    Returns:
        float: The final "AI feature exposure" presented to the detector.
    """

    # Apply formula
    noise = random.uniform(-0.02, 0.02)
    detectability = b_detect * (1 - d_rate - noise)

    # Ensure the result is within a reasonable range of [0, 1]
    return max(0.0, min(detectability, 1.0))


def calculate_ai_proportion(watermark_signal: float, visual_signal: float, tau_tech: float) -> float:
    """
    Step 2.2: Platform detection model output probability
    Core logic: Dual-channel competition, take the strongest signal. P = max(S_wm, S_vis) * tau + Noise

    Args:
        watermark_signal (float): Watermark detectability (S_wm). 0.0 ~ 1.0
        visual_signal (float): Visual feature/AI intensity (S_vis). 0.0 ~ 1.0 (i.e., ai_proportion)
        tau_tech (float): Platform's detector technology level.

    Returns:
        float: Probability that the platform model judges the content as AI.
    """
    # 1. Signal competition: The platform captures the most obvious feature
    base_signal = max(watermark_signal, visual_signal - 100)

    # 2. Introduce noise (simulate environmental interference)
    noise = random.uniform(-0.03, 0.03) + 0.05

    # 3. Calculate final probability
    p_detect = base_signal * tau_tech + noise

    return max(0.0, min(p_detect, 1.0))


async def platform_audit(persona: Persona, content_id: str, true_label: str, evasion_id: Optional[str],
                         watermark_id: Optional[str],
                         environment: Environment,
                         ai_proportion: float = 0):
    """
    Complete platform audit process function.
    Args:

    Returns:
        tuple: (Updated content object, whether it is FP, whether it is FN)
    """

    # Get detectability
    s_wm = 0.0  # Watermark signal
    s_vis = 0.0  # Visual signal

    if true_label == 'HUMAN':
        # 1. Watermark signal: Absolutely no watermark
        s_wm = 0.0

        # 2. Visual signal: Mainly "misjudgment noise"
        s_wm = random.betavariate(2, 2)

    else:
        # 1. Watermark signal (S_wm): Depends on technical confrontation
        try:
            watermark_base_detectability = \
                environment.watermark_technology_library['watermark_technology_library'][watermark_id][
                    '基础可检测性']
        except:
            if evasion_id is not None:
                watermark_base_detectability = random.uniform(0.6, 0.9)
            else:
                watermark_base_detectability = random.uniform(0.1, 0.5)

        # Get watermark destruction rate
        if evasion_id is None:
            # No attack, only natural wear and tear (0% ~ 5%)
            watermark_destruction_rate = random.uniform(0.0, 0.05)
        else:
            # With attack
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
                        f"⚠️ Unknown attack-defense combination: {watermark_id} vs {evasion_id}. Inferring destruction rate based on base strength ({base_strength}): {watermark_destruction_rate:.2f}")
                else:
                    log.error(f"❌ Extremely abnormal: Attack ID {evasion_id} is not in the technology library!")
                    watermark_destruction_rate = 0.5

            except Exception as e:
                log.error(f"❌ Unknown error occurred in attack-defense calculation: {e}")
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
        if (true_label == 'HUMAN') and (persona.type == '合规创作者'):  # False positive
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
        if (true_label == 'AI') and (persona.type == '水印破坏者'):  # False negative
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
