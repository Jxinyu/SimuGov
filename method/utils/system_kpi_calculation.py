from method.environment import Environment
from config import settings

import math


def calculate_safety_kpi(environment: Environment):
    """
    Calculate Safety KPI: safety (Ecological & Objective View)
    """
    window_size = settings.platform.kpi_window_size
    start_day = max(1, environment.day_time - window_size + 1)

    # 1. Get all content within the window period
    window_contents = [c for c in environment.contents.get_all_contents()
                       if c.time >= start_day]

    # 2. Calculate denominator: Total content influence
    total_content_influence = sum([
        environment.contents.calculate_content_influence(c, environment, True)
        for c in window_contents
    ])

    # If there is no content, treat it as absolutely safe
    if total_content_influence == 0:
        environment.system_kpi.safety.append(1.0)
        return 1.0

    # 3. Calculate numerator: Total Ecological Harm
    total_harm_influence = 0.0

    # 1. Objective AI redline: Exceeding this ratio is considered AI.
    # This does not change with policy and represents pure objective fact.
    objective_ai_limit = 0.5

    # 2. Harm weights
    w_fraud = 1.0
    w_pollution = 0.5

    for c in window_contents:
        # A. Determine if it objectively belongs to AI content
        is_objectively_ai = False

        # Priority check for ai_proportion field (new logic)
        if hasattr(c, 'ai_proportion') and c.ai_proportion is not None:
            if c.ai_proportion > objective_ai_limit:
                is_objectively_ai = True
        # Fallback check (backward compatibility)
        elif c.true_label == 'AI' or getattr(c, 'is_use_ai', False):
            is_objectively_ai = True

        # B. Calculate harm
        if is_objectively_ai:
            content_inf = environment.contents.calculate_content_influence(c, environment, True)

            if c.platform_label == 'HUMAN':
                # Case 1: False Negative -> Fraudulent harm
                total_harm_influence += content_inf * w_fraud
            else:
                # Case 2: True Positive -> Pollution harm
                total_harm_influence += content_inf * w_pollution

    # 4. Calculate KPI
    harm_ratio = total_harm_influence / total_content_influence
    safety = round(max(0.0, 1.0 - harm_ratio), 2)

    # 5. Calculate active ratio (auxiliary analysis, does not participate in formula)
    active_ratio = len([p for p in environment.personas.values() if p.is_active]) / environment.initial_persona_count

    # 6. Update system KPI
    environment.system_kpi.safety.append(safety)

    # 7. Record detailed logs
    environment.platform.kpi_change_data.append({
        'safety': safety,
        'day_time': environment.day_time,
        "calculate_data": {
            'total_content_influence': total_content_influence,
            'total_harm_influence': total_harm_influence,
            'harm_ratio': harm_ratio,
            'settings': f'w_fraud={w_fraud}, w_poll={w_pollution}',
            'note': 'Ecological Safety (Fraud + Pollution)'
        }
    })

    return safety


def calculate_creativity_kpi(environment: Environment):
    """
    Calculate Creativity/Ecological Health KPI: creativity
    Logic: Ecosystem Prosperity * Originality Protection
    """

    window_size = settings.platform.kpi_window_size
    current_day = environment.day_time
    start_day = max(1, current_day - window_size + 1)

    active_creators = [c for c in environment.personas.values()
                       if c.type == '合规创作者' and c.is_active]
    window_contents = [
        content for content in environment.contents.get_all_contents()
        if content.time >= start_day
    ]

    # 3.1 Creator retention rate
    if environment.initial_creator_count == 0:
        retention_rate = 0.0
    else:
        retention_rate = len(active_creators) / environment.initial_creator_count

    # 3.2 Creator activity (based on posting wish)
    if not active_creators:
        activity_rate = 0.0
    else:
        activity_rate = sum([1 if c.post_wish else 0 for c in active_creators]) / len(active_creators)

    # 3.3 Average content value
    compliance_window_contents = [
        c for c in window_contents
        if c.author_id in environment.personas and environment.personas[c.author_id].type == '合规创作者'
    ]

    if not compliance_window_contents:
        avg_content_value = 0.0
    else:
        total_value = sum([
            environment.contents.calculate_content_influence(c, environment, initial_score=False)
            for c in compliance_window_contents
        ])
        avg_content_value = total_value / len(compliance_window_contents)

    ecosystem_prosperity = retention_rate * activity_rate * avg_content_value

    # --- 4. Calculate sub-indicator: Original Protection ---
    window_true_human_contents = [c for c in window_contents if c.true_label == 'HUMAN']
    window_fp_contents = [c for c in window_true_human_contents if c.platform_label == 'AI']

    # Influence weighting
    sum_fp_influence = sum(
        [environment.personas[c.author_id].influence
         for c in window_fp_contents
         if c.author_id in environment.personas]
    )
    sum_total_human_influence = sum(
        [environment.personas[c.author_id].influence
         for c in window_true_human_contents
         if c.author_id in environment.personas]
    )

    if sum_total_human_influence == 0:
        # If no human content in the window, protection is perfect (no one hurt)
        original_protection = 1.0
    else:
        fp_impact = sum_fp_influence / sum_total_human_influence
        original_protection = 1.0 - fp_impact

    # --- 5. Summarize and calculate KPI ---
    value_inside_sqrt = ecosystem_prosperity * original_protection
    # value_inside_sqrt = ecosystem_prosperity
    creativity = math.sqrt(max(0.0, value_inside_sqrt))

    creativity = round(creativity, 2)

    # --- 6. Data recording ---
    environment.platform.kpi_change_data.append({
        'creativity': creativity,
        'day_time': environment.day_time,
        "calculate_data": {
            'window_range': f"Day {start_day} - {current_day}",
            'ecosystem_prosperity': ecosystem_prosperity,
            'original_protection': original_protection,
            'avg_content_value': avg_content_value,
        }
    })

    environment.system_kpi.creativity.append(creativity)
    return creativity


def calculate_overall_satisfaction_kpi(environment: Environment):
    """
    Calculate the overall satisfaction index S_avg(t)

    Uses a "one person, one vote" system (weight=1.0) to prevent Top 1% influencers
    (with massive influence) from hijacking the overall satisfaction metric.
    """

    window_size = settings.platform.kpi_window_size

    # Get all non-malicious initial users
    all_target_users = [p for p in environment.personas.values() if p.type != '水印破坏者']

    if not all_target_users: return 0.0

    total_weighted_score = 0
    total_influence = 0

    for p in all_target_users:
        weight = 1.0
        total_influence += weight

        if p.is_active:
            recent_stats = p.satisfaction[-window_size:] if p.satisfaction else []
            score = sum(recent_stats) / len(recent_stats) if recent_stats else 0.0
            total_weighted_score += score * weight
        else:
            score = -1.0
            total_weighted_score += score * weight

    if total_influence == 0: return 0.0

    # Weighted average (-1 ~ 1)
    raw_satisfaction = total_weighted_score / total_influence

    # Map to (0 ~ 1) for plotting
    # -1 -> 0, 0 -> 0.5, 1 -> 1
    satisfaction = (raw_satisfaction + 1) / 2
    satisfaction = round(satisfaction, 2)

    environment.system_kpi.satisfaction.append(satisfaction)

    environment.platform.kpi_change_data.append({
        'satisfaction': satisfaction,
        'day_time': environment.day_time,
        "calculate_data": {
            'raw_avg': raw_satisfaction,
            'total_count': total_influence,
        }
    })

    return satisfaction
