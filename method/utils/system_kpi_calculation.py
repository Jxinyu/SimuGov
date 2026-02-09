from method.environment import Environment
from config import settings

import math


def calculate_safety_kpi(environment: Environment):
    """
    计算安全性 KPI: safety (Ecological & Objective View / 生态客观视角)

    【核心逻辑重构】：
    从单纯的“反欺诈安全”（只惩罚漏报）升级为“生态安全”（惩罚所有挤占人类生态位的AI）。
    即使平台正确标注了AI内容，如果AI内容泛滥成灾，安全性指标也应当下降，反映人类创作者的生存空间被压缩。
    """
    window_size = settings.platform.kpi_window_size
    start_day = max(1, environment.day_time - window_size + 1)

    # 1. 获取窗口期内的所有内容
    window_contents = [c for c in environment.contents.get_all_contents()
                       if c.time >= start_day]

    # 2. 计算分母：总内容影响力
    # 使用 initial_score=True 捕捉原始流量，反映该内容在社区中的实际存在感
    total_content_influence = sum([
        environment.contents.calculate_content_influence(c, environment, True)
        for c in window_contents
    ])

    # 如果没内容，视作绝对安全
    if total_content_influence == 0:
        environment.system_kpi.safety.append(1.0)
        return 1.0

    # 3. 计算分子：综合生态危害 (Total Ecological Harm)
    total_harm_influence = 0.0

    # === 核心参数配置 ===
    # 1. 客观AI红线：超过此比例即视为AI，不随政策变化，代表纯粹的客观事实
    objective_ai_limit = 0.5

    # 2. 危害权重 (关键修改)
    # w_fraud (欺诈危害): 1.0 -> 漏报。这是最严重的，不仅挤占生态，还破坏信任。
    # w_pollution (污染危害): 0.5 -> 诚实的垃圾。即使标了AI，也在视觉和流量上构成了污染。
    w_fraud = 1.0
    w_pollution = 0.5

    for c in window_contents:
        # A. 判定是否客观上属于 AI 内容
        is_objectively_ai = False

        # 优先检查是否存在 ai_proportion 字段 (新版逻辑)
        if hasattr(c, 'ai_proportion') and c.ai_proportion is not None:
            if c.ai_proportion > objective_ai_limit:
                is_objectively_ai = True
        # 兜底检查 (旧版逻辑兼容)
        elif c.true_label == 'AI' or getattr(c, 'is_use_ai', False):
            is_objectively_ai = True

        # B. 计算危害
        if is_objectively_ai:
            content_inf = environment.contents.calculate_content_influence(c, environment, True)

            if c.platform_label == 'HUMAN':
                # 情况 1: 漏报 (False Negative) -> 欺诈性危害
                total_harm_influence += content_inf * w_fraud
            else:
                # 情况 2: 正报 (True Positive) -> 污染性危害
                # 平台正确识别了它，但它依然占据了版面和注意力
                total_harm_influence += content_inf * w_pollution

    # 4. 计算 KPI
    # 即使全站都是正确标注的 AI，Safety 最多也只有 1.0 - 0.5 = 0.5
    harm_ratio = total_harm_influence / total_content_influence
    # 截断防止出现负数（理论上不会，但为了健壮性）
    safety = round(max(0.0, 1.0 - harm_ratio), 2)

    # 5. 计算活跃比 (用于辅助分析，不参与公式)
    active_ratio = len([p for p in environment.personas.values() if p.is_active]) / environment.initial_persona_count

    # 6. 更新系统 KPI
    environment.system_kpi.safety.append(safety)

    # 7. 记录详细日志
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
    计算创造力/生态健康度 KPI: creativity
    逻辑：生态繁荣度 * 原创保护度
    """

    # --- 1. 确定滑动窗口范围 ---
    window_size = settings.platform.kpi_window_size
    current_day = environment.day_time
    start_day = max(1, current_day - window_size + 1)

    # --- 2. 准备基础数据 ---
    active_creators = [c for c in environment.personas.values()
                       if c.type == '合规创作者' and c.is_active]
    window_contents = [
        content for content in environment.contents.get_all_contents()
        if content.time >= start_day
    ]

    # --- 3. 计算子指标：生态繁荣度 (Ecosystem Prosperity) ---

    # 3.1 创作者留存率
    if environment.initial_creator_count == 0:
        retention_rate = 0.0
    else:
        retention_rate = len(active_creators) / environment.initial_creator_count

    # 3.2 创作者活跃度 (基于发布意愿)
    if not active_creators:
        activity_rate = 0.0
    else:
        activity_rate = sum([1 if c.post_wish else 0 for c in active_creators]) / len(active_creators)

    # 3.3 平均内容价值
    compliance_window_contents = [
        c for c in window_contents
        if c.author_id in environment.personas and environment.personas[c.author_id].type == '合规创作者'
    ]

    if not compliance_window_contents:
        # 【修正】如果没有合规内容，价值就是0，不要给0.5的同情分，否则无法反映生态崩塌
        avg_content_value = 0.0
    else:
        total_value = sum([
            environment.contents.calculate_content_influence(c, environment, initial_score=False)
            for c in compliance_window_contents
        ])
        avg_content_value = total_value / len(compliance_window_contents)

    ecosystem_prosperity = retention_rate * activity_rate * avg_content_value

    # --- 4. 计算子指标：原创保护度 (Original Protection) ---
    window_true_human_contents = [c for c in window_contents if c.true_label == 'HUMAN']
    window_fp_contents = [c for c in window_true_human_contents if c.platform_label == 'AI']

    # 影响力加权
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
        # 窗口期内没有人类发文，保护度为满分（没人受伤）
        original_protection = 1.0
    else:
        fp_impact = sum_fp_influence / sum_total_human_influence
        original_protection = 1.0 - fp_impact

    # --- 5. 汇总计算 KPI ---
    value_inside_sqrt = ecosystem_prosperity * original_protection
    # value_inside_sqrt = ecosystem_prosperity
    creativity = math.sqrt(max(0.0, value_inside_sqrt))

    creativity = round(creativity, 2)

    # --- 6. 数据记录 ---
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
    计算总体满意度指数 S_avg(t)

    【核心逻辑修改】：
    使用“一人一票”制（weight=1.0），防止Top 1%的大V（影响力极大）绑架整个满意度指标。
    """

    window_size = settings.platform.kpi_window_size

    # 获取所有非恶意的初始用户
    all_target_users = [p for p in environment.personas.values() if p.type != '水印破坏者']

    if not all_target_users: return 0.0

    total_weighted_score = 0
    total_influence = 0

    for p in all_target_users:
        # 【修正】权重设为 1.0，体现普惠性
        # 如果你想保留一点影响力的作用，可以用 math.log(p.influence + 1)
        weight = 1.0
        total_influence += weight

        if p.is_active:
            # 活跃用户：取最近窗口期的平均值
            recent_stats = p.satisfaction[-window_size:] if p.satisfaction else []
            score = sum(recent_stats) / len(recent_stats) if recent_stats else 0.0
            total_weighted_score += score * weight
        else:
            # 流失用户：强制记为 -1.0 (极度失望)
            # 这会让流失率高的策略 KPI 迅速崩盘
            score = -1.0
            total_weighted_score += score * weight

    if total_influence == 0: return 0.0

    # 加权平均 (-1 ~ 1)
    raw_satisfaction = total_weighted_score / total_influence

    # 映射到 (0 ~ 1) 用于绘图
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