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

                    
    window_contents = [c for c in environment.contents.get_all_contents()
                       if c.time >= start_day]

                    
                                                  
    total_content_influence = sum([
        environment.contents.calculate_content_influence(c, environment, True)
        for c in window_contents
    ])

                  
    if total_content_influence == 0:
        environment.system_kpi.safety.append(1.0)
        return 1.0

                                            
    total_harm_influence = 0.0

                    
                                           
    objective_ai_limit = 0.5

                    
                                                    
                                                            
    w_fraud = 1.0
    w_pollution = 0.5

    for c in window_contents:
                            
        is_objectively_ai = False

                                          
        if hasattr(c, 'ai_proportion') and c.ai_proportion is not None:
            if c.ai_proportion > objective_ai_limit:
                is_objectively_ai = True
                       
        elif c.true_label == 'AI' or getattr(c, 'is_use_ai', False):
            is_objectively_ai = True

                 
        if is_objectively_ai:
            content_inf = environment.contents.calculate_content_influence(c, environment, True)

            if c.platform_label == 'HUMAN':
                                                    
                total_harm_influence += content_inf * w_fraud
            else:
                                                   
                                        
                total_harm_influence += content_inf * w_pollution

               
                                                 
    harm_ratio = total_harm_influence / total_content_influence
                            
    safety = round(max(0.0, 1.0 - harm_ratio), 2)

                             
    active_ratio = len([p for p in environment.personas.values() if p.is_active]) / environment.initial_persona_count

                 
    environment.system_kpi.safety.append(safety)

               
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

                         
    window_size = settings.platform.kpi_window_size
    current_day = environment.day_time
    start_day = max(1, current_day - window_size + 1)

                       
    active_creators = [c for c in environment.personas.values()
                       if c.type == '合规创作者' and c.is_active]
    window_contents = [
        content for content in environment.contents.get_all_contents()
        if content.time >= start_day
    ]

                                                   

                
    if environment.initial_creator_count == 0:
        retention_rate = 0.0
    else:
        retention_rate = len(active_creators) / environment.initial_creator_count

                         
    if not active_creators:
        activity_rate = 0.0
    else:
        activity_rate = sum([1 if c.post_wish else 0 for c in active_creators]) / len(active_creators)

                
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

                                                  
    window_true_human_contents = [c for c in window_contents if c.true_label == 'HUMAN']
    window_fp_contents = [c for c in window_true_human_contents if c.platform_label == 'AI']

           
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
                                 
        original_protection = 1.0
    else:
        fp_impact = sum_fp_influence / sum_total_human_influence
        original_protection = 1.0 - fp_impact

                         
    value_inside_sqrt = ecosystem_prosperity * original_protection
                                              
    creativity = math.sqrt(max(0.0, value_inside_sqrt))

    creativity = round(creativity, 2)

                     
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

                   
    raw_satisfaction = total_weighted_score / total_influence

                      
                               
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