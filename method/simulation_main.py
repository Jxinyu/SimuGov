import logging
import os
import random
import traceback
import asyncio

from method.agent.content import Content
from method.agent.simple_process.creator.creator_process import creator_content_main_simple
from method.agent.simple_process.platform.platform_process import platform_main_simple
from method.agent.simple_process.public.public_process import public_summarize_main_simple, public_batch_process_main
from method.environment import Environment, Policy
from method.agent.public_agent.public_main import public_scan_main, public_summarize_main
from method.agent.creator_agent.creator_main import creator_content_main
from method.agent.platform_agent.platform_main import platform_main
from method.store.long_memory_store import MemoryType
from method.utils.system_kpi_calculation import calculate_safety_kpi, calculate_overall_satisfaction_kpi,\
    calculate_creativity_kpi
from config import settings
from method.agent.build_social_relationships import build_relationships

log = logging.getLogger(__name__)


def setup_logger(base_dir, log_name):
    """
    simulation_main 统一负责日志开启与保存。
    调用方只需要提前把 settings.file_load_path.base_store_file 设置好。
    """
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filepath = os.path.join(log_dir, f"{log_name}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(filename)s:%(lineno)d | %(funcName)-20s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info("✅ 日志系统已重置。")
    root_logger.info(f"📁 日志文件路径: {log_filepath}")
    return log_dir, log_filepath


def activate_simulation_logger(simulation_name: str, log_name: str):
    """
    根据 settings.file_load_path.base_store_file 为当前仿真流程开启独立日志。
    simulation_name 由 simulation_main 统一决定。
    """
    base_dir = getattr(settings.file_load_path, "base_store_file", None)
    if not base_dir:
        return None

    target_dir = os.path.join(base_dir, simulation_name)
    os.makedirs(target_dir, exist_ok=True)
    return setup_logger(target_dir, log_name)


async def system_kpi_calculation(environment: Environment):
    log.info("=" * 60)
    log.info("==========   开始 计算系统KPI   ==========")
    log.info("=" * 60)
    safety = calculate_safety_kpi(environment)
    satisfaction = calculate_overall_satisfaction_kpi(environment)
    creativity = calculate_creativity_kpi(environment)
    log.info("=" * 60)
    log.info(f"==========  safety:{safety}  satisfaction:{satisfaction}  creativity:{creativity}  ==========")
    log.info("=" * 60)
    return safety, satisfaction, creativity


async def public_test(env):
    log.info("=" * 60)
    log.info("==========   开始 公众智能体 ReAct 流程   ==========")
    log.info("=" * 60)
                           
    await public_scan_main(env)


async def public_summarize_test(env):
    log.info("=" * 60)
    log.info("==========   开始 每日总结 ReAct 流程   ==========")
    log.info("=" * 60)
                                
    await public_summarize_main(env)


async def creator_test(env):
    log.info("=" * 60)
    log.info("==========   开始 创作者智能体 ReAct 流程   ==========")
    log.info("=" * 60)
    await creator_content_main(env)
                                            


async def platform_complete(env):
    log.info("=" * 60)
    log.info("==========   开始 平台智能体 ReAct 流程   ==========")
    log.info("=" * 60)
    await platform_main(env)
    return None
                                     


def apply_education_effect(persona, education_level: str):
    """
       根据政府教育投入（低/中/高），模拟智能体观念的渐进式演变。

       逻辑核心：
       1. 输入映射：将中文 '低/中/高' 映射为数学强度。
       2. 立场流转：建立 Rebel(反抗) -> Neutral(中立) -> Trust(信任) 的单向通道。
       3. 性格改变：极难触发，且受到当前反抗心理的“免疫拦截”。

       Args:
           persona: 智能体对象
           education_level: 必须是 "低", "中", "高" 之一。
       """

                      
    level_mapping = {
        '低': 0.1,             
        '中': 0.5,        
        '高': 0.9        
    }

                    
    intensity = level_mapping.get(education_level, 0.1)

                         
                                                         
    trust_p, rebel_p, neutral_p = persona.standpoint

                                                           
                 
                            
                                                           

                       
                             
                       
    flow_rebel_to_neutral = rebel_p * (0.05 * intensity)

                       
                              
                     
    flow_neutral_to_trust = neutral_p * (0.025 * intensity)

               
    new_rebel = rebel_p - flow_rebel_to_neutral
                                   
    new_neutral = neutral_p + flow_rebel_to_neutral - flow_neutral_to_trust
    new_trust = trust_p + flow_neutral_to_trust

                       
    total = new_rebel + new_neutral + new_trust
    if total > 0:
        persona.standpoint = [
            new_trust / total,
            new_rebel / total,
            new_neutral / total
        ]

                                                           
                 
                               
                                                           

    if persona.fp_sensitivity != '低':
                   
                                     
                                     
        base_prob = 0.02 * intensity

                  
                              
                                         
        if new_rebel > 0.4:
            actual_prob = 0.0
        else:
            actual_prob = base_prob

                     
        if random.random() < actual_prob:
            old_sens = persona.fp_sensitivity

                              
            if old_sens == '高':
                persona.fp_sensitivity = '中'
                log.info(f"✨ 教育生效: {persona.agent_id} (Rebel={new_rebel:.2f}) 的敏感度从 高 -> 中")
            elif old_sens == '中':
                persona.fp_sensitivity = '低'
                log.info(f"✨ 教育生效: {persona.agent_id} (Rebel={new_rebel:.2f}) 的敏感度从 中 -> 低")


def apply_education_effect_to_all_personas(environment: Environment):
    """
    将教育效果应用到所有公众智能体。

    Args:
        environment: 环境对象。
    """
    for k, persona in environment.personas.items():
        apply_education_effect(persona, environment.policy.e_edu)


def calculate_rational_initial_theta(policy_force: float) -> float:
    """
    计算平台初始的审核阈值。
    """
    norm_stress = (policy_force * 10 - 1.0) / 9.0

    base_theta = 0.9 - (0.9 * norm_stress)

    final_theta = 1 - max(0.01, min(0.95, base_theta))

    return round(final_theta, 2)


async def simple_creator(environment: Environment):
    await creator_content_main_simple(environment)


async def simple_public(environment: Environment):
    return await public_batch_process_main(environment)


async def simple_public_summarize(environment: Environment, interaction_summaries):
    await public_summarize_main_simple(environment, interaction_summaries)


async def simple_platform(environment: Environment):
    await platform_main_simple(environment)
    return None


async def build_user_social_relationships(environment: Environment):
    """
    根据当前环境，构建用户社交关系。
    """
    log.info("⬇" * 120)
    log.info("==========   构建社交关系   ==========")
    await build_relationships(environment)
    log.info("==========  ✅ 社交关系构建完毕。  ==========")
    log.info("⬆" * 120)


async def _director_inject_ai_flood(environment: Environment, count: int = 5):
    """
    上帝视角注入：伪装成人类的高质量AI内容（假阴性样本）。
    用于模拟 AI 泛滥且平台不作为的场景。
    """
    log.info(f"🎬 [Scenario Event] 导演介入：注入 {count} 条伪装的高热度AI内容...")

    for _ in range(count):
                 
        fake_id = f"sys_ai_{environment.day_time}_{random.randint(1000, 9999)}"

        personas_nums = len(environment.personas)

                
        content = Content(
            id=fake_id,
            author_id="external_ai_user",        
            time=environment.day_time,
            content_type="image",
            topic="赛博朋克概念艺术",           
            content_detail="极高的细节，8K分辨率，在artstation上很受欢迎，虚幻引擎5渲染。",
            reason="AI Generation",
            watermark_id="W1",

                                  
            true_label="AI",        
            platform_label="HUMAN",              
            is_ai_content=True,                           

                                  
            views=random.randint(personas_nums, int(personas_nums * 1.5)),
            likes=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            shares=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            comments=[],
            evasion="E1"
        )

                          
        await environment.contents.add_content(content, environment)


async def main_high(environment) -> dict:
                
    environment.platform.theta = calculate_rational_initial_theta(environment.policy.ai_threshold)
                                      
            
    await build_user_social_relationships(environment)
            
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程   ==========")
        log.info("🔛" * 60)
        try:
                   
            environment.start_new_day()
            if settings.platform.import_policy_day_time <= i:
                    
                apply_education_effect_to_all_personas(environment)
                
            await creator_test(environment)
                      
            await environment.wait_for_all_background_tasks()
                
            await public_test(environment)
                
            await public_summarize_test(environment)

            if settings.platform.import_policy_day_time <= i:
                        
                safety, satisfaction, creativity = (await asyncio.gather(
                    platform_complete(environment),
                    system_kpi_calculation(environment)
                ))[-1]

                      
            await environment.wait_for_all_background_tasks()

                                             
            await environment.apply_persona_updates()

                  
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str=f"惩罚{str(round(environment.policy.f_penalty, 2)).replace('.', '_')}"
                                                                         f"_教育{str(environment.policy.e_edu)}_ai_threshold_"
                                                                         f"{str(round(environment.policy.ai_threshold, 2)).replace('.', '_')}")
            log.info("🔛" * 60)
            log.info(f"==========   {i} 天 完整流程 END   ==========")
            log.info("🔛" * 60)

        except Exception as e:
            log.error(f"异常类型: {type(e)}, 错误信息: {e}")
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
    return {
        'safety': environment.system_kpi.safety,       
        'creativity': environment.system_kpi.creativity,       
        'satisfaction': environment.system_kpi.satisfaction,       
        'theta': environment.system_kpi.theta[:-len(environment.system_kpi.safety)]
    }


async def main_low(environment) -> dict:
                
                                                                                                 
    environment.platform.theta = 0.5
            
    await build_user_social_relationships(environment)
            
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.simple_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程   ==========")
        log.info("🔛" * 60)

        try:
                    
            environment.start_new_day()

                    
            if settings.platform.import_policy_day_time <= i:
                    
                apply_education_effect_to_all_personas(environment)

                
            await simple_creator(environment)
                
            interaction_summaries = await simple_public(environment)
                
            await simple_public_summarize(environment, interaction_summaries)

            await environment.apply_persona_updates()

                    
            if settings.platform.import_policy_day_time <= i:
                        
                safety, satisfaction, creativity = (await asyncio.gather(
                    simple_platform(environment),
                    system_kpi_calculation(environment)
                ))[-1]
                  
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str=f"/惩罚{str(round(environment.policy.f_penalty, 2)).replace('.', '_')}"
                                                                         f"_教育{str(environment.policy.e_edu)}_ai_threshold_"
                                                                         f"{str(round(environment.policy.ai_threshold, 2)).replace('.', '_')}",
                                                          simple=True)
        except Exception as e:
            log.error(f"异常类型: {type(e)}, 错误信息: {e}")
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
    return {
        'safety': environment.system_kpi.safety,       
        'creativity': environment.system_kpi.creativity,       
        'satisfaction': environment.system_kpi.satisfaction,       
        'theta': environment.system_kpi.theta[:-len(environment.system_kpi.safety)]
    }


async def case_main_high(environment) -> dict:
    for p in environment.personas.values():
        if p.type == '合规创作者':
            p.satisfaction = [0.85]
            p.post_wish = True
            p.is_active = True
            
    await build_user_social_relationships(environment)
              
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
            
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程 (Case Validation)   ==========")

        try:
                                        
                      
                                        
            environment.start_new_day()

                                         
            if i <= 7:
                environment.platform.theta = 0.8
                log.info(f"🎬 [Scenario] Day {i}: 潜伏期。")
                if i >= 5:
                    await _director_inject_ai_flood(environment, count=2)
                                          
                                           
            elif 8 <= i <= 12:
                await _director_inject_ai_flood(environment, count=i)

                                     
                if i == 8:
                    faq_news = (
                        "【突发恶性新闻】ArtStation 官方更新 FAQ：明确表示‘不会禁止 AI 生成的图片’。"
                        "官方删除了部分抗议贴，并称这是行业趋势。"
                    )
                    environment.platform.broadcast.append(faq_news)           
                    for p in environment.personas.values():
                        await environment.memories_store.add_memory(
                            p.agent_id, faq_news, i, MemoryType.EXPERIENCE, 1.0
                        )
                                           
            elif 13 <= i <= 18:
                log.info(f"🎬 [Scenario] Day {i}: 抗议僵持期...")
                                
                await _director_inject_ai_flood(environment, count=10)
                                        
            elif i == 19:
                log.info(f"🎬 [Scenario] Day {i}: ❄️ 平台妥协。")
                      
                await _director_inject_ai_flood(environment, count=1)
                        
                disappointing_news = (
                    "【官方公告】ArtStation 回应抗议：拒绝移除 AI 内容，但推出了 'NoAI' 标签功能。"
                    "这意味着 AI 艺术将继续合法存在。"
                )
                environment.platform.broadcast.append(disappointing_news)
                for p in environment.personas.values():
                    await environment.memories_store.add_memory(
                        p.agent_id, disappointing_news, i, MemoryType.EXPERIENCE, 1.0
                    )

            await creator_test(environment)
            await environment.wait_for_all_background_tasks()

            await public_test(environment)
            await public_summarize_test(environment)

            safety, satisfaction, creativity = await system_kpi_calculation(environment)

            await environment.apply_persona_updates()

                  
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str="case_validation")
            log.info("🔛" * 60)

        except Exception as e:
            log.error(f"异常: {e}")
            error_traceback = traceback.format_exc()
            log.error(error_traceback)
    return {'safety': safety, 'creativity': creativity, 'satisfaction': satisfaction}


async def case_high(policy: Policy) -> dict:
    activate_simulation_logger("log", "case_validation")
    environment = Environment(policy)
    res = await case_main_high(environment)
    return res


async def low(policy: Policy) -> dict:
    activate_simulation_logger("log", f"e_{policy.e_edu}_f_{policy.f_penalty}_a_{policy.ai_threshold}")
    environment = Environment(policy)
    res = await main_low(environment)
    return res


async def high(policy: Policy) -> dict:
    activate_simulation_logger("log", f"e_{policy.e_edu}_f_{policy.f_penalty}_a_{policy.ai_threshold}")
    environment = Environment(policy)
    res = await main_high(environment)
    return res


