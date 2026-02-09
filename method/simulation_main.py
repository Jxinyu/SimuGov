import logging
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
from method.utils.system_kpi_calculation import calculate_safety_kpi, calculate_overall_satisfaction_kpi, \
    calculate_creativity_kpi
from method.utils.calculation_token_nums import clear_token_csv_file
from config import settings
from method.agent.build_social_relationships import build_relationships

log = logging.getLogger(__name__)


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
    # public_scan_main(env)
    await public_scan_main(env)


async def public_summarize_test(env):
    log.info("=" * 60)
    log.info("==========   开始 每日总结 ReAct 流程   ==========")
    log.info("=" * 60)
    # public_summarize_main(env)
    await public_summarize_main(env)


async def creator_test(env):
    log.info("=" * 60)
    log.info("==========   开始 创作者智能体 ReAct 流程   ==========")
    log.info("=" * 60)
    await creator_content_main(env)
    # asyncio.run(creator_content_main(env))


async def platform_complete(env):
    log.info("=" * 60)
    log.info("==========   开始 平台智能体 ReAct 流程   ==========")
    log.info("=" * 60)
    await platform_main(env)
    return None
    # asyncio.run(platform_main(env))


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

    # 1. 参数映射：明确定义三档强度
    level_mapping = {
        '低': 0.1,  # 几乎无效，自然演化
        '中': 0.5,  # 温和影响
        '高': 0.9  # 强力干预
    }

    # 获取强度系数，默认为 '低'
    intensity = level_mapping.get(education_level, 0.1)

    # 获取当前立场 [信任, 反抗, 中立]
    # 假设 persona.standpoint 存储顺序为 [trust, rebel, neutral]
    trust_p, rebel_p, neutral_p = persona.standpoint

    # =====================================================
    # 机制一：渐进式立场流转
    # 逻辑：反抗者先变理智(中立)，理智者再变信任
    # =====================================================

    # 1. 去激进化: 反抗 -> 中立
    # 即使是高投入，每天也只能转化当前反抗值的 5%
    # 理由：消除敌意比建立信任容易一点点
    flow_rebel_to_neutral = rebel_p * (0.05 * intensity)

    # 2. 建立信任: 中立 -> 信任
    # 这是一个更加漫长的过程，转化率更低 (2.5%)
    # 理由：从中立变成“粉丝”非常难
    flow_neutral_to_trust = neutral_p * (0.025 * intensity)

    # 3. 执行流转计算
    new_rebel = rebel_p - flow_rebel_to_neutral
    # 中立派 = 原有 + 新来的(从反抗) - 走的(去信任)
    new_neutral = neutral_p + flow_rebel_to_neutral - flow_neutral_to_trust
    new_trust = trust_p + flow_neutral_to_trust

    # 4. 重新归一化 (防止浮点误差)
    total = new_rebel + new_neutral + new_trust
    if total > 0:
        persona.standpoint = [
            new_trust / total,
            new_rebel / total,
            new_neutral / total
        ]

    # =====================================================
    # 机制二：高阻尼心理脱敏
    # 降低 fp_sensitivity (误伤敏感度)
    # =====================================================

    if persona.fp_sensitivity != '低':
        # 基础变异概率：极低
        # 高投入(0.9) -> 0.018 (1.8% 概率)
        # 低投入(0.1) -> 0.002 (0.2% 概率)
        base_prob = 0.02 * intensity

        # 【心理防御机制】
        # 如果当前的反抗值依然很高 (> 0.4)
        # 说明该用户处于"防御/敌对模式"，教育会被视为洗脑，完全无效。
        if new_rebel > 0.4:
            actual_prob = 0.0
        else:
            actual_prob = base_prob

        # 掷骰子决定是否改变性格
        if random.random() < actual_prob:
            old_sens = persona.fp_sensitivity

            # 降级逻辑：高 -> 中 -> 低
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


async def main_complete(environment) -> dict:
    # 注入平台初始审核阈值
    environment.platform.theta = calculate_rational_initial_theta(environment.policy.ai_threshold)
    # environment.platform.theta = 0.5
    # 构建社交关系
    await build_user_social_relationships(environment)
    # 初始化KPI
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程   ==========")
        log.info("🔛" * 60)
        clear_token_csv_file()  # 清空token.csv文件
        try:
            # 初始化参数
            environment.start_new_day()
            if settings.platform.import_policy_day_time <= i:
                # 教育
                apply_education_effect_to_all_personas(environment)
            # 创作
            await creator_test(environment)
            # 等待后台任务结束
            await environment.wait_for_all_background_tasks()
            # 浏览
            await public_test(environment)
            # 总结
            await public_summarize_test(environment)

            if settings.platform.import_policy_day_time <= i:
                # 平台+kpi
                safety, satisfaction, creativity = (await asyncio.gather(
                    platform_complete(environment),
                    system_kpi_calculation(environment)
                ))[-1]

            # 等待后台任务结束
            await environment.wait_for_all_background_tasks()

            # 5. 在 KPI 计算完、且在进入下一天之前，正式应用状态变更
            await environment.apply_persona_updates()

            # 导出记忆
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
        'safety': environment.system_kpi.safety,  # 安全性
        'creativity': environment.system_kpi.creativity,  # 创造力
        'satisfaction': environment.system_kpi.satisfaction,  # 满意度
        'theta': environment.system_kpi.theta[:-len(environment.system_kpi.safety)]
    }


async def main_simple(environment) -> dict:
    # 注入平台初始审核阈值
    # environment.platform.theta = calculate_rational_initial_theta(environment.policy.f_penalty)
    environment.platform.theta = 0.5
    # 构建社交关系
    await build_user_social_relationships(environment)
    # 初始化KPI
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.simple_run_days + 1):
        clear_token_csv_file()  # 清空token.csv文件
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程   ==========")
        log.info("🔛" * 60)
        try:
            # 更新每日参数
            environment.start_new_day()

            # 导入策略时间
            if settings.platform.import_policy_day_time <= i:
                # 教育
                apply_education_effect_to_all_personas(environment)

            # 创作
            await simple_creator(environment)
            # 浏览
            interaction_summaries = await simple_public(environment)
            # 总结
            await simple_public_summarize(environment, interaction_summaries)

            await environment.apply_persona_updates()

            # 导入策略时间
            if settings.platform.import_policy_day_time <= i:
                # 审核+kpi
                safety, satisfaction, creativity = (await asyncio.gather(
                    simple_platform(environment),
                    system_kpi_calculation(environment)
                ))[-1]
            # 导出记忆
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str=f"简化/惩罚{str(round(environment.policy.f_penalty, 2)).replace('.', '_')}"
                                                                         f"_教育{str(environment.policy.e_edu)}_ai_threshold_"
                                                                         f"{str(round(environment.policy.ai_threshold, 2)).replace('.', '_')}",
                                                          simple=True)
        except Exception as e:
            log.error(f"异常类型: {type(e)}, 错误信息: {e}")
            error_traceback = traceback.format_exc()
            log.error("完整的堆栈跟踪信息如下:\n" + error_traceback)
    return {
        'safety': environment.system_kpi.safety,  # 安全性
        'creativity': environment.system_kpi.creativity,  # 创造力
        'satisfaction': environment.system_kpi.satisfaction,  # 满意度
        'theta': environment.system_kpi.theta[:-len(environment.system_kpi.safety)]
    }


async def simple(policy: Policy) -> dict:
    environment = Environment(policy)
    return await main_simple(environment)


async def complete(policy: Policy) -> dict:
    environment = Environment(policy)
    return await main_complete(environment)


async def _director_inject_ai_flood(environment: Environment, count: int = 5):
    """
    上帝视角注入：伪装成人类的高质量AI内容（假阴性样本）。
    用于模拟 AI 泛滥且平台不作为的场景。
    """
    log.info(f"🎬 [Scenario Event] 导演介入：注入 {count} 条伪装的高热度AI内容...")

    for _ in range(count):
        # 伪造一个 ID
        fake_id = f"sys_ai_{environment.day_time}_{random.randint(1000, 9999)}"

        personas_nums = len(environment.personas)

        # 创建内容对象
        content = Content(
            id=fake_id,
            author_id="external_ai_user",  # 虚拟作者
            time=environment.day_time,
            content_type="image",
            topic="赛博朋克概念艺术",  # 典型AI重灾区
            content_detail="极高的细节，8K分辨率，在artstation上很受欢迎，虚幻引擎5渲染。",
            reason="AI Generation",
            watermark_id="W1",

            # === 关键设定：激怒用户的源头 ===
            true_label="AI",  # 它是AI
            platform_label="HUMAN",  # 平台却说是人（漏报）
            is_ai_content=True,  # 假设你的Content类有这个字段，没有可忽略

            # === 设定高热度，确保必被刷到 ===
            views=random.randint(personas_nums, int(personas_nums * 1.5)),
            likes=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            shares=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            comments=[],
            evasion="E1"
        )

        # 强行插入内容库，绕过正常审核逻辑
        await environment.contents.add_content(content, environment)


async def case_main_complete(environment) -> dict:
    for p in environment.personas.values():
        if p.type == '合规创作者':
            p.satisfaction = [0.85]
            p.post_wish = True
            p.is_active = True
    # 构建社交关系
    await build_user_social_relationships(environment)
    # 初始化KPI容器
    safety, satisfaction, creativity = 0.01, 0.01, 0.01

    # 开始仿真循环
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   开始 {i} 天 完整流程 (Case Validation)   ==========")
        clear_token_csv_file()

        try:
            # ==========================
            # 1. 每日初始化
            # ==========================
            environment.start_new_day()

            # --- 第一幕：潜伏期 (Day 1 - 7) ---
            if i <= 7:
                environment.platform.theta = 0.8
                log.info(f"🎬 [Scenario] Day {i}: 潜伏期。")
                if i >= 5:
                    await _director_inject_ai_flood(environment, count=2)
            # --- 第二幕：爆发期 (Day 8 - 12) ---
            # 目标：通过环境压力（AI刷屏）触发Agent的“逆反心理”
            elif 8 <= i <= 12:
                await _director_inject_ai_flood(environment, count=i)

                # 2. 坏消息广播 (仅在 Day 8)
                if i == 8:
                    faq_news = (
                        "【突发恶性新闻】ArtStation 官方更新 FAQ：明确表示‘不会禁止 AI 生成的图片’。"
                        "官方删除了部分抗议贴，并称这是行业趋势。"
                    )
                    environment.platform.broadcast.append(faq_news)  # 添加到平台广播
                    for p in environment.personas.values():
                        await environment.memories_store.add_memory(
                            p.agent_id, faq_news, i, MemoryType.EXPERIENCE, 1.0
                        )
            # --- 第三幕：僵持期 (Day 13 - 18) ---
            elif 13 <= i <= 18:
                log.info(f"🎬 [Scenario] Day {i}: 抗议僵持期...")
                # 持续高压，测试用户的耐受极限
                await _director_inject_ai_flood(environment, count=10)
            # --- 第四幕：妥协与分流 (Day 19) ---
            elif i == 19:
                log.info(f"🎬 [Scenario] Day {i}: ❄️ 平台妥协。")
                # 减少注入
                await _director_inject_ai_flood(environment, count=1)
                # 广播妥协新闻
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

            # 导出数据
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str="case_validation")
            log.info("🔛" * 60)

        except Exception as e:
            log.error(f"异常: {e}")
            error_traceback = traceback.format_exc()
            log.error(error_traceback)

    return {'safety': safety, 'creativity': creativity, 'satisfaction': satisfaction}


async def case_complete(policy: Policy) -> dict:
    environment = Environment(policy)
    return await case_main_complete(environment)
