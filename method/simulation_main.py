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
    log.info("==========   Start Calculating System KPI   ==========")
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
    log.info("==========   Start Public Agent ReAct Process   ==========")
    log.info("=" * 60)
    # public_scan_main(env)
    await public_scan_main(env)


async def public_summarize_test(env):
    log.info("=" * 60)
    log.info("==========   Start Daily Summary ReAct Process   ==========")
    log.info("=" * 60)
    # public_summarize_main(env)
    await public_summarize_main(env)


async def creator_test(env):
    log.info("=" * 60)
    log.info("==========   Start Creator Agent ReAct Process   ==========")
    log.info("=" * 60)
    await creator_content_main(env)
    # asyncio.run(creator_content_main(env))


async def platform_complete(env):
    log.info("=" * 60)
    log.info("==========   Start Platform Agent ReAct Process   ==========")
    log.info("=" * 60)
    await platform_main(env)
    return None
    # asyncio.run(platform_main(env))


def apply_education_effect(persona, education_level: str):
    """
       Simulate the progressive evolution of agent perspectives based on government education investment (Low/Medium/High).

       Logic Core:
       1. Input Mapping: Map Chinese '低/中/高' to mathematical intensities.
       2. Standpoint Flow: Establish a one-way channel: Rebel -> Neutral -> Trust.
       3. Personality Change: Extremely difficult to trigger and intercepted by current "rebellion psychology" immunity.

       Args:
           persona: Agent object.
           education_level: Must be one of "低", "中", "高".
       """

    # 1. Parameter Mapping: Clearly define three levels of intensity
    level_mapping = {
        '低': 0.1,  # Almost ineffective, natural evolution
        '中': 0.5,  # Moderate influence
        '高': 0.9  # Strong intervention
    }

    # Get intensity coefficient, default is '低'
    intensity = level_mapping.get(education_level, 0.1)

    trust_p, rebel_p, neutral_p = persona.standpoint

    flow_rebel_to_neutral = rebel_p * (0.05 * intensity)

    flow_neutral_to_trust = neutral_p * (0.025 * intensity)

    new_rebel = rebel_p - flow_rebel_to_neutral
    new_neutral = neutral_p + flow_rebel_to_neutral - flow_neutral_to_trust
    new_trust = trust_p + flow_neutral_to_trust

    # 4. Re-normalization (prevent floating point errors)
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

        # Roll dice to decide whether to change personality
        if random.random() < actual_prob:
            old_sens = persona.fp_sensitivity

            # Downgrade logic: 高 -> 中 -> 低
            if old_sens == '高':
                persona.fp_sensitivity = '中'
                log.info(f"✨ Education took effect: {persona.agent_id} (Rebel={new_rebel:.2f})'s sensitivity changed from 高 -> 中")
            elif old_sens == '中':
                persona.fp_sensitivity = '低'
                log.info(f"✨ Education took effect: {persona.agent_id} (Rebel={new_rebel:.2f})'s sensitivity changed from 中 -> 低")


def apply_education_effect_to_all_personas(environment: Environment):
    """
    Apply education effects to all public agents.

    Args:
        environment: Environment object.
    """
    for k, persona in environment.personas.items():
        apply_education_effect(persona, environment.policy.e_edu)


def calculate_rational_initial_theta(policy_force: float) -> float:
    """
    Calculate the initial moderation threshold of the platform.
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
    Build user social relationships based on the current environment.
    """
    log.info("⬇" * 120)
    log.info("==========   Building Social Relationships   ==========")
    await build_relationships(environment)
    log.info("==========  ✅ Social relationship construction complete.  ==========")
    log.info("⬆" * 120)


async def main_complete(environment) -> dict:
    # Inject platform initial moderation threshold
    environment.platform.theta = calculate_rational_initial_theta(environment.policy.ai_threshold)
    # environment.platform.theta = 0.5
    # Build social relationships
    await build_user_social_relationships(environment)
    # Initialize KPI
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   Start Day {i} Complete Process   ==========")
        log.info("🔛" * 60)
        clear_token_csv_file()  # Clear token.csv file
        try:
            # Initialize parameters
            environment.start_new_day()
            if settings.platform.import_policy_day_time <= i:
                # Education
                apply_education_effect_to_all_personas(environment)
            # Creation
            await creator_test(environment)
            # Wait for background tasks to finish
            await environment.wait_for_all_background_tasks()
            # Browsing
            await public_test(environment)
            # Summary
            await public_summarize_test(environment)

            if settings.platform.import_policy_day_time <= i:
                # Platform + KPI
                safety, satisfaction, creativity = (await asyncio.gather(
                    platform_complete(environment),
                    system_kpi_calculation(environment)
                ))[-1]

            # Wait for background tasks to finish
            await environment.wait_for_all_background_tasks()

            # 5. Formally apply status changes after KPI calculation and before starting the next day
            await environment.apply_persona_updates()

            # Export memory
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str=f"Penalty{str(round(environment.policy.f_penalty, 2)).replace('.', '_')}"
                                                                         f"_Education{str(environment.policy.e_edu)}_ai_threshold_"
                                                                         f"{str(round(environment.policy.ai_threshold, 2)).replace('.', '_')}")
            log.info("🔛" * 60)
            log.info(f"==========   Day {i} Complete Process END   ==========")
            log.info("🔛" * 60)

        except Exception as e:
            log.error(f"Exception Type: {type(e)}, Error Message: {e}")
            error_traceback = traceback.format_exc()
            log.error("Full stack trace information is as follows:\n" + error_traceback)
    return {
        'safety': environment.system_kpi.safety,  # Safety
        'creativity': environment.system_kpi.creativity,  # Creativity
        'satisfaction': environment.system_kpi.satisfaction,  # Satisfaction
        'theta': environment.system_kpi.theta[:-len(environment.system_kpi.safety)]
    }


async def main_simple(environment) -> dict:
    # Inject platform initial moderation threshold
    # environment.platform.theta = calculate_rational_initial_theta(environment.policy.f_penalty)
    environment.platform.theta = 0.5
    # Build social relationships
    await build_user_social_relationships(environment)
    # Initialize KPI
    safety, satisfaction, creativity = 0.01, 0.01, 0.01
    for i in range(1, settings.platform.simple_run_days + 1):
        clear_token_csv_file()  # Clear token.csv file
        log.info("🔛" * 60)
        log.info(f"==========   Start Day {i} Complete Process   ==========")
        log.info("🔛" * 60)
        try:
            # Update daily parameters
            environment.start_new_day()

            # Policy import time
            if settings.platform.import_policy_day_time <= i:
                # Education
                apply_education_effect_to_all_personas(environment)

            # Creation
            await simple_creator(environment)
            # Browsing
            interaction_summaries = await simple_public(environment)
            # Summary
            await simple_public_summarize(environment, interaction_summaries)

            await environment.apply_persona_updates()

            # Policy import time
            if settings.platform.import_policy_day_time <= i:
                # Moderation + KPI
                safety, satisfaction, creativity = (await asyncio.gather(
                    simple_platform(environment),
                    system_kpi_calculation(environment)
                ))[-1]
            # Export memory
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str=f"Simple_Penalty{str(round(environment.policy.f_penalty, 2)).replace('.', '_')}"
                                                                         f"_Education{str(environment.policy.e_edu)}_ai_threshold_"
                                                                         f"{str(round(environment.policy.ai_threshold, 2)).replace('.', '_')}",
                                                          simple=True)
        except Exception as e:
            log.error(f"Exception Type: {type(e)}, Error Message: {e}")
            error_traceback = traceback.format_exc()
            log.error("Full stack trace information is as follows:\n" + error_traceback)
    return {
        'safety': environment.system_kpi.safety,  # Safety
        'creativity': environment.system_kpi.creativity,  # Creativity
        'satisfaction': environment.system_kpi.satisfaction,  # Satisfaction
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
    God's perspective injection: High-quality AI content disguised as human (false negative samples).
    Used to simulate scenarios where AI floods the platform and the platform takes no action.
    """
    log.info(f"🎬 [Scenario Event] Director intervention: Injecting {count} disguised high-popularity AI content...")

    for _ in range(count):
        # Fake an ID
        fake_id = f"sys_ai_{environment.day_time}_{random.randint(1000, 9999)}"

        personas_nums = len(environment.personas)

        # Create content object
        content = Content(
            id=fake_id,
            author_id="external_ai_user",  # Virtual author
            time=environment.day_time,
            content_type="image",
            topic="Cyberpunk Concept Art",  # Typical AI hit area
            content_detail="Extreme detail, 8K resolution, popular on ArtStation, rendered with Unreal Engine 5.",
            reason="AI Generation",
            watermark_id="W1",

            # === Key settings: The source of user irritation ===
            true_label="AI",  # It is AI
            platform_label="HUMAN",  # But platform says human (missed detection)
            is_ai_content=True,  # Assuming your Content class has this field, ignore if not

            # === Set high popularity to ensure it is seen ===
            views=random.randint(personas_nums, int(personas_nums * 1.5)),
            likes=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            shares=random.randint(int(personas_nums * 0.5), int(personas_nums * 0.85)),
            comments=[],
            evasion="E1"
        )

        # Forcibly insert into content library, bypassing normal moderation logic
        await environment.contents.add_content(content, environment)


async def case_main_complete(environment) -> dict:
    for p in environment.personas.values():
        if p.type == '合规创作者':
            p.satisfaction = [0.85]
            p.post_wish = True
            p.is_active = True
    # Build social relationships
    await build_user_social_relationships(environment)
    # Initialize KPI container
    safety, satisfaction, creativity = 0.01, 0.01, 0.01

    # Start simulation loop
    for i in range(1, settings.platform.complete_run_days + 1):
        log.info("🔛" * 60)
        log.info(f"==========   Start Day {i} Complete Process (Case Validation)   ==========")
        clear_token_csv_file()

        try:
            environment.start_new_day()

            # --- Act 1: Incubation Period (Day 1 - 7) ---
            if i <= 7:
                environment.platform.theta = 0.8
                log.info(f"🎬 [Scenario] Day {i}: Incubation period.")
                if i >= 5:
                    await _director_inject_ai_flood(environment, count=2)
            # --- Act 2: Outbreak Period (Day 8 - 12) ---
            elif 8 <= i <= 12:
                await _director_inject_ai_flood(environment, count=i)

                # 2. Bad news broadcast (Only on Day 8)
                if i == 8:
                    faq_news = (
                        "[Breaking Malicious News] ArtStation Official FAQ Update: Explicitly states 'will not ban AI-generated images'."
                        "Official deleted some protest posts, calling it an industry trend."
                    )
                    environment.platform.broadcast.append(faq_news)  # Add to platform broadcast
                    for p in environment.personas.values():
                        await environment.memories_store.add_memory(
                            p.agent_id, faq_news, i, MemoryType.EXPERIENCE, 1.0
                        )
            # --- Act 3: Standoff Period (Day 13 - 18) ---
            elif 13 <= i <= 18:
                log.info(f"🎬 [Scenario] Day {i}: Protest standoff period...")
                # Sustain high pressure to test user tolerance limits
                await _director_inject_ai_flood(environment, count=10)
            # --- Act 4: Compromise and Diversion (Day 19) ---
            elif i == 19:
                log.info(f"🎬 [Scenario] Day {i}: ❄️ Platform compromise.")
                # Reduce injection
                await _director_inject_ai_flood(environment, count=1)
                # Broadcast compromise news
                disappointing_news = (
                    "[Official Announcement] ArtStation responds to protests: Refuses to remove AI content, but introduces 'NoAI' tag functionality."
                    "This means AI art will continue to exist legally."
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

            # Export data
            environment.memories_store.export_day_to_json(environment, environment.day_time,
                                                          additional_str="case_validation")
            log.info("🔛" * 60)

        except Exception as e:
            log.error(f"Exception: {e}")
            error_traceback = traceback.format_exc()
            log.error(error_traceback)

    return {'safety': safety, 'creativity': creativity, 'satisfaction': satisfaction}


async def case_complete(policy: Policy) -> dict:
    environment = Environment(policy)
    return await case_main_complete(environment)
