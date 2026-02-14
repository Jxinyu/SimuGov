import asyncio
import json
import os
import time
import logging
from datetime import datetime

from config import settings
from method.environment import Policy
from method import environment as env_module
from method.simulation_main import case_complete
from method.utils.calculation_token_nums import clear_token_csv_file, calculate_token_nums
from utils.context import current_sim_subdir
from utils.experience_utils import generate_experimental_policies
from utils.results_handler import save_experiment_results

from framework_utils import (
    setup_logger,
    simple_framework,
    complete_framework,
    data_save,
    get_current_timestamp_dir,
    nsga2_framework_evaluation,
    complete_framework_evaluation,
    test_complete_three_policies,
    test_simple_three_policies,
    high_low_evaluation,
    run_complete_in_one_policy
)

log = logging.getLogger(__name__)


async def low_model_effective_filter_experiment(policies_num):
    """
    Low-granularity model effectiveness screening experiment - cannot screen out excellent policies
    1. Generate random policy groups.
    2. Run Simple and Complete frameworks respectively.
    3. [Physical Isolation] Simple and Complete databases are stored in different folders to ensure absolute one-to-one correspondence.
    """
    # 1. Prepare policies (using extreme anchor strategies, better effect)
    sampled_policies = generate_experimental_policies(policies_num=policies_num)

    # 2. Generate a unified experimental total root directory
    # For example: D:\Assign\...\result_data\20260107_193000_TrendTest
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        r'result_data',
        f"{now_str}_TrendTest"
    )

    if not os.path.exists(experiment_root):
        os.makedirs(experiment_root)

    # Initialize log (logs are saved in the total root directory)
    setup_logger(experiment_root)

    # Save original configuration for recovery
    original_chroma_path = settings.file_load_path.chroma_db_file
    original_export_path = settings.file_load_path.daily_memory_exports_file

    try:
        log.info("=" * 60)
        log.info(f"🚀 Start trend consistency verification experiment | Experiment ID: {now_str}")
        log.info("=" * 60)

        # =================================================================
        # Phase 1: Run simplified framework (Simple Framework)
        # =================================================================
        log.info(">>> [Phase 1] Run simplified framework (Simple Framework) <<<")

        # [Dynamic Configuration Injection] Point all Simple outputs (including DB) to experiment_root/Simple_Run
        simple_root = os.path.join(experiment_root, "Simple_Run")

        # 1. Set DB storage path: each policy will create its own UUID subfolder inside
        settings.file_load_path.chroma_db_file = os.path.join(simple_root, "chroma_db")
        # 2. Set JSON export path
        settings.file_load_path.daily_memory_exports_file = os.path.join(simple_root, "exports")

        log.info(f"    📂 Simple DB path has been isolated to: {settings.file_load_path.chroma_db_file}")

        # 3. Concurrent execution
        simple_tasks = [simple_framework(p) for p in sampled_policies]
        simple_results = await asyncio.gather(*simple_tasks)

        # 4. Save summary results
        simple_summary = {
            "run_type": "simple",
            "policies": [{"ai": p.ai_threshold, "f": p.f_penalty, "edu": p.e_edu} for p in sampled_policies],
            "results": simple_results
        }
        save_experiment_results(simple_summary, simple_root)

        # =================================================================
        # Phase 2: Run complete framework (Complete Framework)
        # =================================================================
        log.info("\n" + "=" * 40 + "\n")
        log.info(">>> [Phase 2] Run complete framework (Complete Framework) <<<")

        # [Dynamic Configuration Injection] Point all Complete outputs (including DB) to experiment_root/Complete_Run
        # This ensures that the Complete database and the Simple database are two completely physically isolated folders
        complete_root = os.path.join(experiment_root, "Complete_Run")

        # 1. Set DB storage path
        settings.file_load_path.chroma_db_file = os.path.join(complete_root, "chroma_db")
        # 2. Set JSON export path
        settings.file_load_path.daily_memory_exports_file = os.path.join(complete_root, "exports")

        log.info(f"    📂 Complete DB path has been isolated to: {settings.file_load_path.chroma_db_file}")

        # semaphore = asyncio.Semaphore(5)
        # # 2. Define a wrapper function controlled by a semaphore
        # async def limited_complete_framework(policy):
        #     async with semaphore:  # Only after obtaining the semaphore (token) can execution continue
        #         # Can add a log here to see if it is indeed queuing
        #         # log.info(f"🚦 Start executing policy simulation: {policy}")
        #         return await complete_framework(policy)
        #
        # # 3. Create task list (Note: the wrapped limited_complete_framework is called here)
        # complete_tasks = [limited_complete_framework(p) for p in sampled_policies]
        #
        # # 4. Concurrent execution (asyncio.gather will automatically schedule, only tasks that obtain the semaphore will actually run)
        # complete_results = await asyncio.gather(*complete_tasks)

        # 4. Save summary results
        complete_summary = {
            "run_type": "complete",
            "policies": [{"ai": p.ai_threshold, "f": p.f_penalty, "edu": p.e_edu} for p in sampled_policies],
            "results": None
        }
        save_experiment_results(complete_summary, complete_root)

        # =================================================================
        # End
        # =================================================================
        log.info("=" * 60)
        log.info("✅ Experiment ended successfully!")
        log.info(f"📊 Simple data location:   {os.path.join(simple_root, 'exports')}")
        log.info(f"📊 Complete data location: {os.path.join(complete_root, 'exports')}")
        log.info("=" * 60)

        # Print analysis command prompt
        print("\nNow the analysis script can be run:")
        print(
            f"verify_trend_consistency(\n    r'{os.path.join(simple_root, 'exports')}',\n    r'{os.path.join(complete_root, 'exports')}'\n)")
    except:
        log.error("An error occurred! Skip")
    finally:
        # Restore original configuration, does not affect other code
        settings.file_load_path.chroma_db_file = original_chroma_path
        settings.file_load_path.daily_memory_exports_file = original_export_path


async def framework_efficiency_experiment(population_size):
    """
    Framework efficiency verification experiment
    Compare Group A (Baseline: full high-granularity) with Group B (Proposed: high-low granularity coupling)


    Question: Should the number of elite solution sets screened by the high-low granularity coupling method be consistent with the number of solution sets screened by the full high-granularity method???

    """
    # 1. Experiment initialization
    settings.nsga.population_size = population_size
    settings.nsga.generations = 2

    settings.platform.simple_run_days = 4
    settings.platform.complete_run_days = 4

    settings.platform.import_policy_day_time = 1
    settings.platform.kpi_window_size = 1

    # settings.file_load_path.personas_file = r'method/data/test_persona.json'
    settings.file_load_path.personas_file = r'method/data/inner_consistency_personas_20.json'

    # Backup original export path, restore after experiment ends
    original_export_path = settings.file_load_path.daily_memory_exports_file
    original_chroma_path = settings.file_load_path.chroma_db_file

    # Save original datetime class for recovery
    original_datetime_class = env_module.datetime.datetime

    # Create experimental root directory
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    log.info("=" * 60)
    log.info(f"🚀 Start framework efficiency verification experiment (Time Freeze Mode)")
    log.info("=" * 60)

    results_summary = {
        "experiment_name": "Framework Efficiency Verification",
        "parameters": {
            "population": settings.nsga.population_size,
            "generations": settings.nsga.generations,
            "simple_run_days": settings.platform.simple_run_days,
            "complete_run_days": settings.platform.complete_run_days,
            "import_policy_day_time": settings.platform.import_policy_day_time,
            "kpi_window_size": settings.platform.kpi_window_size,
            "persona_file": settings.file_load_path.personas_file
        },
        "groups": {}
    }

    try:
        # ==========================================
        # 2. Run Group A: Baseline
        # ==========================================
        group_a_dir = os.path.join(current_dir, "Group_A_Full_High_Granularity")
        log_path_a = setup_logger(group_a_dir)

        # --- Freeze time ---
        target_now_a = datetime.now()

        class FixedDatetimeA(datetime):
            @classmethod
            def now(cls, tz=None):
                return target_now_a

        env_module.datetime.datetime = FixedDatetimeA  # Applying patch
        log.info(f"🔒 [Group A] Time has been frozen to: {target_now_a.strftime('%H:%M:%S')}")

        settings.file_load_path.daily_memory_exports_file = os.path.join(group_a_dir, "sim_data")
        settings.file_load_path.chroma_db_file = os.path.join(group_a_dir, "chroma_db")

        log.info("\n>>> Phase 1/2: Run Group A (Baseline - full high-granularity) <<<")
        log.info(f"📄 Group A log has been redirected to: {log_path_a}")

        settings.platform.efficiency_validation = True
        clear_token_csv_file()
        start_time_a = time.perf_counter()

        # Run simulation
        elites_a, history_a = await nsga2_framework_evaluation(
            settings.nsga.population_size,
            settings.nsga.generations
        )

        end_time_a = time.perf_counter()
        token_cost_a = calculate_token_nums()
        time_cost_a = (end_time_a - start_time_a) / 60

        results_summary["groups"]["Group_A_Baseline"] = {
            "time_minutes": round(time_cost_a, 2),
            "token_cost": token_cost_a,
            "elite_solutions": elites_a
        }

        data_save({
            "run_parameters": results_summary["parameters"],
            "evolution_history": history_a,
            "elite_solutions_results": elites_a
        }, group_a_dir)

        log.info(f"✅ Group A complete: took {time_cost_a:.2f} minutes")

        # ==========================================
        # 3. Run Group B: Proposed
        # ==========================================
        group_b_dir = os.path.join(current_dir, "Group_B_High_Low_Granularity")
        log_path_b = setup_logger(group_b_dir)

        target_now_b = datetime.now()

        class FixedDatetimeB(datetime):
            @classmethod
            def now(cls, tz=None):
                return target_now_b

        env_module.datetime.datetime = FixedDatetimeB  # Update patch
        log.info(f"🔒 [Group B] Time has been frozen to: {target_now_b.strftime('%H:%M:%S')}")

        settings.file_load_path.daily_memory_exports_file = os.path.join(group_b_dir, "sim_data")
        settings.file_load_path.chroma_db_file = os.path.join(group_b_dir, "chroma_db")

        log.info("\n>>> Phase 2/2: Run Group B (Proposed - high-low granularity coupling) <<<")
        log.info(f"📄 Group B log has been redirected to: {log_path_b}")

        clear_token_csv_file()
        start_time_b = time.perf_counter()
        settings.platform.efficiency_validation = False

        # (1) Evolution phase
        elites_b_temp, history_b = await nsga2_framework_evaluation(
            settings.nsga.population_size,
            settings.nsga.generations
        )

        # (2) Verification phase
        log.info("   -> Perform high-granularity verification on elite policies...")
        elites_b_final = await complete_framework_evaluation(elites_b_temp[:len(elites_a)])

        end_time_b = time.perf_counter()
        token_cost_b = calculate_token_nums()
        time_cost_b = (end_time_b - start_time_b) / 60

        results_summary["groups"]["Group_B_Proposed"] = {
            "time_minutes": round(time_cost_b, 2),
            "token_cost": token_cost_b,
            "elite_solutions": elites_b_final
        }

        data_save({
            "run_parameters": results_summary["parameters"],
            "evolution_history": history_b,
            "elite_solutions_results": elites_b_final
        }, group_b_dir)

        log.info(f"✅ Group B complete: took {time_cost_b:.2f} minutes")

        # ==========================================
        # 4. Summary
        # ==========================================
        setup_logger(current_dir)

        speedup = time_cost_a / time_cost_b if time_cost_b > 0 else 0
        token_saving = (1 - token_cost_b / token_cost_a) * 100 if token_cost_a > 0 else 0

        log.info("=" * 60)
        log.info(f"📊 Experimental result summary")
        log.info(f"   Speedup ratio (Speedup): {speedup:.2f}x")
        log.info(f"   Cost saving (Saving): {token_saving:.2f}%")
        log.info("=" * 60)

        save_experiment_results(results_summary, current_dir)

    finally:
        # 1. Unlock time
        env_module.datetime.datetime = original_datetime_class
        # 2. Restore path configuration
        settings.file_load_path.daily_memory_exports_file = original_export_path
        settings.file_load_path.chroma_db_file = original_chroma_path

        print("🔓 [System] Simulation environment timestamp has been unlocked, path has been restored.")


async def complete_three_policies():
    await test_complete_three_policies()


async def simple_three_policies():
    await test_simple_three_policies()


async def effect_verification():
    """
    closed_loop_effectiveness_experiment
    :return:
    """
    # 1. Experiment initialization
    settings.nsga.population_size = 20  # Initial population size
    settings.nsga.generations = 6  # Iteration rounds

    settings.platform.simple_run_days = 15  # Low-granularity run days
    settings.platform.complete_run_days = 15  # High-granularity run days

    settings.platform.kpi_window_size = 3  # KPI window size
    settings.platform.import_policy_day_time = 3  # Policy import time

    # settings.file_load_path.personas_file = r'method/data/personas_30.json'  # People file
    # settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'
    settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'
    # settings.file_load_path.personas_file = r'method/data/personas_50.json'  # People file
    # settings.file_load_path.personas_file = r'method/data/test_persona.json'  # People file

    await high_low_evaluation()


async def adaptive_experiment_filter_elites(high: bool = True, open_high_model: bool = True):
    """
    Policy adaptation experiment - screen optimal solutions
    high-true: Screen optimal solutions in a rebellious society
    high-false: Screen optimal solutions in a compliant society
    """
    settings.nsga.population_size = 20  # Initial population size
    settings.nsga.generations = 15  # Iteration rounds

    settings.platform.simple_run_days = 15  # Low-granularity run days
    settings.platform.complete_run_days = 15  # High-granularity run days

    settings.platform.kpi_window_size = 3  # KPI window size
    settings.platform.import_policy_day_time = 3  # Policy import time

    if high:
        settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'
    else:
        settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'

    await high_low_evaluation(open_high_simulation=False)  # Both high and low granularity run, completed in one go


async def adaptive_experiment_low_in_high(e_edu, f_penalty, ai_threshold):
    """
    Policy adaptation experiment - put the optimal solutions screened from the compliant society into the rebellious society
    :return:
    """
    settings.platform.complete_run_days = 15  # High-granularity run days

    settings.platform.kpi_window_size = 3  # KPI window size
    settings.platform.import_policy_day_time = 3  # Policy import time

    settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'

    await run_complete_in_one_policy(e_edu, f_penalty, ai_threshold)


async def case_experiment():
    """
    Case validation experiment  Turn on PAEP
    :return:
    """
    # Case validation experiment
    settings.platform.case_validation = True
    settings.platform.complete_run_days = 30
    settings.file_load_path.personas_file = 'method/data/scenario_protest.json'

    await run_complete_in_one_policy('Low', 0.01, 0.8)


async def case_ablation_experiment():
    """
    Case validation ablation experiment - Turn off PAEP
    :return:
    """
    settings.platform.ablation_validation = True  # Turn on ablation experiment - Turn off PAEP
    settings.platform.case_validation = True
    settings.platform.complete_run_days = 30
    settings.file_load_path.personas_file = 'method/data/scenario_protest-2.json'

    await run_complete_in_one_policy('Low', 0.01, 0.8)


def low_model_effectiveness_experiment(policies_num):
    """
    Low-granularity model effectiveness screening experiment
    :return:
    """
    settings.platform.complete_run_days = 15
    settings.platform.simple_run_days = 15
    settings.file_load_path.personas_file = r'method\data\trend_consistency_30.json'
    asyncio.run(low_model_effective_filter_experiment(policies_num))


async def read_lf_policy_run_complete(file_path: str):
    """
    Read the elite solution sets screened by the low-granularity model and put them into the high-granularity model for simulation
    :param file_path: result_data\20260118\134648
    :return:
    """
    with open(fr'{file_path}\Experimental_Data\experiment_results.json', "r", encoding="utf-8") as f:
        results_data = json.load(f)

    elite_policy = results_data['elite_solutions']

    # 1. Determine the total directory for this experiment (using real physical time)
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)
    now = datetime.now()
    fixed_subdir = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    # 2. Set context variables (returns a token for later reset, although not strictly necessary as it's destroyed when the script ends)
    token = current_sim_subdir.set(fixed_subdir)
    log.info(f"🔒 [Context] Unified simulation directory set to: {fixed_subdir}")

    for i in elite_policy:
        print(i['policy'])
    try:
        if elite_policy:
            log.info(f" Elite policy precision evaluation starts {len(elite_policy)}")

            settings.platform.simple_run_days = 15  # Low-granularity run days
            settings.platform.complete_run_days = 15  # High-granularity run days

            settings.platform.kpi_window_size = 3  # KPI window size
            settings.platform.import_policy_day_time = 3  # Policy import time

            settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'

            # Complete simulation framework evaluation
            final_results_data = await complete_framework_evaluation(elite_policy)

            experiment_data_to_save = {
                "run_parameters": [],
                "timings": [],
                "elite_solutions_results": final_results_data,
                "evolution_history": []
            }

            data_save(experiment_data_to_save, current_dir)
    finally:
        current_sim_subdir.reset(token)
