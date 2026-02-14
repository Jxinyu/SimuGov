import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from config import settings
from method.environment import Policy
from method.simulation_main import simple, complete
from method.utils.get_personas.build_atrstation_personas import build_artstation_personas_main
from nsga.nsga2_framework import run_nsga2
from utils.context import current_sim_subdir
from utils.draw_kpi_contrast import draw_kpi_main
from utils.results_handler import save_experiment_results
from utils.visualize_evolution import nsga_evaluation_data_draw_main
from method import environment as env_module

log = logging.getLogger(__name__)


def setup_logger(base_dir):
    """
    Called every time it runs, used to redirect log files to a new folder.
    """
    # 1. Determine the log directory
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Generate a new log file name
    log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # 3. Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- [Key modification] Force clear all old handlers ---
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)

    # --- 4. Re-add: File processor (FileHandler) ---
    file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(filename)s:%(lineno)d | %(funcName)-20s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- 5. Re-add: Console processor (StreamHandler) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"✅ Log system has been reset.")
    root_logger.info(f"📁 Log file path: {log_filepath}")

    return log_filepath


def get_current_timestamp_dir():
    base = r'result_data'
    base += '/' + str(time.strftime("%Y%m%d", time.localtime()))
    base += '/' + str(time.strftime("%H%M%S", time.localtime()))
    return base


async def complete_framework(policy: Policy) -> dict:
    res = await complete(policy)
    return res


async def simple_framework(policy: Policy) -> dict:
    return await simple(policy)


async def nsga2_framework_evaluation(population_size, generations):
    """
    Genetic algorithm evolution
    """
    # Run NSGA-II
    elite_solutions, evolution_history = await run_nsga2(population_size=population_size, generations=generations)
    log.info("\n--- Final elite solutions (Pareto optimal set) ---")

    # Return complete elite solutions
    final_elites = []
    for i, solution in enumerate(elite_solutions):
        original_kpi = {k: -v for k, v in solution['kpi'].items()}
        log.info(f"Solution {i + 1}:")
        log.info(f"  Policy: {solution['policy']}")
        log.info(f"  KPIs (Simple): {original_kpi}")
        log.info("-" * 20)
        # Also store the KPIs restored to positive values in the solution for subsequent use
        solution['kpi'] = original_kpi
        final_elites.append(solution)

    return elite_solutions, evolution_history


async def complete_framework_evaluation(elite_solutions_from_nsga):
    """
    Receive the list of elite solutions found by NSGA-II and run complete simulations for them.
    """
    tasks = []
    for solution in elite_solutions_from_nsga:
        p = solution['policy']
        task = complete_framework(Policy(f_penalty=p['f_penalty'], e_edu=p['e_edu'], ai_threshold=p['ai_threshold']))
        tasks.append(task)

    # Concurrently execute all complete simulation tasks
    complete_kpi_results = await asyncio.gather(*tasks)

    # Merge complete simulation results with elite solution data
    for i, solution in enumerate(elite_solutions_from_nsga):
        solution['complete_kpi'] = complete_kpi_results[i]

    return elite_solutions_from_nsga


def data_save(experiment_data_to_save: dict, output_dir: str):
    """
    Data analysis and preservation
    """
    try:
        # Save experimental data
        experiment_results_file_path = save_experiment_results(experiment_data_to_save, str(output_dir) + "/Experimental_Data")
    except:
        experiment_results_file_path = save_experiment_results(experiment_data_to_save, str(output_dir))
    try:
        # nsga evolution plotting
        nsga_evaluation_data_draw_main(experiment_results_file_path, str(output_dir) + "/nsga_evolution_plotting")
    except:
        nsga_evaluation_data_draw_main(experiment_results_file_path, str(output_dir))

    # Draw various types of charts
    path_obj = Path(output_dir)
    draw_kpi_main(plot_mode='T', use_simplified_data=True, output_dir=path_obj / "simplified_framework_kpi_plotting")
    draw_kpi_main(plot_mode='T', use_simplified_data=False, output_dir=path_obj / "complete_framework_kpi_plotting")
    draw_kpi_main(plot_mode='P', use_simplified_data=True, output_dir=path_obj / "simplified_framework_Pareto_frontier_plotting")
    draw_kpi_main(plot_mode='P', use_simplified_data=False, output_dir=path_obj / "final_Pareto_frontier_plotting")

    # Write termination time
    with open(path_obj / "termination_time.txt", "w") as f:
        f.write(str(datetime.now()))


async def test_complete_three_policies():
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    # System effectiveness closed-loop experiment
    settings.platform.complete_run_days = 20  # High-granularity run days
    settings.platform.kpi_window_size = 3  # KPI window size
    settings.platform.import_policy_day_time = 3  # Policy import time
    settings.file_load_path.personas_file = r'method/data/scenario_protest.json'

    policy1 = Policy(e_edu='Low', f_penalty=0.01, ai_threshold=0.99)
    policy2 = Policy(e_edu='Medium', f_penalty=0.5, ai_threshold=0.5)
    policy4 = Policy(e_edu='Low', f_penalty=0.99, ai_threshold=0.01)
    policy3 = Policy(e_edu='High', f_penalty=0.99, ai_threshold=0.01)  # Low creativity, high safety
    res = await asyncio.gather(
        # complete_framework(policy1),
        complete_framework(policy2),
        # complete_framework(policy3),
        # complete_framework(policy4)
    )

    print(res)
    experiment_data_to_save = {
        "run_parameters": [],
        "timings": [],
        "elite_solutions_results": res,
        "evolution_history": []
    }

    data_save(experiment_data_to_save, current_dir)


async def test_simple_three_policies():
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    now = datetime.now()
    fixed_subdir = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    token = current_sim_subdir.set(fixed_subdir)

    settings.platform.case_validation = False
    settings.platform.simple_run_days = 15

    settings.file_load_path.personas_file = r'method\data\personas_30.json'

    settings.platform.kpi_window_size = 3  # KPI window size
    settings.platform.import_policy_day_time = 2  # Policy import time

    try:
        policy1 = Policy(e_edu='Low', f_penalty=0.01, ai_threshold=0.99)  # High creativity, low safety
        # policy2 = Policy(e_edu='Medium', f_penalty=0.6, ai_threshold=0.6)
        # policy4 = Policy(e_edu='Low', f_penalty=0.2, ai_threshold=0.2)
        policy3 = Policy(e_edu='High', f_penalty=0.99, ai_threshold=0.01)  # Low creativity, high safety
        res = await asyncio.gather(
            simple_framework(policy1),
            # simple_framework(policy2),
            simple_framework(policy3),
            # simple_framework(policy4)
        )
        # res = ['1']
        print(res)

        experiment_data_to_save = {
            "run_parameters": [],
            "timings": [],
            "elite_solutions_results": res,
            "evolution_history": []
        }

        data_save(experiment_data_to_save, current_dir)
    finally:
        current_sim_subdir.reset(token)


async def run_complete_in_one_policy(e_edu, f_penalty, ai_threshold):
    """
    Test a policy's effectiveness experiment in high-granularity

    No parameters are set in this function

    :param e_edu:
    :param f_penalty:
    :param ai_threshold:
    :return:
    """
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    policy1 = Policy(e_edu=e_edu, f_penalty=f_penalty, ai_threshold=ai_threshold)
    res = await asyncio.gather(
        complete_framework(policy1),
    )

    print(res)
    experiment_data_to_save = {
        "run_parameters": [],
        "timings": [],
        "elite_solutions_results": res,
        "evolution_history": []
    }

    data_save(experiment_data_to_save, current_dir)


async def high_low_evaluation(open_high_simulation: bool = False):
    """
    Complete main process, including timing, execution, and preservation.
    open_high_simulation: Turn on the high-granularity model
    """
    # 1. Determine the total directory for this experiment (using real physical time)
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    now = datetime.now()
    fixed_subdir = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"

    # 2. Set context variables (returns a token for later reset, though not necessary to reset as the script will automatically destroy it)
    token = current_sim_subdir.set(fixed_subdir)

    log.info(f"🔒 [Context] Unified simulation directory set: {fixed_subdir}")

    try:
        run_params = {
            "population_size": settings.nsga.population_size,  # Initial population size
            "generations": settings.nsga.generations  # Iteration rounds
        }
        timings = {}
        total_start_time = time.perf_counter()

        log.info("Phase 1: NSGA-II rapid optimization starts")
        phase1_start_time = time.perf_counter()

        # Genetic algorithm evolution
        elite_solutions, evolution_history = await nsga2_framework_evaluation(
            population_size=settings.nsga.population_size,
            generations=settings.nsga.generations
        )

        print()

        phase1_end_time = time.perf_counter()
        timings["phase1_nsga2_duration_seconds"] = round(phase1_end_time - phase1_start_time, 2) / 60
        log.info(f"Phase 1 completed, duration: {timings['phase1_nsga2_duration_seconds']:.2f} minutes ")

        final_results_data = []
        if elite_solutions and open_high_simulation:
            log.info(f" Phase 2: Elite strategy precision evaluation starts {len(elite_solutions)}")
            phase2_start_time = time.perf_counter()

            # Complete simulation framework evaluation
            final_results_data = await complete_framework_evaluation(elite_solutions)

            phase2_end_time = time.perf_counter()
            timings["phase2_complete_sim_duration_seconds"] = round(phase2_end_time - phase2_start_time, 2) / 60
            log.info(f"Phase 2 completed, duration: {timings['phase2_complete_sim_duration_seconds']:.2f} minutes ")

        total_end_time = time.perf_counter()
        timings["total_duration_seconds"] = round(total_end_time - total_start_time, 2) / 60
        log.info(f"\n=== All processes ended, total duration: {timings['total_duration_seconds']:.2f} minutes ===")

        experiment_data_to_save = {
            "run_parameters": run_params,
            "timings": timings,
            "nums": len(elite_solutions),
            "elite_solutions_results": final_results_data,
            "evolution_history": evolution_history,
            "elite_solutions": elite_solutions
        }

        data_save(experiment_data_to_save, current_dir)

    finally:
        current_sim_subdir.reset(token)


def build_personas_polls():
    """
    Construct agent pool
    :return:
    """
    from method.utils.get_personas.build_artstation_dataset_poolsl import build_agent_pools_demo
    build_agent_pools_demo(100, 100, 100)


def build_artstation_personas():
    """
    Construct personas dataset for ArtStation
    :return:
    """
    build_artstation_personas_main()
