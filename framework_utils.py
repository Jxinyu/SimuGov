import asyncio
import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from config import settings
from method.environment import Policy
from method.simulation_main import high, low
from nsga.nsga2_framework import nsga2_entrance
from utils.draw_kpi_contrast import draw_kpi_main
from utils.visualize_evolution import nsga_evaluation_data_draw_main


log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def get_current_timestamp_dir() -> str:
    now = time.localtime()
    base_dir = PROJECT_ROOT / "result_data" / time.strftime("%Y%m%d", now) / time.strftime("%H%M%S", now)
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)


def experiment_saver(save: bool = True):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if save:
                    settings.file_load_path.base_store_file = get_current_timestamp_dir()
                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if save:
                settings.file_load_path.base_store_file = get_current_timestamp_dir()
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator


async def run_high_model(policy: Policy) -> dict:
    try:
        return await high(policy)
    except Exception as exc:
        print(f"High-fidelity simulation failed: {exc}")
        return {}


async def run_low_model(policy: Policy) -> dict:
    try:
        return await low(policy)
    except Exception as exc:
        print(f"Low-fidelity simulation failed: {exc}")
        return {}


def final_result_data_analysis(experiment_data_to_save: dict = None, output_dir: str | Path = None) -> None:
    target_dir = output_dir or settings.file_load_path.base_store_file
    if not target_dir:
        log.warning("No output directory is configured; skip result persistence.")
        return

    result_dir = Path(target_dir) / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    if experiment_data_to_save:
        experiment_results_file_path = save_experiment_results(experiment_data_to_save, result_dir)
        if experiment_results_file_path:
            try:
                nsga_evaluation_data_draw_main(experiment_results_file_path, str(result_dir))
            except Exception as exc:
                print(f"Failed to draw NSGA results: {exc}")

    draw_kpi_main(plot_mode="T", use_simplified_data=True, output_dir=result_dir / "simple_kpi_timeseries")
    draw_kpi_main(plot_mode="T", use_simplified_data=False, output_dir=result_dir / "complete_kpi_timeseries")
    draw_kpi_main(plot_mode="P", use_simplified_data=True, output_dir=result_dir / "simple_pareto_front")
    draw_kpi_main(plot_mode="P", use_simplified_data=False, output_dir=result_dir / "complete_pareto_front")

    (result_dir / "termination_time.txt").write_text(str(datetime.now()), encoding="utf-8")


async def run_nsga2(population_size, generations):
    elite_solutions, evolution_history = await nsga2_entrance(
        population_size=population_size,
        generations=generations,
    )

    for i, solution in enumerate(elite_solutions):
        original_kpi = {k: -v for k, v in solution["kpi"].items()}
        print(f"Solution {i + 1}:")
        print(f"  Policy: {solution['policy']}")
        print(f"  KPIs (Simple): {original_kpi}")
        print("-" * 20)
        solution["kpi"] = original_kpi

    return elite_solutions, evolution_history


async def run_high_model_after_nsga2(elite_solutions_from_nsga):
    tasks = []
    for solution in elite_solutions_from_nsga:
        policy = solution["policy"]
        tasks.append(
            run_high_model(
                Policy(
                    f_penalty=policy["f_penalty"],
                    e_edu=policy["e_edu"],
                    ai_threshold=policy["ai_threshold"],
                )
            )
        )

    complete_kpi_results = await asyncio.gather(*tasks)
    for i, solution in enumerate(elite_solutions_from_nsga):
        solution["complete_kpi"] = complete_kpi_results[i]

    return elite_solutions_from_nsga


async def high_low_evaluation(open_high_simulation: bool = False):
    run_params = {
        "population_size": settings.nsga.population_size,
        "generations": settings.nsga.generations,
    }

    start_time = time.perf_counter()
    print("Stage 1: start NSGA-II search")
    elite_solutions, evolution_history = await run_nsga2(
        population_size=settings.nsga.population_size,
        generations=settings.nsga.generations,
    )
    nsga2_time = round(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    final_results_data = []
    if elite_solutions and open_high_simulation:
        print(f"Stage 2: high-fidelity evaluation for {len(elite_solutions)} elite policies")
        final_results_data = await run_high_model_after_nsga2(elite_solutions)
    high_time = round(time.perf_counter() - start_time)

    final_result_data_analysis(
        {
            "run_parameters": run_params,
            "timings": {
                "nsga2_time": nsga2_time,
                "high_time": high_time,
            },
            "nums": len(elite_solutions),
            "elite_solutions_results": final_results_data,
            "elite_solutions": elite_solutions,
            "evolution_history": evolution_history,
        }
    )


def save_experiment_results(data: dict, file_path: str | Path = None):
    target_dir = Path(file_path) if file_path else PROJECT_ROOT / "result_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    full_file_path = target_dir / "experiment_results.json"

    try:
        with full_file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved experiment results to: {full_file_path}")
        return full_file_path
    except Exception as exc:
        print(f"Failed to save experiment results: {exc}")
        return None
