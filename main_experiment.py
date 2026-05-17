import argparse
import asyncio
import json
from pathlib import Path

from config import settings
from framework_utils import (
    experiment_saver,
    final_result_data_analysis,
    high_low_evaluation,
    run_high_model,
    run_high_model_after_nsga2,
    run_low_model,
)
from method.environment import Policy


PROJECT_ROOT = Path(__file__).resolve().parent


def set_personas_file(name: str) -> None:
    settings.file_load_path.personas_file = str(PROJECT_ROOT / "method" / "data" / name)


@experiment_saver(save=True)
async def run_single_low() -> None:
    settings.platform.simple_run_days = 15
    settings.platform.kpi_window_size = 3
    settings.platform.import_policy_day_time = 3
    set_personas_file("trend_consistency_30.json")
    result = await run_low_model(Policy(e_edu="medium", f_penalty=0.4, ai_threshold=0.65))
    final_result_data_analysis({"mode": "single_low", "result": result})


@experiment_saver(save=True)
async def run_single_high() -> None:
    settings.platform.complete_run_days = 15
    settings.platform.kpi_window_size = 3
    settings.platform.import_policy_day_time = 3
    set_personas_file("personas_30.json")
    result = await run_high_model(Policy(e_edu="medium", f_penalty=0.4, ai_threshold=0.65))
    final_result_data_analysis({"mode": "single_high", "result": result})


@experiment_saver(save=True)
async def run_policy_optimization(population_size: int, generations: int, high_eval: bool) -> None:
    settings.nsga.population_size = population_size
    settings.nsga.generations = generations
    settings.platform.simple_run_days = 15
    settings.platform.complete_run_days = 15
    settings.platform.kpi_window_size = 3
    settings.platform.import_policy_day_time = 3
    set_personas_file("low_beta_personas_30.json")
    await high_low_evaluation(open_high_simulation=high_eval)


@experiment_saver(save=True)
async def run_high_from_elites(results_file: Path) -> None:
    with results_file.open("r", encoding="utf-8") as f:
        results_data = json.load(f)

    elite_solutions = results_data.get("elite_solutions", [])
    settings.platform.complete_run_days = 15
    settings.platform.kpi_window_size = 3
    settings.platform.import_policy_day_time = 3
    set_personas_file("low_beta_personas_30.json")

    final_results_data = await run_high_model_after_nsga2(elite_solutions)
    final_result_data_analysis(
        {
            "mode": "high_from_elites",
            "source": str(results_file),
            "elite_solutions_results": final_results_data,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimuGov simulations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("single-low")
    subparsers.add_parser("single-high")

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--population-size", type=int, default=10)
    optimize.add_argument("--generations", type=int, default=1)
    optimize.add_argument("--high-eval", action="store_true")

    high_from_elites = subparsers.add_parser("high-from-elites")
    high_from_elites.add_argument("results_file", type=Path)

    args = parser.parse_args()
    if args.command == "single-low":
        asyncio.run(run_single_low())
    elif args.command == "single-high":
        asyncio.run(run_single_high())
    elif args.command == "optimize":
        asyncio.run(run_policy_optimization(args.population_size, args.generations, args.high_eval))
    elif args.command == "high-from-elites":
        asyncio.run(run_high_from_elites(args.results_file))


if __name__ == "__main__":
    main()
