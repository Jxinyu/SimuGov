import asyncio
import os
import sys

from framework_utils import test_simple_three_policies
# Import configuration (ensure environment variables can be set before config is loaded, or let config handle it automatically)
# Here we directly import experimental functions
from main_experiment import (
    low_model_effective_filter_experiment,
    framework_efficiency_experiment,
    complete_three_policies,
    effect_verification, read_lf_policy_run_complete,
    adaptive_experiment_low_in_high,
    adaptive_experiment_filter_elites, low_model_effectiveness_experiment, case_experiment, case_ablation_experiment
)
from config import settings


def adaptive_experiment():
    """
    Policy adaptation experiment
    :return:
    """

    pass

async def run_sequentially():
    for i in range(7, 9):
        await framework_efficiency_experiment(i)


if __name__ == '__main__':
    # 1. Force clear proxy settings in environment variables
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['all_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['no_proxy'] = '*'
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY_IMPL"] = "false"

    # High-low granularity effectiveness experiment
    # asyncio.run(adaptive_experiment_low_in_high('Medium', 0.46, 0.90))
    # asyncio.run(effect_verification())
    # asyncio.run(read_lf_policy_run_complete(r'result_data\20260121\low_reactance_evolution_results'))
    # asyncio.run(read_lf_policy_run_complete(r'result_data\20260118\221824'))
    # asyncio.run(complete_three_policies())  # Three baseline experiments
    # asyncio.run(adaptive_experiment_filter_elites(False, False))  # Low-reactance society screen elite solution sets

    # asyncio.run(complete_three_policies())

    # Framework efficiency experiment
    # asyncio.run(framework_efficiency_experiment(1))
    asyncio.run(run_sequentially())

    # Simplified framework policy test
    # asyncio.run(test_simple_three_policies())
    # asyncio.run(complete_three_policies())

    # Case validation experiment
    # asyncio.run(case_experiment())  # case_verification_and_ablation
    # asyncio.run(case_ablation_experiment())  # Ablation experiment

    # KPI robustness experiment
    # time.sleep(60)
    # settings.platform.case_validation = False
    # settings.platform.complete_run_days = 20
    # settings.file_load_path.personas_file = r'method\data\inner_consistency_personas_60.json'
    #
    # for i in range(5):
    #     asyncio.run(complete_three_policies())
    #     time.sleep(60)

    # High-low granularity trend consistency experiment
    # settings.platform.complete_run_days = 10
    # settings.platform.simple_run_days = 10
    # settings.file_load_path.personas_file = r'method\data\trend_consistency_30.json'
    # asyncio.run(trend_consistency_verification())
    # low_model_effectiveness_experiment(40)