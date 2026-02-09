import asyncio
import os
import sys

from framework_utils import test_simple_three_policies
# 导入配置（确保环境变量能被 config 加载前设置，或者让 config 自动处理）
# 这里我们直接导入实验函数
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
    策略自适应实验
    :return:
    """

    pass

async def run_sequentially():
    for i in range(7, 9):
        await framework_efficiency_experiment(i)


if __name__ == '__main__':
    # 1. 强制清空环境变量中的代理设置
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['all_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['no_proxy'] = '*'
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY_IMPL"] = "false"

    # 高低粒度 有效性 实验
    # asyncio.run(adaptive_experiment_low_in_high('中', 0.46, 0.90))
    # asyncio.run(effect_verification())
    # asyncio.run(read_lf_policy_run_complete(r'result_data\20260121\低逆反进化结果'))
    # asyncio.run(read_lf_policy_run_complete(r'result_data\20260118\221824'))
    # asyncio.run(complete_three_policies())  # 三个基准实验
    # asyncio.run(adaptive_experiment_filter_elites(False, False))  # 低逆反社会  筛选精英解集

    # asyncio.run(complete_three_policies())

    # 框架效率实验
    # asyncio.run(framework_efficiency_experiment(1))
    asyncio.run(run_sequentially())

    # 简化框架政策测试
    # asyncio.run(test_simple_three_policies())
    # asyncio.run(complete_three_policies())

    # 案例验证实验
    # asyncio.run(case_experiment())  # 案例验证
    # asyncio.run(case_ablation_experiment())  # 消融实验

    # KPI鲁棒性实验
    # time.sleep(60)
    # settings.platform.case_validation = False
    # settings.platform.complete_run_days = 20
    # settings.file_load_path.personas_file = r'method\data\inner_consistency_personas_60.json'
    #
    # for i in range(5):
    #     asyncio.run(complete_three_policies())
    #     time.sleep(60)

    # 高低粒度趋势一致性实验
    # settings.platform.complete_run_days = 10
    # settings.platform.simple_run_days = 10
    # settings.file_load_path.personas_file = r'method\data\trend_consistency_30.json'
    # asyncio.run(trend_consistency_verification())
    # low_model_effectiveness_experiment(40)
