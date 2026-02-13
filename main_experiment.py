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
    低粒度模型有效性筛选实验   不能筛掉优秀的政策
    1. 生成随机策略组。
    2. 分别运行 Simple 和 Complete 框架。
    3. 【物理隔离】Simple 和 Complete 的数据库存放在不同文件夹，确保绝对的一对一。
    """
    # 1. 准备策略 (使用极端锚点策略，效果更好)
    sampled_policies = generate_experimental_policies(policies_num=policies_num)

    # 2. 生成一个统一的实验总根目录
    # 例如: D:\Assign\...\result_data\20260107_193000_TrendTest
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        r'result_data',
        f"{now_str}_TrendTest"
    )

    if not os.path.exists(experiment_root):
        os.makedirs(experiment_root)

    # 初始化日志 (日志保存在总根目录下)
    setup_logger(experiment_root)

    # 保存原始配置以便恢复
    original_chroma_path = settings.file_load_path.chroma_db_file
    original_export_path = settings.file_load_path.daily_memory_exports_file

    try:
        log.info("=" * 60)
        log.info(f"🚀 开始趋势一致性验证实验 | 实验ID: {now_str}")
        log.info("=" * 60)

        # =================================================================
        # 第一阶段：运行简化框架 (Simple Framework)
        # =================================================================
        log.info(">>> [Phase 1] 运行简化框架 (Simple Framework) <<<")

        # 【动态配置注入】将 Simple 的所有输出（包括DB）指向 experiment_root/Simple_Run
        simple_root = os.path.join(experiment_root, "Simple_Run")

        # 1. 设置 DB 存储路径：每个策略会在这里面创建自己的 UUID 子文件夹
        settings.file_load_path.chroma_db_file = os.path.join(simple_root, "chroma_db")
        # 2. 设置 JSON 导出路径
        settings.file_load_path.daily_memory_exports_file = os.path.join(simple_root, "exports")

        log.info(f"    📂 Simple DB 路径已隔离至: {settings.file_load_path.chroma_db_file}")

        # 3. 并发执行
        simple_tasks = [simple_framework(p) for p in sampled_policies]
        simple_results = await asyncio.gather(*simple_tasks)

        # 4. 保存汇总结果
        simple_summary = {
            "run_type": "simple",
            "policies": [{"ai": p.ai_threshold, "f": p.f_penalty, "edu": p.e_edu} for p in sampled_policies],
            "results": simple_results
        }
        save_experiment_results(simple_summary, simple_root)

        # =================================================================
        # 第二阶段：运行完整框架 (Complete Framework)
        # =================================================================
        log.info("\n" + "=" * 40 + "\n")
        log.info(">>> [Phase 2] 运行完整框架 (Complete Framework) <<<")

        # 【动态配置注入】将 Complete 的所有输出（包括DB）指向 experiment_root/Complete_Run
        # 这确保了 Complete 的数据库和 Simple 的数据库是完全物理隔离的两个文件夹
        complete_root = os.path.join(experiment_root, "Complete_Run")

        # 1. 设置 DB 存储路径
        settings.file_load_path.chroma_db_file = os.path.join(complete_root, "chroma_db")
        # 2. 设置 JSON 导出路径
        settings.file_load_path.daily_memory_exports_file = os.path.join(complete_root, "exports")

        log.info(f"    📂 Complete DB 路径已隔离至: {settings.file_load_path.chroma_db_file}")

        # semaphore = asyncio.Semaphore(5)
        # # 2. 定义一个受信号量控制的包装函数
        # async def limited_complete_framework(policy):
        #     async with semaphore:  # 只有拿到信号量（令牌）才能继续执行
        #         # 可以在这里加个日志，看看到底是不是在排队
        #         # log.info(f"🚦 开始执行策略仿真: {policy}")
        #         return await complete_framework(policy)
        #
        # # 3. 创建任务列表 (注意：这里调用的是包装后的 limited_complete_framework)
        # complete_tasks = [limited_complete_framework(p) for p in sampled_policies]
        #
        # # 4. 并发执行 (asyncio.gather 会自动调度，只有获得信号量的任务才会真正运行)
        # complete_results = await asyncio.gather(*complete_tasks)

        # 4. 保存汇总结果
        complete_summary = {
            "run_type": "complete",
            "policies": [{"ai": p.ai_threshold, "f": p.f_penalty, "edu": p.e_edu} for p in sampled_policies],
            "results": None
        }
        save_experiment_results(complete_summary, complete_root)

        # =================================================================
        # 结束
        # =================================================================
        log.info("=" * 60)
        log.info("✅ 实验圆满结束！")
        log.info(f"📊 Simple 数据位置:   {os.path.join(simple_root, 'exports')}")
        log.info(f"📊 Complete 数据位置: {os.path.join(complete_root, 'exports')}")
        log.info("=" * 60)

        # 打印分析命令提示
        print("\n现在可以运行分析脚本：")
        print(
            f"verify_trend_consistency(\n    r'{os.path.join(simple_root, 'exports')}',\n    r'{os.path.join(complete_root, 'exports')}'\n)")
    except:
        log.error("发生错误！跳过")
    finally:
        # 恢复原始配置，不影响其他代码
        settings.file_load_path.chroma_db_file = original_chroma_path
        settings.file_load_path.daily_memory_exports_file = original_export_path


async def framework_efficiency_experiment(population_size):
    """
    框架效率验证实验
    对比 Group A (Baseline: 全高粒度) 与 Group B (Proposed: 高低粒度耦合)


    问题：高低粒度耦合方法筛选出的精英解集数量 要不要和 全高粒度筛选出的解集数量  保持一致？？？

    """
    # 1. 实验初始化
    settings.nsga.population_size = population_size
    settings.nsga.generations = 2

    settings.platform.simple_run_days = 4
    settings.platform.complete_run_days = 4

    settings.platform.import_policy_day_time = 1
    settings.platform.kpi_window_size = 1

    # settings.file_load_path.personas_file = r'method/data/test_persona.json'
    settings.file_load_path.personas_file = r'method/data/inner_consistency_personas_20.json'

    # 备份原始的导出路径，实验结束后恢复
    original_export_path = settings.file_load_path.daily_memory_exports_file
    original_chroma_path = settings.file_load_path.chroma_db_file

    # 保存原始的时间类，用于恢复
    original_datetime_class = env_module.datetime.datetime

    # 创建实验总根目录
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    log.info("=" * 60)
    log.info(f"🚀 开始框架效率验证实验 (Time Freeze Mode)")
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
        # 2. 运行 Group A: Baseline
        # ==========================================
        group_a_dir = os.path.join(current_dir, "Group_A_全高粒度")
        log_path_a = setup_logger(group_a_dir)

        # --- 锁定时间 ---
        target_now_a = datetime.now()

        class FixedDatetimeA(datetime):
            @classmethod
            def now(cls, tz=None):
                return target_now_a

        env_module.datetime.datetime = FixedDatetimeA  # 打补丁
        log.info(f"🔒 [Group A] 时间已冻结为: {target_now_a.strftime('%H:%M:%S')}")

        settings.file_load_path.daily_memory_exports_file = os.path.join(group_a_dir, "sim_data")
        settings.file_load_path.chroma_db_file = os.path.join(group_a_dir, "chroma_db")

        log.info("\n>>> 阶段 1/2: 运行 Group A (Baseline - 全高粒度) <<<")
        log.info(f"📄 Group A 日志已重定向至: {log_path_a}")

        settings.platform.efficiency_validation = True
        clear_token_csv_file()
        start_time_a = time.perf_counter()

        # 运行仿真
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

        log.info(f"✅ Group A 完成: 耗时 {time_cost_a:.2f} 分钟")

        # ==========================================
        # 3. 运行 Group B: Proposed
        # ==========================================
        group_b_dir = os.path.join(current_dir, "Group_B_高低粒度")
        log_path_b = setup_logger(group_b_dir)

        target_now_b = datetime.now()

        class FixedDatetimeB(datetime):
            @classmethod
            def now(cls, tz=None):
                return target_now_b

        env_module.datetime.datetime = FixedDatetimeB  # 更新补丁
        log.info(f"🔒 [Group B] 时间已冻结为: {target_now_b.strftime('%H:%M:%S')}")

        settings.file_load_path.daily_memory_exports_file = os.path.join(group_b_dir, "sim_data")
        settings.file_load_path.chroma_db_file = os.path.join(group_b_dir, "chroma_db")

        log.info("\n>>> 阶段 2/2: 运行 Group B (Proposed - 高低粒度耦合) <<<")
        log.info(f"📄 Group B 日志已重定向至: {log_path_b}")

        clear_token_csv_file()
        start_time_b = time.perf_counter()
        settings.platform.efficiency_validation = False

        # (1) 进化阶段
        elites_b_temp, history_b = await nsga2_framework_evaluation(
            settings.nsga.population_size,
            settings.nsga.generations
        )

        # (2) 验证阶段
        log.info("   -> 对精英策略进行高粒度验证...")
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

        log.info(f"✅ Group B 完成: 耗时 {time_cost_b:.2f} 分钟")

        # ==========================================
        # 4. 汇总
        # ==========================================
        setup_logger(current_dir)

        speedup = time_cost_a / time_cost_b if time_cost_b > 0 else 0
        token_saving = (1 - token_cost_b / token_cost_a) * 100 if token_cost_a > 0 else 0

        log.info("=" * 60)
        log.info(f"📊 实验结果摘要")
        log.info(f"   加速比 (Speedup): {speedup:.2f}x")
        log.info(f"   成本节约 (Saving): {token_saving:.2f}%")
        log.info("=" * 60)

        save_experiment_results(results_summary, current_dir)

    finally:
        # 1. 解锁时间
        env_module.datetime.datetime = original_datetime_class
        # 2. 还原路径配置
        settings.file_load_path.daily_memory_exports_file = original_export_path
        settings.file_load_path.chroma_db_file = original_chroma_path

        print("🔓 [System] 仿真环境时间戳已解锁，路径已还原。")


async def complete_three_policies():
    await test_complete_three_policies()


async def simple_three_policies():
    await test_simple_three_policies()


async def effect_verification():
    """
    closed_loop_effectiveness_experiment
    :return:
    """
    # 1. 实验初始化
    settings.nsga.population_size = 20  # 初代数量
    settings.nsga.generations = 6  # 迭代次数

    settings.platform.simple_run_days = 15  # 低粒度运行天数
    settings.platform.complete_run_days = 15  # 高粒度运行天数

    settings.platform.kpi_window_size = 3  # KPI 窗口大小
    settings.platform.import_policy_day_time = 3  # 策略导入时间

    # settings.file_load_path.personas_file = r'method/data/personas_30.json'  # 人群文件
    # settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'
    settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'
    # settings.file_load_path.personas_file = r'method/data/personas_50.json'  # 人群文件
    # settings.file_load_path.personas_file = r'method/data/test_persona.json'  # 人群文件

    await high_low_evaluation()


async def adaptive_experiment_filter_elites(high: bool = True, open_high_model: bool = True):
    """
    策略自适应实验-筛选最优解
    high-true: 在反抗社会中，筛选最优解
    high-false：在顺从社会中，筛选最优解
    """
    settings.nsga.population_size = 20  # 初代数量
    settings.nsga.generations = 15  # 迭代次数

    settings.platform.simple_run_days = 15  # 低粒度运行天数
    settings.platform.complete_run_days = 15  # 高粒度运行天数

    settings.platform.kpi_window_size = 3  # KPI 窗口大小
    settings.platform.import_policy_day_time = 3  # 策略导入时间

    if high:
        settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'
    else:
        settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'

    await high_low_evaluation(open_high_simulation=False)  # 高低粒度都运行,一次性完成


async def adaptive_experiment_low_in_high(e_edu, f_penalty, ai_threshold):
    """
    策略自适应实验-将顺从社会筛选出的最优解放入反抗社会中
    :return:
    """
    settings.platform.complete_run_days = 15  # 高粒度运行天数

    settings.platform.kpi_window_size = 3  # KPI 窗口大小
    settings.platform.import_policy_day_time = 3  # 策略导入时间

    settings.file_load_path.personas_file = r'method/data/high_beta_personas_30.json'

    await run_complete_in_one_policy(e_edu, f_penalty, ai_threshold)


async def case_experiment():
    """
    案例验证实验  Turn on PAEP
    :return:
    """
    # 案例验证实验
    settings.platform.case_validation = True
    settings.platform.complete_run_days = 30
    settings.file_load_path.personas_file = 'method/data/scenario_protest.json'

    await run_complete_in_one_policy('低', 0.01, 0.8)


async def case_ablation_experiment():
    """
    案例验证消融实验-Turn off PAEP
    :return:
    """
    settings.platform.ablation_validation = True  # 开启消融实验-Turn off PAEP
    settings.platform.case_validation = True
    settings.platform.complete_run_days = 30
    settings.file_load_path.personas_file = 'method/data/scenario_protest-2.json'

    await run_complete_in_one_policy('低', 0.01, 0.8)


def low_model_effectiveness_experiment(policies_num):
    """
    低粒度模型有效性筛选实验
    :return:
    """
    settings.platform.complete_run_days = 15
    settings.platform.simple_run_days = 15
    settings.file_load_path.personas_file = r'method\data\trend_consistency_30.json'
    asyncio.run(low_model_effective_filter_experiment(policies_num))


async def read_lf_policy_run_complete(file_path: str):
    """
    读取低粒度模型筛选出的精英解集  放入高粒度模型中模拟
    :param file_path: result_data\20260118\134648
    :return:
    """
    with open(fr'{file_path}\实验数据\experiment_results.json', "r", encoding="utf-8") as f:
        results_data = json.load(f)

    elite_policy = results_data['elite_solutions']

    # 1. 确定本次实验的总目录 (使用真实的物理时间)
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)
    now = datetime.now()
    fixed_subdir = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
    # 2. 设置上下文变量 (返回一个 token，用于稍后重置，虽然脚本结束自动销毁也不必重置)
    token = current_sim_subdir.set(fixed_subdir)
    log.info(f"🔒 [Context] 已设置统一仿真目录: {fixed_subdir}")

    for i in elite_policy:
        print(i['policy'])
    try:
        if elite_policy:
            log.info(f" 精英策略精准评估开始 {len(elite_policy)}")

            settings.platform.simple_run_days = 15  # 低粒度运行天数
            settings.platform.complete_run_days = 15  # 高粒度运行天数

            settings.platform.kpi_window_size = 3  # KPI 窗口大小
            settings.platform.import_policy_day_time = 3  # 策略导入时间

            settings.file_load_path.personas_file = r'method/data/low_beta_personas_30.json'

            # 完整仿真框架评估
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
















