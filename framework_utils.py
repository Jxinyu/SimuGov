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
    每次运行时调用，用于重定向日志文件到新的文件夹。
    """
    # 1. 确定日志目录
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. 生成新的日志文件名
    log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # 3. 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- 【关键修改】暴力清空所有旧的处理器 ---
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            root_logger.removeHandler(handler)

    # --- 4. 重新添加：文件处理器 (FileHandler) ---
    file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(filename)s:%(lineno)d | %(funcName)-20s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- 5. 重新添加：控制台处理器 (StreamHandler) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"✅ 日志系统已重置。")
    root_logger.info(f"📁 日志文件路径: {log_filepath}")

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
    遗传算法进化
    """
    # 运行NSGA-II
    elite_solutions, evolution_history = await run_nsga2(population_size=population_size, generations=generations)
    log.info("\n--- 最终精英解决方案（帕累托最优集） ---")

    # 返回完整的精英解
    final_elites = []
    for i, solution in enumerate(elite_solutions):
        original_kpi = {k: -v for k, v in solution['kpi'].items()}
        log.info(f"Solution {i + 1}:")
        log.info(f"  Policy: {solution['policy']}")
        log.info(f"  KPIs (Simple): {original_kpi}")
        log.info("-" * 20)
        # 将恢复为正值的KPI也存入solution，方便后续使用
        solution['kpi'] = original_kpi
        final_elites.append(solution)

    return elite_solutions, evolution_history


async def complete_framework_evaluation(elite_solutions_from_nsga):
    """
    接收 NSGA-II找到的精英解列表，并为它们运行完整的仿真。
    """
    tasks = []
    for solution in elite_solutions_from_nsga:
        p = solution['policy']
        task = complete_framework(Policy(f_penalty=p['f_penalty'], e_edu=p['e_edu'], ai_threshold=p['ai_threshold']))
        tasks.append(task)

    # 并发执行所有完整仿真任务
    complete_kpi_results = await asyncio.gather(*tasks)

    # 将完整仿真结果与精英解数据合并
    for i, solution in enumerate(elite_solutions_from_nsga):
        solution['complete_kpi'] = complete_kpi_results[i]

    return elite_solutions_from_nsga


def data_save(experiment_data_to_save: dict, output_dir: str):
    """
    数据分析与保存
    """
    try:
        # 保存实验数据
        experiment_results_file_path = save_experiment_results(experiment_data_to_save, str(output_dir) + "/实验数据")
    except:
        experiment_results_file_path = save_experiment_results(experiment_data_to_save, str(output_dir))
    try:
        # nsga进化绘制
        nsga_evaluation_data_draw_main(experiment_results_file_path, str(output_dir) + "/nsga进化绘制")
    except:
        nsga_evaluation_data_draw_main(experiment_results_file_path, str(output_dir))

    # 绘制各类图表
    path_obj = Path(output_dir)
    draw_kpi_main(plot_mode='T', use_simplified_data=True, output_dir=path_obj / "简化框架kpi绘制")
    draw_kpi_main(plot_mode='T', use_simplified_data=False, output_dir=path_obj / "完整框架kpi绘制")
    draw_kpi_main(plot_mode='P', use_simplified_data=True, output_dir=path_obj / "简化框架的帕累托前沿绘制")
    draw_kpi_main(plot_mode='P', use_simplified_data=False, output_dir=path_obj / "最终的帕累托前沿绘制")

    # 写入终止时间
    with open(path_obj / "termination_time.txt", "w") as f:
        f.write(str(datetime.now()))


async def test_complete_three_policies():
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    # 系统有效性闭环实验
    settings.platform.complete_run_days = 20  # 高粒度运行天数
    settings.platform.kpi_window_size = 3  # KPI 窗口大小
    settings.platform.import_policy_day_time = 3  # 策略导入时间
    settings.file_load_path.personas_file = r'method/data/scenario_protest.json'

    policy1 = Policy(e_edu='低', f_penalty=0.01, ai_threshold=0.99)
    policy2 = Policy(e_edu='中', f_penalty=0.5, ai_threshold=0.5)
    policy4 = Policy(e_edu='低', f_penalty=0.99, ai_threshold=0.01)
    policy3 = Policy(e_edu='高', f_penalty=0.99, ai_threshold=0.01)  # 创造力低、安全性高
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

    settings.platform.kpi_window_size = 3  # KPI 窗口大小
    settings.platform.import_policy_day_time = 2  # 策略导入时间

    try:
        policy1 = Policy(e_edu='低', f_penalty=0.01, ai_threshold=0.99)  # 创造力高、安全性低
        # policy2 = Policy(e_edu='中', f_penalty=0.6, ai_threshold=0.6)
        # policy4 = Policy(e_edu='低', f_penalty=0.2, ai_threshold=0.2)
        policy3 = Policy(e_edu='高', f_penalty=0.99, ai_threshold=0.01)  # 创造力低、安全性高
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
    测试一个政策在高粒度中效果实验

    这个函数里不设置任何参数

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
    完整的主流程，包括计时、执行和保存。
    open_high_simulation: 开启高粒度模型
    """
    # 1. 确定本次实验的总目录 (使用真实的物理时间)
    current_dir = get_current_timestamp_dir()
    setup_logger(current_dir)

    now = datetime.now()
    fixed_subdir = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"

    # 2. 设置上下文变量 (返回一个 token，用于稍后重置，虽然脚本结束自动销毁也不必重置)
    token = current_sim_subdir.set(fixed_subdir)

    log.info(f"🔒 [Context] 已设置统一仿真目录: {fixed_subdir}")

    try:
        run_params = {
            "population_size": settings.nsga.population_size,  # 初始种群大小
            "generations": settings.nsga.generations  # 迭代轮数
        }
        timings = {}
        total_start_time = time.perf_counter()

        log.info("阶段一: NSGA-II 快速寻优开始")
        phase1_start_time = time.perf_counter()

        # 遗传算法进化
        elite_solutions, evolution_history = await nsga2_framework_evaluation(
            population_size=settings.nsga.population_size,
            generations=settings.nsga.generations
        )

        print()

        phase1_end_time = time.perf_counter()
        timings["phase1_nsga2_duration_seconds"] = round(phase1_end_time - phase1_start_time, 2) / 60
        log.info(f"阶段一完成, 耗时: {timings['phase1_nsga2_duration_seconds']:.2f} 分钟 ")

        final_results_data = []
        if elite_solutions and open_high_simulation:
            log.info(f" 阶段二: 精英策略精准评估开始 {len(elite_solutions)}")
            phase2_start_time = time.perf_counter()

            # 完整仿真框架评估
            final_results_data = await complete_framework_evaluation(elite_solutions)

            phase2_end_time = time.perf_counter()
            timings["phase2_complete_sim_duration_seconds"] = round(phase2_end_time - phase2_start_time, 2) / 60
            log.info(f"阶段二完成, 耗时: {timings['phase2_complete_sim_duration_seconds']:.2f} 分钟 ")

        total_end_time = time.perf_counter()
        timings["total_duration_seconds"] = round(total_end_time - total_start_time, 2) / 60
        log.info(f"\n=== 全部流程结束, 总耗时: {timings['total_duration_seconds']:.2f} 分钟 ===")

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
    构建智能体池子
    :return:
    """
    from method.utils.get_personas.build_artstation_dataset_poolsl import build_agent_pools_demo
    build_agent_pools_demo(100, 100, 100)


def build_artstation_personas():
    """
    构建artstaion的personas数据集
    :return:
    """
    build_artstation_personas_main()



















