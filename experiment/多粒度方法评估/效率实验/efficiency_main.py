import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pygmo as pg

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def load_experiment_data(base_dir: str) -> Dict[str, Any]:
    """
    【加载根目录下的 experiment_results.json 汇总文件。
    """
    path = Path(base_dir)
    summary_path = path / "experiment_results.json"

    data = {}

    if not summary_path.exists():
        log.error(f"❌ 找不到汇总文件: {summary_path}")
        return {}

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)

        # 1. 提取实验参数
        data["parameters"] = raw_json.get("parameters", {})

        # 2. 提取 Group A 数据
        # 注意：这里使用 get 避免 key 不存在报错，给予默认空字典
        groups = raw_json.get("groups", {})
        data["group_a"] = groups.get("Group_A_Baseline", {})
        data["group_b"] = groups.get("Group_B_Proposed", {})

        # 简单校验
        if not data["group_a"] or not data["group_b"]:
            log.warning("⚠️ 警告：汇总文件中缺少 Group A 或 Group B 的数据，分析可能不完整。")

    except Exception as e:
        log.error(f"❌ 读取 JSON 文件失败: {e}")
        return {}

    return data


def calculate_efficiency_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """计算效率指标"""
    time_a = data["group_a"].get("time_minutes", 0)
    time_b = data["group_b"].get("time_minutes", 0)
    token_a = data["group_a"].get("token_cost", 0)
    token_b = data["group_b"].get("token_cost", 0)

    pop_size = data["parameters"].get("population", 1)
    generations = data["parameters"].get("generations", 1)
    total_evaluations = pop_size * generations

    metrics = {
        "time_a": time_a,
        "time_b": time_b,
        "token_a": token_a,
        "token_b": token_b
    }

    # 计算指标
    metrics["speedup"] = time_a / time_b if time_b > 0 else 0.0
    metrics["token_saving"] = (1 - (token_b / token_a)) * 100 if token_a > 0 else 0.0

    if total_evaluations > 0:
        metrics["latency_a"] = (time_a * 60) / total_evaluations
        metrics["latency_b"] = (time_b * 60) / total_evaluations
    else:
        metrics["latency_a"] = 0.0
        metrics["latency_b"] = 0.0

    return metrics


def calculate_quality_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """计算质量指标"""

    def extract_kpi_points(group_data: Dict) -> List[List[float]]:
        # --- Fix: 兼容不同的 Key 名称 ---
        elites = group_data.get("elite_solutions") or \
                 group_data.get("elite_solutions_results") or \
                 []

        points = []
        for ind in elites:
            # 优先取高粒度验证值 complete_kpi
            kpi_dict = ind.get("complete_kpi")
            if not kpi_dict:
                kpi_dict = ind.get("kpi", {})

            if not kpi_dict: continue

            # 转为正数 [0,1]
            s = abs(kpi_dict.get("safety", 0))
            c = abs(kpi_dict.get("creativity", 0))
            sat = abs(kpi_dict.get("satisfaction", 0))
            points.append([s, c, sat])
        return points

    points_a = extract_kpi_points(data.get("group_a", {}))
    points_b = extract_kpi_points(data.get("group_b", {}))

    metrics = {
        "count_a": len(points_a),
        "count_b": len(points_b)
    }

    # 计算 HV (目标最小化 [-s, -c, -sat]，参考点 [0,0,0])
    ref_point = [0.0, 0.0, 0.0]

    def compute_hv(points):
        if not points: return 0.0
        neg_points = [[-p[0], -p[1], -p[2]] for p in points]
        return pg.hypervolume(neg_points).compute(ref_point)

    metrics["hv_a"] = compute_hv(points_a)
    metrics["hv_b"] = compute_hv(points_b)
    metrics["hv_ratio"] = metrics["hv_b"] / metrics["hv_a"] if metrics["hv_a"] > 0 else 0.0

    return metrics


def save_analysis_summary_human_readable(efficiency: Dict, quality: Dict, output_dir: str):
    """
    保存分析报告。
    """

    # 构造详尽的报告结构
    report = {
        "0_Meta_信息": {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据来源": output_dir
        },

        "1_效率评估": {
            "1.1_时间成本对比": {
                "全高组耗时": f"{efficiency['time_a']:.2f} 分钟",
                "高低组耗时": f"{efficiency['time_b']:.2f} 分钟",
                "加速比": {
                    "数值": f"{efficiency['speedup']:.2f} x",
                    "解释": "实验组比基准组快多少倍 (越高越好)"
                }
            },
            "1.2_算力成本对比": {
                "全高组Token": f"{int(efficiency['token_a']):,} Tokens",
                "高低组Token": f"{int(efficiency['token_b']):,} Tokens",
                "节约率": {
                    "数值": f"{efficiency['token_saving']:.2f} %",
                    "解释": "节省了多少百分比的Token成本 (越高越好)"
                }
            },
            "1.3_扩展性指标": {
                "全高组单策略延迟": f"{efficiency['latency_a']:.2f} 秒/策略",
                "高低组单策略延迟": f"{efficiency['latency_b']:.2f} 秒/策略",
                "解释": "平均评估一个策略所需的物理时间 (排除实验规模影响)"
            }
        },

        "2_质量评估": {
            "2.1_解集数量": {
                "全高组精英数": efficiency.get('elite_count_a', quality['count_a']),  # 兼容写法
                "高低组精英数": efficiency.get('elite_count_b', quality['count_b'])
            },
            "2.2_超体积 ": {
                "全高组 HV": f"{quality['hv_a']:.4f}",
                "高低组 HV": f"{quality['hv_b']:.4f}",
                "解释": "衡量帕累托前沿的综合质量 (收敛性+多样性)"
            },
            "2.3_一致性指标": {
                "HV保持率": {
                    "数值": f"{quality['hv_ratio'] * 100:.2f} %",
                    "解释": "实验组复现了基准组多少质量。接近 100% 表示完美，>85% 为优秀。"
                }
            }
        },

        "3_一句话结论": (
            f"本实验中，高低粒度耦合框架实现了 {efficiency['speedup']:.1f} 倍的时间加速，"
            f"并节省了 {efficiency['token_saving']:.1f}% 的算力成本。"
            f"解集质量保持率为 {quality['hv_ratio'] * 100:.1f}%。"
        )
    }

    target_path = Path(output_dir) / f"analysis_summary_{str(datetime.now()).replace(':', '-').replace('.', '-')}.json"
    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 让中文正常显示，indent=4 让格式美观
            json.dump(report, f, indent=4, ensure_ascii=False)
        log.info(f"✅ 人类可读报告已保存至: {target_path}")
    except Exception as e:
        log.error(f"❌ 保存报告失败: {e}")


def print_analysis_report(
        efficiency: Dict[str, float],
        quality: Dict[str, Any],
        path: str
):
    """
    打印格式化的分析报告。
    """
    print("\n" + "=" * 60)
    print(f"📊 框架效率实验分析报告")
    print(f"📁 数据源: {path}")
    print("=" * 60)

    print(f"\n【1. 效率评估】")
    print(f"--------------------------------------------------")
    print(f"⏱️  总耗时对比:")
    print(f"    - Baseline (A): {efficiency['time_a']:.2f} 分钟")
    print(f"    - Proposed (B): {efficiency['time_b']:.2f} 分钟")
    print(f"🚀 时间加速比: {efficiency['speedup']:.2f} x")

    print(f"\n💰 算力消耗对比:")
    print(f"    - Baseline (A): {efficiency['token_a']:,} Tokens")
    print(f"    - Proposed (B): {efficiency['token_b']:,} Tokens")
    print(f"📉 Token 节约率: {efficiency['token_saving']:.2f} %")

    print(f"\n⚡ 单策略推理延迟 :")
    print(f"    - Baseline: {efficiency['latency_a']:.2f} 秒/策略")
    print(f"    - Proposed: {efficiency['latency_b']:.2f} 秒/策略")

    print(f"\n【2. 质量评估】")
    print(f"--------------------------------------------------")
    print(f"🎯 精英解数量:")
    print(f"    - Baseline: {quality['count_a']} 个")
    print(f"    - Proposed: {quality['count_b']} 个")

    print(f"\n📐 超体积:")
    print(f"    - Baseline HV: {quality['hv_a']:.4f}")
    print(f"    - Proposed HV: {quality['hv_b']:.4f}")
    print(f"⚖️  HV 保持率: {quality['hv_ratio'] * 100:.2f} %")

    print("=" * 60)
    print("结论建议:")
    if efficiency['speedup'] > 10 and quality['hv_ratio'] > 0.8:
        print("✅ 实验成功：在保持了 80% 以上解质量的同时，实现了 10 倍以上的效率提升。")
    elif efficiency['speedup'] < 5:
        print("⚠️ 警告：加速比不明显，请检查 Group B 是否真的走了低粒度路径。")
    elif quality['hv_ratio'] < 0.5:
        print("⚠️ 警告：解质量损失严重，低粒度模型可能未能正确捕捉梯度方向。")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # r'result_data\20260107\165500'
    input_path = r'experiment\多粒度方法评估\效率实验\data\155812'
    output_path = r'experiment\多粒度方法评估\效率实验\output'

    if os.path.exists(input_path):
        # 1. 加载数据
        raw_data = load_experiment_data(input_path)

        # 2. 计算指标
        eff_metrics = calculate_efficiency_metrics(raw_data)
        qual_metrics = calculate_quality_metrics(raw_data)

        # 3. 打印报告 (控制台可见)
        print_analysis_report(eff_metrics, qual_metrics, input_path)

        # 4. 【新增】保存结果到文件
        save_analysis_summary_human_readable(eff_metrics, qual_metrics, output_path)
    else:
        print(f"❌ 路径不存在: {input_path}")
