import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress
from typing import List, Tuple, Dict


def verify_power_law_fit(data_values: List[float], output_file_path) -> Tuple[float, float]:
    """
    幂律分布拟合
    计算一组数据的幂律拟合优度 (R^2) 和幂指数 (Alpha)。
    用于验证内容热度、粉丝数是否符合“长尾分布”。

    Args:
        data_values (List[float] or List[int]):
            需要分析的数值列表。
            - 类型: 浮点数或整数列表
            - 格式: [数值1, 数值2, 数值3, ...]
            - 约束: 列表长度应大于 10 以保证统计意义；无需预先排序；0值会被自动过滤。

    Example:
        # 假设这是从所有内容中提取出的点赞数列表
        input_data = [1024, 512, 128, 64, 32, 1, 0, 1, 5]
        r2, alpha = verify_power_law_fit(input_data)

    Returns:
        Tuple[float, float]: (R_squared, Alpha)
            - R_squared: 拟合优度 (0~1)，越接近 1 表示越符合幂律。
            - Alpha: 幂律指数，社交网络通常在 2.0~3.0 之间。
    """
    # 1. 数据清洗：过滤 0 和 负数 (log无法处理)，并从大到小排序
    clean_data = [float(x) for x in data_values if x > 0]
    if len(clean_data) < 2:
        print("Error: 有效数据量不足 (N < 2)，无法拟合。")
        return 0.0, 0.0

    clean_data.sort(reverse=True)

    # 2. 转换为双对数坐标 (Log-Log Plot)
    # X轴: 排名 (1, 2, 3...)
    # Y轴: 数值 (1024, 512...)
    ranks = np.arange(1, len(clean_data) + 1)
    log_ranks = np.log10(ranks)
    log_values = np.log10(clean_data)

    # 3. 线性回归拟合
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_values)

    r_squared = r_value ** 2
    alpha = -slope  # 斜率通常为负，Alpha取正值

    # 4. 绘图 (可选，用于调试)
    plt.figure(figsize=(6, 4))
    plt.scatter(log_ranks, log_values, alpha=0.5, s=10, label='Raw Data')
    plt.plot(log_ranks, slope * log_ranks + intercept, 'r--', label=f'Fit (R2={r_squared:.2f})')
    plt.xlabel('Log(Rank)')
    plt.ylabel('Log(Value)')
    plt.title(f'Power Law Verification (alpha={alpha:.2f})')
    plt.legend()

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
    plt.savefig(fr'{output_file_path}\幂律分布拟合.png', dpi=300, bbox_inches='tight')
    # 4. 关闭画布 (释放内存，防止程序“越跑越慢”)
    plt.close()

    return round(r_squared, 2), round(alpha, 2)


def compare_time_series_trends(timeline_a: List[float], timeline_b: List[float], metric_name: str, output_file_path) -> float:
    """
    KPI 时序趋势对比
    对比两组策略下某项 KPI 随时间的变化趋势，并计算平均差距。
    用于验证“严管 vs 放任”等极端测试的逻辑是否符合预期。

    Args:
        timeline_a (List[float]):
            策略 A (如: 放任策略) 在每一天产生的 KPI 数值列表。
            - 格式: [第1天数值, 第2天数值, ..., 第N天数值]
        timeline_b (List[float]):
            策略 B (如: 严管策略) 在每一天产生的 KPI 数值列表。
            - 格式: [第1天数值, 第2天数值, ..., 第N天数值]
            - 注意: timeline_a 和 timeline_b 的长度最好一致。
        metric_name (str):
            KPI 的名称，仅用于图表标题显示 (如 "Creativity", "Safety")。

    Example:
        # 模拟 5 天的创造力指数
        creativity_free = [0.8, 0.85, 0.90, 0.88, 0.92]
        creativity_strict = [0.8, 0.60, 0.40, 0.20, 0.10]
        diff = compare_time_series_trends(creativity_free, creativity_strict, "Creativity")

    Returns:
        float: 平均差异值 (Mean Difference)。
               如果是 "A - B"，正值表示 A 组表现更高。
    """
    # 确保长度对齐
    min_len = min(len(timeline_a), len(timeline_b))
    data_a = timeline_a[:min_len]
    data_b = timeline_b[:min_len]
    days = range(1, min_len + 1)

    # 1. 绘图对比
    plt.figure(figsize=(8, 4))
    plt.plot(days, data_a, label='Group A', marker='o', linestyle='--')
    plt.plot(days, data_b, label='Group B', marker='x', linestyle='-')
    plt.xlabel('Simulation Day')
    plt.ylabel(metric_name)
    plt.title(f'Mechanism Verification: {metric_name} Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok=True)
    plt.savefig(fr'{output_file_path}\{metric_name}.png', dpi=300, bbox_inches='tight')
    # 4. 关闭画布 (释放内存，防止程序“越跑越慢”)
    plt.close()

    # 2. 计算平均差距
    diffs = [a - b for a, b in zip(data_a, data_b)]
    avg_diff = sum(diffs) / len(diffs)

    return round(avg_diff, 2)


def calculate_clustering_coefficient(edges: List[Tuple[str, str]]) -> float:
    """
    网络真实性：聚类系数
    根据社交关系列表构建网络图，并计算平均聚类系数。
    用于验证社交网络是否存在“小圈子”特征 (Clustering)。

    Args:
        edges (List[Tuple[str, str]]):
            社交网络中的边列表。
            - 类型: 包含元组的列表，每个元组代表一条关注关系。
            - 格式: [(关注者ID, 被关注者ID), (A, B), (B, C), ...]
            - 注意: ID 必须是字符串或整数。

    Example:
        # 用户关系数据
        social_graph_edges = [
            ("user_1", "user_2"),
            ("user_2", "user_3"),
            ("user_3", "user_1"),  # 形成闭环三角形
            ("user_4", "user_5")
        ]
        coef = calculate_clustering_coefficient(social_graph_edges)

    Returns:
        float: 平均聚类系数 (0.0 ~ 1.0)。
               真实社交网络通常 > 0.1，随机网络通常接近 0。
    """
    if not edges:
        print("Error: 边列表为空，无法构建网络。")
        return 0.0

    # 1. 构建有向图
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 2. 计算平均聚类系数
    # NetworkX 的 average_clustering 会自动处理有向图
    # (有些版本可能会将其视为无向图计算，这在社会仿真基准中通常是可以接受的近似)
    try:
        avg_clustering = nx.average_clustering(G)
    except Exception as e:
        print(f"Calculation Error: {e}")
        return 0.0

    return round(avg_clustering, 2)  # 保留两位小数


def calculate_homophily_score(edges: List[Tuple[str, str]], node_attributes: Dict[str, str]) -> float:
    """
    社会动力学：同质性系数
    计算网络的属性同配系数 (Assortativity)。
    验证“物以类聚”现象：相同类型的用户是否倾向于相互关注？

    Args:
        edges (List[Tuple[str, str]]):
            社交网络的边列表 (同上)。
            - 格式: [("user_A", "user_B"), ("user_C", "user_A")]

        node_attributes (Dict[str, str]):
            节点属性字典。键是用户ID，值是用户类型/属性。
            - 格式: {"user_A": "Rebel", "user_B": "Rebel", "user_C": "Trust"}
            - 约束: 字典中的 key 必须涵盖 edges 中出现的节点 ID。

    Example:
        edges = [("u1", "u2"), ("u3", "u4")]
        # u1关注u2 (都是Rebel)，u3关注u4 (都是Trust) -> 高同质性
        attrs = {"u1": "Rebel", "u2": "Rebel", "u3": "Trust", "u4": "Trust"}
        score = calculate_homophily_score(edges, attrs)

    Returns:
        float: 同配系数 (-1.0 ~ 1.0)。
            - 正值 (>0): 同类相吸 (Homophily)。
            - 0: 随机连接。
            - 负值 (<0): 异类相吸 (Heterophily)。
    """
    if not edges or not node_attributes:
        return 0.0

    # 1. 构建图
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 2. 为节点绑定属性
    # 这一步至关重要：必须将字典中的属性赋值给图中的节点
    # 我们只给图中实际存在的节点赋值，避免报错
    valid_attrs = {n: node_attributes[n] for n in G.nodes() if n in node_attributes}
    nx.set_node_attributes(G, valid_attrs, 'user_type')

    # 3. 计算离散属性的同配系数
    try:
        # 'user_type' 是我们在上一步设定的属性键名
        score = nx.attribute_assortativity_coefficient(G, 'user_type')

        # 处理可能的 NaN (当网络完全不连通或属性完全一致导致分母为0时)
        if np.isnan(score):
            return 0.0

    except Exception as e:
        print(f"Error computing assortativity: {e}")
        return 0.0

    return round(score, 2)


def calculate_gini_coefficient(wealth_distribution: List[float]) -> float:
    """
    生态演化：基尼系数
    计算基尼系数，衡量资源分配的不平等程度。
    用于验证影响力是否呈现马太效应 (富者越富)。

    Args:
        wealth_distribution (List[float]):
            每个个体的资源数值列表 (如影响力、点赞总数、积分)。
            - 格式: [用户A影响力, 用户B影响力, 用户C影响力, ...]
            - 约束: 数值不能为负。

    Example:
        # 极度不平等：一个人拥有所有资源
        influences = [0, 0, 0, 0, 100]
        gini = calculate_gini_coefficient(influences) # 接近 1.0

    Returns:
        float: 基尼系数 (0.0 ~ 1.0)。
            - 0.0: 完全平等。
            - 1.0: 绝对不平等。
            - 0.4以上: 通常被认为是不平等警戒线。
    """
    # 1. 数据预处理
    wealths = [x for x in wealth_distribution if x >= 0]
    if not wealths or sum(wealths) == 0:
        return 0.0

    # 2. 排序 (基尼系数计算必须排序)
    wealths = sorted(wealths)
    n = len(wealths)

    # 3. 使用洛伦兹曲线面积公式计算
    # 公式: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    # 其中 i 是 1-based index (1, 2, ..., n)

    index = np.arange(1, n + 1)
    numerator = np.sum(index * wealths)
    denominator = n * np.sum(wealths)

    gini = (2 * numerator) / denominator - (n + 1) / n

    return round(float(gini), 2)



























