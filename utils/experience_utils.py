import random
from typing import List
from method.environment import Policy
from nsga.lhs import latin_hypercube_sampling


def generate_policies_low_screening(policies_num=10) -> List[Policy]:
    """
    随机生成多组实验策略用于趋势一致性验证。

    Returns:
        List[Policy]: 包含生成的 Policy 对象的列表。
    """
    policies1 = [
        Policy(f_penalty=0.0, e_edu='低', ai_threshold=0.1),
        Policy(f_penalty=0.1, e_edu='中', ai_threshold=0.9),
        Policy(f_penalty=0.2, e_edu='低', ai_threshold=0.3),
        Policy(f_penalty=0.4, e_edu='中', ai_threshold=0.2),
        Policy(f_penalty=0.5, e_edu='中', ai_threshold=0.5),
        Policy(f_penalty=0.5, e_edu='低', ai_threshold=0.6),
        Policy(f_penalty=0.5, e_edu='高', ai_threshold=0.9),
        Policy(f_penalty=0.8, e_edu='中', ai_threshold=0.7),
        Policy(f_penalty=0.9, e_edu='低', ai_threshold=0.1),
        Policy(f_penalty=1.0, e_edu='高', ai_threshold=0.9),
    ]

    policies2 = [
        Policy(ai_threshold=0.1, f_penalty=0.0, e_edu='低'),
        Policy(ai_threshold=0.1, f_penalty=0.9, e_edu='低'),
        Policy(ai_threshold=0.2, f_penalty=0.4, e_edu='中'),
        Policy(ai_threshold=0.3, f_penalty=0.2, e_edu='低'),
        Policy(ai_threshold=0.3, f_penalty=0.8, e_edu='低'),
        Policy(ai_threshold=0.5, f_penalty=0.5, e_edu='中'),
        Policy(ai_threshold=0.5, f_penalty=0.8, e_edu='中'),
        Policy(ai_threshold=0.9, f_penalty=0.1, e_edu='中'),
        Policy(ai_threshold=0.4, f_penalty=0.1, e_edu='中'),
        Policy(ai_threshold=0.7, f_penalty=0.8, e_edu='中'),
        Policy(ai_threshold=0.6, f_penalty=0.5, e_edu='低'),
        Policy(ai_threshold=0.2, f_penalty=0.5, e_edu='中'),
        Policy(ai_threshold=0.9, f_penalty=1.0, e_edu='高'),
        Policy(ai_threshold=0.8, f_penalty=0.6, e_edu='高'),
        Policy(ai_threshold=0.9, f_penalty=0.5, e_edu='高'),
        Policy(ai_threshold=0.2, f_penalty=0.5, e_edu='低')
    ]

    policies3 = [
        Policy(f_penalty=0.0, e_edu='低', ai_threshold=0.1),
        Policy(f_penalty=0.1, e_edu='中', ai_threshold=0.9),
        Policy(f_penalty=0.2, e_edu='低', ai_threshold=0.3),
        Policy(f_penalty=0.4, e_edu='中', ai_threshold=0.2),
        Policy(f_penalty=0.5, e_edu='中', ai_threshold=0.5),
        Policy(f_penalty=0.5, e_edu='低', ai_threshold=0.6),
        Policy(f_penalty=0.5, e_edu='高', ai_threshold=0.9),
        Policy(f_penalty=0.8, e_edu='中', ai_threshold=0.7),
        Policy(f_penalty=0.9, e_edu='低', ai_threshold=0.1),
        Policy(f_penalty=1.0, e_edu='高', ai_threshold=0.9),
        Policy(f_penalty=0.0, e_edu='高', ai_threshold=0.5),
        Policy(f_penalty=0.1, e_edu='低', ai_threshold=0.7),
        Policy(f_penalty=0.2, e_edu='高', ai_threshold=0.1),
        Policy(f_penalty=0.3, e_edu='高', ai_threshold=0.8),
        Policy(f_penalty=0.6, e_edu='低', ai_threshold=0.2),
        Policy(f_penalty=0.6, e_edu='中', ai_threshold=0.9),
        Policy(f_penalty=0.7, e_edu='高', ai_threshold=0.4),
        Policy(f_penalty=0.8, e_edu='低', ai_threshold=0.8),
        Policy(f_penalty=0.0, e_edu='中', ai_threshold=0.3),
        Policy(f_penalty=1.0, e_edu='中', ai_threshold=0.1),
    ]

    if policies_num == 10:
        return policies1
    elif policies_num == 15:
        return policies2
    elif policies_num == 20:
        return policies3

    return policies1


def generate_policies_baseline(policies_num=10) -> List[Policy]:
    """生成基准对比策略

    这组策略用的是 分层随机（stratified random）+ 空间填充（space-filling） 的思路，不是最朴素的纯随机 iid 采样。

        简要说就是三步：

        先对教育强度分层
        把 e_edu 按“低 / 中 / 高”分开采样，避免 10 个点都落在同一个教育档位。
        再对连续参数做分散覆盖
        对 ai_threshold 和 f_penalty 不让它们扎堆，而是尽量覆盖低、中、高不同区域，保证有边界点，也有中间点。
        优先选彼此距离较远的点
        后续点尽量和已选点拉开距离，这样 10 个策略能更像一个小型 random-search baseline，而不是几组很相似的参数。

        所以它更准确地说是：

        固定预算下的、覆盖性更强的随机策略集
    """

    policies = [
        Policy(f_penalty=0.94, e_edu='低', ai_threshold=0.33),
        Policy(f_penalty=0.46, e_edu='中', ai_threshold=0.42),
        Policy(f_penalty=0.14, e_edu='高', ai_threshold=0.94),
        Policy(f_penalty=0.99, e_edu='高', ai_threshold=0.90),
        Policy(f_penalty=0.05, e_edu='低', ai_threshold=0.99),
        Policy(f_penalty=0.99, e_edu='高', ai_threshold=0.01),
        Policy(f_penalty=0.02, e_edu='低', ai_threshold=0.02),
        Policy(f_penalty=0.75, e_edu='低', ai_threshold=0.98),
        Policy(f_penalty=0.41, e_edu='中', ai_threshold=0.99),
        Policy(f_penalty=0.98, e_edu='中', ai_threshold=0.64),
    ]

    return policies



















