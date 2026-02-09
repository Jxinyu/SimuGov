import random
from typing import List
from method.environment import Policy
from nsga.lhs import latin_hypercube_sampling


def generate_experimental_policies(policies_num=10) -> List[Policy]:
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

    population = latin_hypercube_sampling(n_samples=policies_num)
    policies3 = [Policy(ai_threshold=p['ai_threshold'], f_penalty=p['f_penalty'], e_edu=p['f_penalty']) for p in population]

    return policies1
