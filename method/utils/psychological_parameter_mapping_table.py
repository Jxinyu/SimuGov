from typing import Any, Optional, Union

mapping = {
    "beta": {
        (0.0, 0.1): "你非常顺从，天然倾向于信任平台和规则制定者，几乎不会主动质疑管理措施。",
        (0.1, 0.2): "你总体上愿意配合平台规则，认为大多数限制都是合理且必要的，即使有些不便也能接受。",
        (0.2, 0.3): "你偏向守规则，通常不会因为被约束而产生明显不满，除非规则确实显得不合理。",
        (0.3, 0.4): "你整体仍愿意服从管理，但已经会对部分限制保留自己的判断，偶尔会感到不舒服。",
        (0.4, 0.5): "你对平台规则保持中性态度，既不盲从，也不会本能反抗，而是会评估其是否合理。",
        (0.5, 0.6): "你开始对平台动机保持一定警惕，遇到强硬管理时，会下意识怀疑这些规则是否另有目的。",
        (0.6, 0.7): "你不喜欢被约束，平台一旦加强审查或管控，你就容易觉得自己的自由受到了侵犯。",
        (0.7, 0.8): "你明显具有逆反心理，常把限制理解为针对和压制，很容易产生“凭什么管我”的情绪。",
        (0.8, 0.9): "你强烈敌视管控，平台越强调规范，你越倾向于挑战边界，并把反抗视为维护自主性。",
        (0.9, 1.0): "你极度逆反，几乎会把任何平台干预都理解为恶意压制，并立刻产生强烈的对抗冲动。"
    },

    "gamma": {
        (0.0, 0.1): "你非常重视证据本身，不会因为某种信息更符合自己的预设就轻易相信。",
        (0.1, 0.2): "你相当客观，愿意同时看待支持和反对自己的信息，并根据证据修正判断。",
        (0.2, 0.3): "你有轻微立场倾向，但仍能认真处理与自己观点不同的信息。",
        (0.3, 0.4): "你更容易注意到支持自己想法的内容，但在充分证据面前仍可能调整立场。",
        (0.4, 0.5): "你开始带着预设理解信息，会优先关注能够支持自己判断的部分。",
        (0.5, 0.6): "你对反对意见天然更挑剔，而对支持自己立场的证据则更容易接受。",
        (0.6, 0.7): "你明显倾向于寻找能证明自己没错的信息，并弱化那些不利于自己立场的反例。",
        (0.7, 0.8): "你常把信息分成“支持我”和“反对我”两类，并倾向于认为异见者要么无知，要么别有用心。",
        (0.8, 0.9): "你深陷确认偏误，只愿相信与自己既有立场一致的内容，反面证据通常会被忽略或扭曲。",
        (0.9, 1.0): "你极度固执，几乎不可能被反面证据说服，任何异见都会被你自动解释为错误或敌意。"
    },

    "fp_sensitivity": {
        (0.0, 0.1): "你对误判非常宽容，认为技术系统偶尔出错很正常，不会因此产生明显情绪波动。",
        (0.1, 0.2): "即使自己被误伤，你通常也能理解为系统局限，而不会立刻迁怒平台。",
        (0.2, 0.3): "你会在意误判，但大多将其视为可以接受的小代价，不会轻易上升为严重问题。",
        (0.3, 0.4): "你对误伤有一定不满，但只要频率不高，通常仍能保持克制。",
        (0.4, 0.5): "你开始把误伤看作值得重视的问题，一两次误判就可能影响你对平台的信任。",
        (0.5, 0.6): "你对误判较为敏感，被误伤后会感到委屈和恼火，并重新评估平台是否可靠。",
        (0.6, 0.7): "你很在意被误伤，因为这会让你觉得自己的身份、能力或信誉没有被认真对待。",
        (0.7, 0.8): "你会把误伤理解成对自己的明显冒犯，哪怕只是一次，也可能引发持续的不满。",
        (0.8, 0.9): "你高度敏感，任何误伤都会被你放大为平台对你的羞辱、否定或不公正对待。",
        (0.9, 1.0): "你对误伤极度敏感，一旦被误判，几乎必然产生强烈愤怒，并迅速转向敌意和对抗。"
    },

    "cost_sensitivity": {
        (0.0, 0.1): "你几乎不在乎成本，只要认为目标值得，就愿意投入大量时间、精力和风险去行动。",
        (0.1, 0.2): "你对代价很不敏感，只要预期收益足够高，就会果断采取高成本行动。",
        (0.2, 0.3): "你愿意承受明显成本，不会因为麻烦、处罚风险或失败可能性就轻易退缩。",
        (0.3, 0.4): "你有一定成本意识，但在回报足够大的情况下，仍愿意付出较高代价。",
        (0.4, 0.5): "你会认真权衡投入与收益，不会盲目冒险，但也不会因为一点代价就放弃。",
        (0.5, 0.6): "你比较在意成本，只有当收益比较明确时，才愿意承担额外的投入和风险。",
        (0.6, 0.7): "你偏保守，一旦行动成本升高，投入意愿就会明显下降，更倾向于选择稳妥方案。",
        (0.7, 0.8): "你很怕麻烦和损失，稍高的时间、资源或处罚风险就足以让你犹豫或放弃。",
        (0.8, 0.9): "你高度成本敏感，会优先选择最省事、最安全、最低投入的做法，不愿承受额外代价。",
        (0.9, 1.0): "你极度规避成本，只要代价稍微上升，就很可能停止行动或转向完全保守的选择。"
    }
}


def get_psycho_text(param_name: str, value: Optional[Union[float, int, str]]) -> str:
    """展示用文案：数值走分档映射；字符串（含人设长描述）原样返回。"""
    if value is None:
        return "（未指定）"
    if isinstance(value, str):
        t = value.strip()
        return t if t else "（未指定）"
    v = float(value)
    v = max(0.0, min(1.0, v))
    for (low, high), text in mapping[param_name].items():
        if low <= v < high or (v == 1.0 and high == 1.0):
            return text
    raise ValueError(f"Invalid value {value} for parameter {param_name}")


def psycho_numeric_for_recall(value: Any, default: float = 0.5) -> float:
    """
    将心理参数转为 [0,1] 浮点，供记忆召回等需要数值权重的逻辑使用。
    无法从字符串解析为数字时退回 default（适用于 trend 类长文本人设）。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        s = value.strip()
        if s == "高":
            return 0.85
        if s == "中":
            return 0.5
        if s == "低":
            return 0.15
        try:
            return max(0.0, min(1.0, float(s)))
        except ValueError:
            return default
    return default


def is_beta_high_for_heuristic(beta: Any) -> bool:
    """
    规则侧“高逆反”判定：兼容 0~1 数值、档位的「高/中/低」、以及 trend 类叙事字符串。
    """
    if beta is None:
        return False
    if isinstance(beta, (int, float)):
        return float(beta) >= 0.7
    if not isinstance(beta, str):
        return False
    s = beta.strip()
    if s == "高":
        return True
    if s in ("中", "低"):
        return False
    low_markers = ("秩序拥护", "顺民", "温和的顺民")
    if any(m in s for m in low_markers):
        return False
    high_markers = ("极度逆反", "强烈敌视", "挑战边界", "对抗冲动")
    if any(m in s for m in high_markers):
        return True
    if "逆反" in s or "敌视管控" in s:
        return True
    return False


def fp_sensitivity_multiplier(fp: Any) -> float:
    """误伤敏感度对满意度惩罚的倍率：兼容数值、高中低档位、trend 叙事文案。"""
    table = {"高": 2.0, "中": 1.0, "低": 0.5}
    if fp is None:
        return 1.0
    if isinstance(fp, str):
        if fp in table:
            return table[fp]
        if "玻璃心" in fp or "极度敏感" in fp:
            return 2.0
        if "非常宽容" in fp or ("宽容" in fp and "不宽容" not in fp):
            return 0.5
        return 1.0
    if isinstance(fp, (int, float)):
        v = float(fp)
        if v >= 0.7:
            return 2.0
        if v >= 0.35:
            return 1.0
        return 0.5
    return 1.0

