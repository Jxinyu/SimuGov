import numpy as np
import random
from typing import List, Dict, Tuple
from method.agent.persona import Persona
from method.environment import Environment


class GroupManager:
    """
    负责将智能体进行聚类分组，并生成群体的宏观状态描述。
    """

    @staticmethod
    def _get_dominant_standpoint(persona: Persona) -> str:
        """根据概率分布确定主导立场"""
        idx = np.argmax(persona.standpoint)
        mapping = {0: "信任派(Trust)", 1: "反抗派(Rebel)", 2: "中立派(Neutral)"}
        return mapping.get(idx, "中立派(Neutral)")

    @staticmethod
    def cluster_creators(environment: Environment) -> Dict[str, List[Persona]]:
        """
        对创作者进行分组。
        """
        groups = {}
        for p in environment.personas.values():
            if not p.is_active:
                continue
            if p.type == '合规创作者':
                key = f"合规创作者_{p.fp_sensitivity}敏感度"
            elif p.type == '水印破坏者':
                key = f"水印破坏者_{p.cost_sensitivity}成本敏感"
            else:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(p)
        return groups

    @staticmethod
    def cluster_public(environment: Environment) -> Dict[str, List[Persona]]:
        """
        对公众进行分组。
        """
        groups = {}
        for p in environment.personas.values():
            if not p.is_active or p.type == '水印破坏者':
                continue
            standpoint = GroupManager._get_dominant_standpoint(p)
            key = f"公众_{standpoint}_逆反心理({p.beta})"
            if key not in groups:
                groups[key] = []
            groups[key].append(p)
        return groups

    @staticmethod
    def get_representative_sample(agents: List[Persona], ratio: float = 0.3) -> Tuple[List[Persona], List[Persona]]:
        """
        返回: (代表列表, 跟随者列表)
        """
        if not agents:
            return [], []

        # === 核心修改：生存红线===
        # 当幸存者少于 6 人时，为了保住社区火种，强制全员晋升为代表。
        # 这样每个人都拥有独立思考能力(LLM)，不再依赖模仿。
        CRITICAL_SURVIVAL_COUNT = 6

        if len(agents) <= CRITICAL_SURVIVAL_COUNT:
            # log.info(f"⚠️ 群体规模({len(agents)})触发生存红线，全员晋升为代表。")
            return agents, []  # 全员代表，无跟随者

        # === 正常采样逻辑 ===
        # 1. 至少抽取 1 人
        sample_size = max(1, int(len(agents) * ratio))

        # 2. 按影响力排序 (模拟上位机制：大V走了，腰部用户自动变成头部)
        sorted_agents = sorted(agents, key=lambda x: x.influence, reverse=True)

        # 3. 选取前 50% 名额给高影响力者
        top_k = max(1, int(sample_size * 0.5))
        representatives = sorted_agents[:top_k]

        # 4. 剩余名额随机
        candidates = sorted_agents[top_k:]
        if candidates:
            remaining_slots = sample_size - len(representatives)
            if remaining_slots > 0:
                representatives.extend(random.sample(candidates, remaining_slots))

        # 5. 确定跟随者 (Set去重最稳健)
        rep_ids = {p.agent_id for p in representatives}
        followers = [p for p in agents if p.agent_id not in rep_ids]

        return representatives, followers

    @staticmethod
    def get_group_stats_prompt(group_name: str, agents: List[Persona]) -> str:
        """保持原样"""
        count = len(agents)
        if count == 0: return ""
        avg_satisfaction = np.mean([p.satisfaction[-1] if p.satisfaction else 0 for p in agents])
        post_wish_rate = np.mean([1 if p.post_wish else 0 for p in agents])
        return f"""
        【群体名称】: {group_name}
        【群体规模】: {count} 人
        【平均满意度】: {avg_satisfaction:.2f}
        【当前发布意愿率】: {post_wish_rate:.1%}
        【典型画像特征】: {agents[0].description[:100]}...
        """
