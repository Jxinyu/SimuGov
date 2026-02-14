import numpy as np
import random
from typing import List, Dict, Tuple
from method.agent.persona import Persona
from method.environment import Environment


class GroupManager:
    """
    Responsible for clustering agents and generating macro-state descriptions for groups.
    """

    @staticmethod
    def _get_dominant_standpoint(persona: Persona) -> str:
        """Determine the dominant standpoint based on the probability distribution"""
        idx = np.argmax(persona.standpoint)
        mapping = {0: "信任派(Trust)", 1: "反抗派(Rebel)", 2: "中立派(Neutral)"}
        return mapping.get(idx, "中立派(Neutral)")

    @staticmethod
    def cluster_creators(environment: Environment) -> Dict[str, List[Persona]]:
        """
        Group the creators.
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
        Group the public.
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
        Returns: (representative list, follower list)
        """
        if not agents:
            return [], []

        CRITICAL_SURVIVAL_COUNT = 6

        if len(agents) <= CRITICAL_SURVIVAL_COUNT:
            return agents, []  # All are representatives, no followers

        sample_size = max(1, int(len(agents) * ratio))

        sorted_agents = sorted(agents, key=lambda x: x.influence, reverse=True)

        top_k = max(1, int(sample_size * 0.5))
        representatives = sorted_agents[:top_k]

        candidates = sorted_agents[top_k:]
        if candidates:
            remaining_slots = sample_size - len(representatives)
            if remaining_slots > 0:
                representatives.extend(random.sample(candidates, remaining_slots))

        rep_ids = {p.agent_id for p in representatives}
        followers = [p for p in agents if p.agent_id not in rep_ids]

        return representatives, followers

    @staticmethod
    def get_group_stats_prompt(group_name: str, agents: List[Persona]) -> str:
        """Keep as is"""
        count = len(agents)
        if count == 0: return ""
        avg_satisfaction = np.mean([p.satisfaction[-1] if p.satisfaction else 0 for p in agents])
        post_wish_rate = np.mean([1 if p.post_wish else 0 for p in agents])
        return f"""
        [Group Name]: {group_name}
        [Group Size]: {count} people
        [Average Satisfaction]: {avg_satisfaction:.2f}
        [Current Posting Wish Rate]: {post_wish_rate:.1%}
        [Typical Persona Characteristics]: {agents[0].description[:100]}...
        """
