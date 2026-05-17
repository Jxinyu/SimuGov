import numpy as np
import random
from typing import List, Dict, Tuple

from config import settings
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

                          
                                           
                                     
        CRITICAL_SURVIVAL_COUNT = settings.platform.critical_survival_count
        sampling_mode = getattr(settings.platform, "representative_sampling_mode", "mixed").lower()
        if sampling_mode not in {"mixed", "random", "influence"}:
            sampling_mode = "mixed"

        if len(agents) <= CRITICAL_SURVIVAL_COUNT:
                                                                
            return agents, []             

                        
                     
        sample_size = max(1, int(len(agents) * ratio))

                                            
        sorted_agents = sorted(agents, key=lambda x: x.influence, reverse=True)

        if sampling_mode == "random":
            representatives = random.sample(sorted_agents, min(sample_size, len(sorted_agents)))
        elif sampling_mode == "influence":
            representatives = sorted_agents[:sample_size]
        else:
                                 
            top_k = max(1, int(sample_size * 0.5))
            representatives = sorted_agents[:top_k]

                       
            candidates = sorted_agents[top_k:]
            if candidates:
                remaining_slots = sample_size - len(representatives)
                if remaining_slots > 0:
                    representatives.extend(random.sample(candidates, min(remaining_slots, len(candidates))))

                             
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
