from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Literal, Optional, Dict, Any, Union
from config import settings
from method.utils.psychological_parameter_mapping_table import get_psycho_text


class Persona(BaseModel):
    """
    公众与创作者实例
    """
    agent_id: str = Field(..., description="The agent ID")
    name: str = Field(..., description="The agent name")
    type: Literal['合规创作者', '水印破坏者', '公众']
    description: str = Field(..., description="The agent description")
    standpoint: List[float] = Field(...,
                                    description="人物立场，由概率元组构成（信任派、反抗派、中立派），例如（0.6，0.3，0.1）表示这个智能体更倾向于信任。")
    beta: Optional[Union[str, float]] = Field(...,
                                              description="逆反心理参数")
    gamma: Optional[Union[str, float]] = Field(...,
                                               description="确认偏误系数。")
    fp_sensitivity: Optional[Union[str, float]] = Field(...,
                                                        description="误伤敏感度。")
    cost_sensitivity: Optional[Union[str, float]] = Field(...,
                                                          description="愿不愿意花钱")
    attack_resource: Optional[Union[str, float]] = Field(...,
                                                         description="可用的攻击资源值")

    influence: float = Field(..., description="智能体影响力: 0 ~ 1")
    satisfaction: List[float] = Field(..., description="智能体对平台的满意度: -1 ~ 1")

    post_wish: Optional[bool] = Field(..., description="发布意愿: true/false")
    is_active: bool = Field(..., description="当前是否仍在平台: true/false")

    beliefs: List[str] = Field(..., description="智能体的信念: 列表")
    social_relationships: Dict[str, float] = Field(...,
                                                   description="社交关系集合:{agent_id: 关注度}")

    viewed_content: Optional[List[str]] = Field(default=[], description="已经查看过的内容列表")
    reacted_content: Optional[List[str]] = Field(default=[], description="已经对内容进行反应的列表")

                         
                                            
    _pending_state: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @staticmethod
    def _normalize_beliefs_for_llm(
            beliefs: Optional[List[str]],
            *,
            max_items: int = 4,
            max_chars: int = 50
    ) -> List[str]:
        if not beliefs:
            return []
        normalized: List[str] = []
        for item in beliefs:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            normalized.append(text[:max_chars])
            if len(normalized) >= max_items:
                break
        return normalized

    def get_public_prompt(self) -> str:
        """生成一个描述个人信息的提示字符串"""
        base_prompt = f"""
            ---
            关于我的个人信息如下：
            person_id: {self.agent_id}
            姓名: {self.name}
            类型: {self.type}
            """
        if settings.platform.ablation_validation:
            content_description = """
                        我的描述: 我是一个追求利益最大化的数字内容生产者。我没有艺术情怀，只在乎投入产出比。
                        """
            special_trait = """
                        【核心行为准则 - 绝对理性模式】：
                        1. 屏蔽情感：请完全忽略你作为“艺术家”可能产生的愤怒、尊严或被冒犯感。
                        2. 决策依据：你的决策仅基于“期望收益 = 流量 * 转化率 - 风险成本”。
                        3. 对抗议的态度：抗议是浪费时间且无收益的行为，除非系统明确奖励抗议，否则你绝不参与。
                        4. 对误伤的态度：误伤仅仅是 0.1% 的效率损耗，不值得产生情绪波动。
                        """
        else:
            content_description = f"政治立场: {self.standpoint} (顺序:[信任度, 反抗度, 中立度]) \n我的描述: {self.description}"
            special_trait = f"""
                    我的心理个性参数：
                    - 逆反心理参数: {get_psycho_text('beta', self.beta)}
                    - 确认偏误系数: {get_psycho_text('gamma', self.gamma)}
                    """
            if self.type == '合规创作者':
                special_trait += f"- 误伤敏感度: {get_psycho_text('fp_sensitivity', self.fp_sensitivity)}\n"
            elif self.type == '水印破坏者':
                special_trait += f"- 攻击成本敏感度: {get_psycho_text('cost_sensitivity', self.cost_sensitivity)}\n"
                special_trait += f"- 可用攻击资源值: {self.attack_resource}\n"
        state_prompt = f"""
            影响力: {self.influence}
            对平台的满意度变化（近一周）: {self.satisfaction[-7:]}
            发布意愿: {self.post_wish}
            当前是否仍在平台: {self.is_active}
            我的信念: {self._normalize_beliefs_for_llm(self.beliefs)}
            社交关系: {self.social_relationships}
            ---
            """
        return base_prompt + content_description + special_trait + state_prompt

    def verify_content_is_viewed(self, content_id: str) -> bool:
        """
        验证内容是否已经查看过。
        """
        return content_id in self.viewed_content

    def update_viewed_content(self, content_id: List[str]) -> bool:
        """
        更新已经查看过的内容列表。
        """
        self.viewed_content.extend(content_id)
        return True

    def verify_content_is_reacted(self, content_id: str) -> bool:
        """
        验证内容是否已经对内容进行反应。
        """
        return content_id in self.reacted_content

    def update_reacted_content(self, content_id: List[str]) -> bool:
        """
        更新已经对内容进行反应的列表。
        """
        self.reacted_content.extend(content_id)
        return True

    def update_persona_data(self, persona_role_positioning: Literal['合规创作者', '水印破坏者', '公众'],
                            satisfaction: Optional[float],
                            post_wish: Optional[bool], is_active: Optional[bool],
                            beliefs: Optional[List[str]]) -> bool:
        """
       修改后的逻辑：
       1. 情感(satisfaction)和认知(beliefs)：立即更新。
          原因：KPI计算需要反映“今天”反思后的最新情绪。
       2. 身份(type)、活跃度(is_active)、意愿(post_wish)：暂存到缓冲区。
          原因：KPI计算需要基于“今天白天”的身份和在场状态进行统计，避免幸存者偏差。
       """

                               
        if satisfaction is not None:
            self.satisfaction.append(satisfaction)

        if beliefs is not None:
            self.beliefs = self._normalize_beliefs_for_llm(beliefs)

                                
                                                     
        updates = {}
        if persona_role_positioning is not None and persona_role_positioning != self.type:
            updates['type'] = persona_role_positioning

        if post_wish is not None:
            updates['post_wish'] = post_wish

        if is_active is not None:
            updates['is_active'] = is_active

                        
        self._pending_state.update(updates)

        return True

    def commit_state(self):
        """
        【新增方法】提交状态。
        在每日 KPI 计算完成后调用，正式应用身份和活跃度的变更。
        """
        if not self._pending_state:
            return

        if 'type' in self._pending_state:
            self.type = self._pending_state['type']

        if 'post_wish' in self._pending_state:
            self.post_wish = self._pending_state['post_wish']

        if 'is_active' in self._pending_state:
            self.is_active = self._pending_state['is_active']
               
        self._pending_state.clear()
