from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Literal, Optional, Dict, Any
from config import settings


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
    beta: str = Field(...,
                      description="逆反心理参数：高中低。 高：高度叛逆，天然不信任权威，倾向于将平台的行为解读为负面动机，是潜在的抗议者。；中：有独立思考能力，会对信息进行批判性审视，但态度相对中立和务实。；低：基本顺从，倾向于信任权威和官方解释，认为规则是必要的。")
    gamma: str = Field(...,
                       description="确认偏误系数：高中低。 高：信息茧房的重度用户。你极度倾向于寻找、相信并传播符合自己既有立场的信息，并会主动排斥、无视甚至攻击相反的观点。你的信念很难被动摇。；中：有自己的观点和倾向。你更喜欢阅读符合自己立场的信息，但面对强有力的相反证据时，仍然会进行思考和摇摆，具备被说服的可能性。；低：理性客观的观察者。你对所有信息的接受度几乎只取决于信息本身，能公正地评估与自己既有立场相反的证据，不易陷入信息茧房。")
    fp_sensitivity: Optional[str] = Field(...,
                                          description="误伤敏感度，艺术家非常高：高中低。高：极度敏感，将任何一次误伤都视为对其专业性的巨大侮辱和平台的背叛，会引发极其强烈的负面反应。；中：务实且在意声誉，一次误伤会使其感到困扰和不满，并会影响其后续的创作热情。；低：心态开放，认为偶尔的技术错误可以理解，不会因此产生强烈负面情绪。")
    cost_sensitivity: Optional[str] = Field(...,
                                            description="（水印破坏者专属） 决定了其对攻击技术的使用成本（计算资源、时间等）的敏感度: 高中低。 高：精打细算，极其看重成本效益，总是倾向于选择成本最低的攻击方案，即使其成功率并非最高。；中：务实平衡，会在攻击成本和预期成功率之间寻找一个平衡点，追求“性价比”。；低：不计成本，为了确保规避成功，愿意使用最高成本、最复杂的攻击技术。")

    influence: float = Field(..., description="智能体影响力: 0 ~ 1")
    satisfaction: List[float] = Field(..., description="智能体对平台的满意度: -1 ~ 1")

    post_wish: Optional[bool] = Field(..., description="发布意愿: true/false")
    is_active: bool = Field(..., description="当前是否仍在平台: true/false")

    beliefs: List[str] = Field(..., description="智能体的信念: 列表")
    social_relationships: Dict[str, float] = Field(...,
                                                   description="社交关系集合:{agent_id: 关注度}")

    viewed_content: Optional[List[str]] = Field(default=[], description="已经查看过的内容列表")
    reacted_content: Optional[List[str]] = Field(default=[], description="已经对内容进行反应的列表")

    # 1. 新增一个私有属性，作为“缓冲区”
    # 使用 PrivateAttr 避免 Pydantic 把它当作模型字段去校验
    _pending_state: Dict[str, Any] = PrivateAttr(default_factory=dict)

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
                    - 逆反心理参数: {self.beta}
                    - 确认偏误系数: {self.gamma}
                    """
            if self.type == '合规创作者':
                special_trait += f"- 我的误伤敏感度: {self.fp_sensitivity}\n"
            elif self.type == '水印破坏者':
                special_trait += f"- 我的攻击成本敏感度: {self.cost_sensitivity}\n"
        state_prompt = f"""
            影响力: {self.influence}
            对平台的满意度变化（近一周）: {self.satisfaction[-7:]}
            发布意愿: {self.post_wish}
            当前是否仍在平台: {self.is_active}
            我的信念: {self.beliefs}
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
                            satisfaction: float | None,
                            post_wish: bool | None, is_active: bool | None,
                            beliefs: List[str] | None) -> bool:
        """
       修改后的逻辑：
       1. 情感(satisfaction)和认知(beliefs)：立即更新。
          原因：KPI计算需要反映“今天”反思后的最新情绪。
       2. 身份(type)、活跃度(is_active)、意愿(post_wish)：暂存到缓冲区。
          原因：KPI计算需要基于“今天白天”的身份和在场状态进行统计，避免幸存者偏差。
       """

        # --- 立即更新部分 (数据层面) ---
        if satisfaction is not None:
            self.satisfaction.append(satisfaction)

        if beliefs is not None:
            self.beliefs = beliefs

        # --- 暂存/缓冲部分 (结构层面) ---
        # 将决定存入缓冲区，而不是直接修改 self.type 或 self.is_active
        updates = {}
        if persona_role_positioning is not None and persona_role_positioning != self.type:
            updates['type'] = persona_role_positioning

        if post_wish is not None:
            updates['post_wish'] = post_wish

        if is_active is not None:
            updates['is_active'] = is_active

        # 更新缓冲区（覆盖旧的暂存值）
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
        # 清空缓冲区
        self._pending_state.clear()
