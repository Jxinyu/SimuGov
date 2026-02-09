import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from method.utils.get_llm import get_async_llm
from method.environment import Environment
from method.agent.persona import Persona
from pydantic import BaseModel, Field
from typing import List, Optional


async def build_relationships(environment: Environment):
    candidate_list = []
    for persona in environment.personas.values():
        candidate_list.append(
            persona.model_dump(exclude={"post_wish", "is_active", "social_relationships", "cost_sensitivity"}))
    task = []
    for persona in environment.personas.values():
        task.append(public_process(environment, persona, candidate_list))

    await asyncio.gather(*task)


class Relationship(BaseModel):
    agent_id: str = Field(description="代理ID")
    strength: float = Field(description="社交关系强度")


class ReturnFormat(BaseModel):
    social_relationships: Optional[List[Relationship]] = Field(
        default=[],
        description="社交关系列表")


async def construct_prompt():
    return """
# 角色与背景
你正在参与一个名为 "ArtStation" 的虚拟社区仿真。你刚刚进入这个社区，需要建立你的初步社交圈。
你的任务是：浏览社区中的其他用户列表，并根据你的**性格、立场和潜在动机**，决定你要关注谁。

# 你的个人画像
{self_persona}

# 候选用户名单
{candidate_list}

# 决策逻辑
请仔细分析你与候选人之间的关系，你的关注行为可以基于以下两种动机之一：

1.  **共鸣与支持 (Positive Strength: 0.1 to 1.0)**
    *   **同类相吸**：如果对方的立场 (`standpoint`)、信念 (`beliefs`) 与你高度一致。
    *   **欣赏**：如果对方的描述 (`description`) 让你觉得值得学习或尊重。
    *   *示例*：一个“合规创作者”可能会以 0.8 的强度关注另一个“技术大牛”。

2.  **审视与对立 (Negative Strength: -0.1 to -1.0)**
    *   **敌对监控 (Hate-watching)**：如果你是激进派，你可能会关注你的“敌人”，以便寻找攻击素材或嘲笑对象。
    *   **厌恶**：如果对方的行为刚好踩在你的雷点上（例如“原创捍卫者”关注了一个“嚣张的水印破坏者”），你可能会建立一个负向连接。
    *   *示例*：一个“水印破坏者”可能会以 -0.9 的强度关注一个“平台死忠粉”，意为“我盯着你呢”。

# 约束条件
1.  **保持克制**：你不需要关注所有人！请只选择 **3 到 7 个** 最能引发你情绪（无论正面还是负面）的人。
2.  **无需关注自己**：忽略列表中你自己的 ID。
3.  **符合人设**：
    *   如果你是 **Beta(逆反心理) 高** 的人，你更倾向于关注那些挑战规则的人（正向）或维护规则的人（负向）。
    *   如果你是 **Gamma(信息茧房) 高** 的人，请只关注那些和你观点完全一致的人。
    *   如果你是 **公众(吃瓜群众)**，你可以关注影响力(`influence`)高的人，无论立场如何。

# 输出格式

{format_instructions}
    """


async def public_process(environment: Environment, persona: Persona, candidate_list):
    llm = get_async_llm(settings.model.creator_model)
    output_parser = JsonOutputParser(pydantic_object=ReturnFormat)

    prompt = ChatPromptTemplate.from_template(
        template=await construct_prompt(),
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    agent = prompt | llm | output_parser

    response = await agent.ainvoke({
        "self_persona": persona.model_dump(
            exclude={"post_wish", "is_active", "social_relationships", "cost_sensitivity"}),
        "candidate_list": candidate_list
    })
    try:
        for relationships in response["social_relationships"]:
            if relationships['agent_id'] in environment.personas.keys():
                persona.social_relationships[relationships['agent_id']] = relationships['strength']
    except:
        pass
