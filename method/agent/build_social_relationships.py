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
    agent_id: str = Field(description="Agent ID")
    strength: float = Field(description="Strength of the social relationship")


class ReturnFormat(BaseModel):
    social_relationships: Optional[List[Relationship]] = Field(
        default=[],
        description="List of social relationships")


async def construct_prompt():
    return """
# Role and Background
You are participating in a simulation of a virtual community named "ArtStation". You have just entered this community and need to establish your initial social circle.
Your task is: Browse the list of other users in the community and decide whom you want to follow based on your **personality, standpoint, and potential motivations**.

# Your Personal Persona
{self_persona}

# Candidate User List
{candidate_list}

# Decision Logic
Please carefully analyze the relationship between you and the candidates. Your following behavior can be based on one of the following two motivations:

1.  **Resonance and Support (Positive Strength: 0.1 to 1.0)**
    *   **Like attracts like**: If the other party's standpoint (`standpoint`) or beliefs (`beliefs`) are highly consistent with yours.
    *   **Appreciation**: If the other party's description (`description`) makes you feel they are worth learning from or respecting.
    *   *Example*: A "Compliance Creator" might follow another "Tech Guru" with a strength of 0.8.

2.  **Observation and Opposition (Negative Strength: -0.1 to -1.0)**
    *   **Hate-watching**: If you are a radical, you might follow your "enemies" to find attack material or objects of ridicule.
    *   **Dislike**: If the other party's behavior happens to hit your trigger points (e.g., an "Originality Defender" following a "cocky Watermark Breaker"), you might establish a negative connection.
    *   *Example*: A "Watermark Breaker" might follow a "Platform Loyal Fan" with a strength of -0.9, meaning "I'm watching you."

# Constraints
1.  **Exercise Restraint**: You don't need to follow everyone! Please only choose **3 to 7** people who trigger your emotions (whether positive or negative) the most.
2.  **No self-following**: Ignore your own ID in the list.
3.  **Consistent with Persona**:
    *   If you are a person with **High Beta (Rebellion psychology)**, you are more inclined to follow those who challenge rules (positive) or those who maintain rules (negative).
    *   If you are a person with **High Gamma (Information cocoon)**, please only follow those whose views are completely consistent with yours.
    *   If you are a **Public member (Bystander)**, you can follow people with high influence (`influence`), regardless of their standpoint.

# Output Format

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
