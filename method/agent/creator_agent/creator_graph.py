import logging
from typing import TypedDict, Sequence, List, Annotated
import operator
import asyncio

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from openai import BadRequestError

from method.agent.persona import Persona
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from method.utils.token_statistics import token_logger
from config import settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Defines the structure of the agent state.
    Note: We use "Full Coverage" mode for messages here to facilitate the implementation of memory compression nodes.
    """
    messages: Sequence[BaseMessage]
    step_count: Annotated[int, operator.add]


def create_agent_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    Creates and returns the executable graph for LangGraph.
    """
    llm = get_async_llm(settings.model.creator_model)
    llm_with_tools = llm.bind_tools(tools)
    summarizer_llm = get_async_llm(settings.model.dialogue_history_model)

    def get_safe_split_index(messages: List[BaseMessage], keep_last_n: int) -> int:
        total = len(messages)
        if total <= keep_last_n + 1:
            return 1

        split_idx = total - keep_last_n
        if split_idx < 1:
            split_idx = 1

        while split_idx > 1 and isinstance(messages[split_idx], ToolMessage):
            split_idx -= 1

        return split_idx

    def after_agent_router(state: AgentState):
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "action"
        return "end"

    def after_tool_router(state: AgentState):
        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage) and last_message.name == 'push_content':
            return "end"

        if len(state['messages']) > settings.public_agent.number_of_compressions:
            return "memory_compression"

        return "agent"

    async def call_model(state: AgentState):
        current_step = state.get("step_count", 0)
        messages = list(state['messages'])  # Get current full history

        is_force_step = False

        if current_step >= 8:
            log.warning(f"🚨 {persona.agent_id} step count reached {current_step}, triggering forced post!")
            is_force_step = True
            force_prompt = HumanMessage(content="""
            【System Instruction: Time Expired】
            Please **stop thinking immediately**. Based on the current information, immediately call the `push_content` tool to publish content.
            Do not call query tools anymore. You must publish.
            """)
            messages.append(force_prompt)

        log.info(f"🤔 {persona.agent_id} thinking... (Step: {current_step}, HistLen: {len(messages)})")

        max_retries = 2
        attempt = 1
        response = None

        while attempt <= max_retries:
            try:
                async with environment.llm_concurrent_nums_semaphore:
                    # Send full history
                    response = await llm_with_tools.ainvoke(messages)
                break
            except BadRequestError as e:
                log.warning(f"⚠️ API Error (Attempt {attempt}): {e}")
                if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
                    if attempt >= max_retries:
                        response = AIMessage(content="(System: Content intercepted, operation terminated.)")
                        break
                    messages.append(SystemMessage(content="【Warning】Sensitive words detected. Please retry using objective, academic language."))
                    attempt += 1
                else:
                    if attempt >= max_retries:
                        response = AIMessage(content="(System error: Skipping.)")
                        break
                    attempt += 1
            except Exception as e:
                log.error(f"❌ Unknown error: {e}")
                response = AIMessage(content="(System error: Skipping.)")
                break

        token_logger.record(response.usage_metadata)

        if response.content:
            thought_text = f"【Chain of Thought/CoT】{'(Forced)' if is_force_step else ''} {response.content}"
            task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id, content=thought_text, day_time=environment.day_time
            )
            environment.add_background_task(task)
        return {
            "messages": messages + [response],
            "step_count": 1
        }

    async def call_tool(state: AgentState):
        log.info(f"🛠️ {persona.agent_id} calling tools...")

        last_message = state['messages'][-1]
        tool_map = {t.name: t for t in tools}
        tasks = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in tool_map:
                tasks.append(tool_map[tool_name].ainvoke(tool_call['args']))
            else:
                async def _err(msg):
                    return msg

                tasks.append(_err(f"Error: Tool '{tool_name}' not found."))

        tool_responses = await asyncio.gather(*tasks)

        tool_messages = []
        for res, call in zip(tool_responses, last_message.tool_calls):
            tool_messages.append(ToolMessage(content=str(res), tool_call_id=call['id'], name=call['name']))

        return {
            "messages": list(state['messages']) + tool_messages,
            "step_count": 1
        }

    async def memory_compression(state: AgentState):
        log.info(f"✂️ {persona.agent_id} triggering memory compression...")
        messages = state["messages"]
        system_msg = messages[0]

        NUM_TO_KEEP = 2

        split_idx = get_safe_split_index(messages, keep_last_n=NUM_TO_KEEP)

        messages_to_prune = messages[1:split_idx]
        messages_to_keep = messages[split_idx:]

        if not messages_to_prune:
            return {"messages": messages}

        prompt = [
            HumanMessage(
                content="You are a memory compression assistant. Please summarize the following creator's thoughts and operation history, retaining key parameters (such as attack technical indicators) and current intentions."),
            HumanMessage(content="History to be compressed:"),
            *messages_to_prune,
            HumanMessage(content="Generate summary:"),
        ]

        async with environment.llm_concurrent_nums_semaphore:
            summary_res = await summarizer_llm.ainvoke(prompt)

        summary = summary_res.content
        log.info(f"    -> Compression complete.")

        new_messages = [
                           system_msg,
                           HumanMessage(content=f"【Historical Summary】{summary}")
                       ] + messages_to_keep

        return {"messages": new_messages}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.add_node("memory_compression", memory_compression)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", after_agent_router, {"action": "action", "end": END}
    )
    workflow.add_conditional_edges(
        "action", after_tool_router,
        {"agent": "agent", "memory_compression": "memory_compression", "end": END}
    )
    workflow.add_edge("memory_compression", "agent")

    return workflow.compile()
