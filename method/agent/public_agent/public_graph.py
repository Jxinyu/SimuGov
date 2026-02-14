import logging
import asyncio
from typing import TypedDict, Annotated, Sequence, List, Set

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
import operator

from method.agent.persona import Persona
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from method.utils.token_statistics import token_logger
from config import settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Defines the structure of the agent state.
    """
    messages: Sequence[BaseMessage]
    content_already_reacted_ids: Annotated[Set[str], operator.ior]
    content_already_read_ids: Annotated[Set[str], operator.ior]
    step_count: Annotated[int, operator.add]


def create_agent_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    Creates and returns the executable graph for LangGraph.
    This graph is generic and works with any toolset that conforms to the specification.

    Args:
        tools: A list of BaseTool objects to be used by the agent.
        :param persona:
        :param environment:
    """
    llm = get_async_llm(settings.model.public_scan_model)
    llm_with_tools = llm.bind_tools(tools)

    def after_agent_router(state: AgentState):
        """
        After the model thinks, decides whether to act or end.
        """
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "action"  # Call tool
        # End if the model did not call tools
        return "end"

    def after_tool_router(state: AgentState):
        """
        After the tool executes, decides whether to continue the loop or summarize.
        """
        reacted_ids = state.get("content_already_reacted_ids", set())
        current_step = state.get("step_count", 0)
        if len(reacted_ids) >= settings.public_agent.number_of_interactions:
            return "end"
        if current_step > 12:
            return "end"
        if len(state['messages']) > settings.public_agent.number_of_compressions:
            return "memory_compression"
        return "agent"

    async def call_model(state: AgentState):
        new_messages = list(state['messages'])
        log.info(f"🤔🤔🤔 {persona.agent_id} calling model to think... ➡️➡️➡️ {persona.agent_id} model input")
        async with environment.llm_concurrent_nums_semaphore:
            response = await llm_with_tools.ainvoke(new_messages)

        token_logger.record(response.usage_metadata)
        log.info(f"🔚🔚🔚 {persona.agent_id} model thinking returned {response} ...")

        if response.content:
            thought_text = f"[Chain of Thought/CoT] {response.content}"

            save_thought_task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id,
                content=thought_text,
                day_time=environment.day_time,
            )
            environment.add_background_task(save_thought_task)

        new_messages.append(response)
        return {
            "messages": new_messages,
            "step_count": 1
        }

    async def call_tool(state: AgentState):
        log.info(f"🤔🤔🤔 {persona.agent_id} calling tool...")

        new_messages = list(state['messages'])
        last_message = state['messages'][-1]
        tool_map = {t.name: t for t in tools}

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in tool_map:
                tool_to_call = tool_map[tool_name]
                tasks.append(tool_to_call.ainvoke(tool_call['args']))
            else:
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"Error: Tool '{tool_name}' does not exist."))

        try:
            tool_responses = await asyncio.gather(*tasks)
        except Exception as e:
            log.error(f"Error occurred during parallel tool execution: {e}")
            tool_responses = [f"Tool execution failed: {e}"] * len(last_message.tool_calls)

        tool_messages = []
        newly_reacted_ids = set()
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_name = tool_call['name']
            if tool_name == "react_to_content":
                content_id = tool_call['args'].get("content_id")
                if content_id:
                    newly_reacted_ids.add(content_id)

            tool_messages.append(
                ToolMessage(
                    content=str(response),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
            )

        new_messages.extend(tool_messages)
        log.info(f"🔚🔚🔚 {persona.agent_id} tool response: {tool_responses}")
        return {
            "messages": new_messages,
            "content_already_reacted_ids": newly_reacted_ids,
            "step_count": 1
        }

    async def memory_compression(state: AgentState):
        """
        Intelligently compresses message history to save tokens.
        """
        """
        Intelligently compresses message history by only summarizing the oldest parts while keeping the latest working memory.
        """
        log.info(f"✂️ {persona.agent_id} memory count reached compression threshold, starting intelligent pruning...")

        messages = state["messages"]

        system_message = messages[0]

        NUM_TO_KEEP = settings.public_agent.number_of_keep

        messages_to_prune = messages[1:-NUM_TO_KEEP]
        messages_to_keep = messages[-NUM_TO_KEEP:]

        if isinstance(messages_to_keep[-1], AIMessage) and messages_to_keep[-1].tool_calls:
            messages_to_prune = messages[:- (NUM_TO_KEEP + 1)]
            messages_to_keep = messages[- (NUM_TO_KEEP + 1):]

        summarization_prompt = [
            HumanMessage(
                content="""
                # Role: Memory Compression Expert
Your task is to assist an AI agent in organizing its short-term memory. It is currently executing the task "Browsing and interacting with social platforms".
Due to the limited context window, you need to compress its **past thoughts and operation history** into a concise summary so it can continue working.

# Compression Principles (Strictly Follow)
1. **Data Retention (Highest Priority)**: You must explicitly retain all **Content IDs**, **Topics**, and **Platform Labels** that appeared. The agent must not forget which IDs it has seen, otherwise, it will browse repeatedly.
2. **State Retention**: You must retain the agent's **attitude** towards these contents (Like/Dislike/Neutral) and the **actions performed** (Liked/Commented/Skipped).
3. **Intent Retention**: If the agent expressed an intent in the previous step (e.g., "I intend to comment on content X") but has not yet executed it, it must be recorded.
4. **Denoising**: Remove specific JSON format details, duplicate system prompts, and meaningless pleasantries.

# Output Format
Output a **first-person** narrative text. For example: "I browsed content [ID: xxx], the topic was science fiction, I disliked it because... Then I browsed [ID: yyy], and I plan to comment on it."
                """),
            HumanMessage(content="Here is the conversation history that needs to be compressed:"),
            *messages_to_prune,
            HumanMessage(content="Please generate the compressed summary based on the principles above:"),
        ]

        summarizer_llm = get_async_llm(settings.model.dialogue_history_model)
        summary_response = await summarizer_llm.ainvoke(summarization_prompt)

        summary_memory = summary_response.content
        log.info(f"    -> Compressed {len(messages_to_prune)} messages, summary is: {summary_memory}")

        new_messages = [system_message,
                        HumanMessage(content=f"[Memory compressed, the resulting summary is as follows] {summary_memory}")] + messages_to_keep

        return {"messages": new_messages}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.add_node("memory_compression", memory_compression)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",  # Starting node
        after_agent_router,  # Decision function
        {
            "action": "action",
            "end": END,  # If "end" is returned, terminate
        },
    )
    workflow.add_conditional_edges(
        "action",  # Starting node
        after_tool_router,  # Decision function
        {
            "agent": "agent",
            "memory_compression": "memory_compression",
            "end": END,
        },
    )
    workflow.add_edge("memory_compression", "agent")

    graph = workflow.compile()

    graph_viz = graph.get_graph()

    mermaid_text = graph_viz.draw_mermaid()

    with open("public_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("Public Agent Graph has been successfully compiled")
    return graph
