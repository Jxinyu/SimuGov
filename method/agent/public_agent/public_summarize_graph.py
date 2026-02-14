import logging
import asyncio
import operator
from typing import TypedDict, Sequence, List, Annotated

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

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
    tools_call_str: Annotated[List[str], operator.add]
    is_finalizing: bool


def create_agent_summarize_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    Creates a LangGraph executable graph with "final check" logic.
    """
    llm = get_async_llm(settings.model.public_summarize_model)
    llm_with_tools = llm.bind_tools(tools)

    def after_agent_router(state: AgentState):
        """
        After the model thinks in the main loop, decide whether to continue acting,
        enter final check, or end.
        """
        last_message = state['messages'][-1]
        tools_names_str = state.get('tools_call_str', [])

        if len(tools_names_str) > 3 and (len(set(tools_names_str[-3:])) == 1 or len(tools_names_str) >= 8):
            log.info("Loop detected or length limit reached, entering final check process...")
            return "summarize"

        if last_message.tool_calls:
            return "action"

        return "end"

    def after_summarize_router(state: AgentState):
        """
        Routing after the 'summarize' node.
        If the summarize node requests a tool call, go to action; otherwise end.
        """
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            log.info("Tool call required after final check, routing to action.")
            return "action"
        else:
            log.info("Final check completed, no additional operations required, routing to END.")
            return "end"

    def after_tool_router(state: AgentState):
        """
        Routing after a tool call.
        This is key to controlling the loop.
        """
        if state.get("is_finalizing"):
            log.info("Final mandatory tool has been executed, process finished.")
            return "end"

        return "agent"

    async def call_model(state: AgentState):
        """Model thinking node in the main loop."""
        log.info(f"🤔🤔🤔  {persona.agent_id} calling model to think...➡️➡️➡️  {persona.agent_id} model input: {state['messages']}")
        async with environment.llm_concurrent_nums_semaphore:
            response = await llm_with_tools.ainvoke(state['messages'])
        token_logger.record(response.usage_metadata)
        log.info(f"🔚🔚🔚  {persona.agent_id}  model thinking returned {response} ...")

        if response.content:
            thought_text = f"[Chain of Thought/CoT] {response.content}"

            save_thought_task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id,
                content=thought_text,
                day_time=environment.day_time,
            )
            environment.add_background_task(save_thought_task)

        return {"messages": list(state['messages']) + [response]}

    async def call_model_summarize(state: AgentState):
        """
        Final check node.
        Checks if 'update_persona_data' was called; if not, forces the call.
        """
        log.info(f"🤔🤔🤔  {persona.agent_id} entering summarize node for final check...")
        tools_names_str = state.get('tools_call_str', [])

        if "update_persona_data" in tools_names_str:
            log.info("'update_persona_data' has been called, process can safely end.")
            return state

        log.warning("'update_persona_data' was never called! Forcing final operation.")

        messages = list(state['messages'])

        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            log.info(f"Cleaning up unexecuted tool call requests from the previous round: {messages[-1].tool_calls}")
            messages = messages[:-1]

        force_prompt = HumanMessage(
            content="""
            IMPORTANT: Your task is not yet finished. You must call the `update_persona_data` tool to update personal data.
            This is the final and mandatory step of your task. Please call the tool immediately.
            """
        )
        messages_for_force_call = messages + [force_prompt]

        response = await llm_with_tools.ainvoke(messages_for_force_call)
        token_logger.record(response.usage_metadata)
        log.info(f"🔚🔚🔚  {persona.agent_id}  model returned after forced call: {response} ...")

        if response.content:
            thought_text = f"[Chain of Thought/CoT] {response.content}"

            save_thought_task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id,
                content=thought_text,
                day_time=environment.day_time,
            )
            environment.add_background_task(save_thought_task)

        return {
            "messages": messages_for_force_call + [response],
            "is_finalizing": True
        }

    async def call_tool(state: AgentState):
        """Tool execution node (works for both main loop and final calls)."""
        log.info(f"🤔🤔🤔  {persona.agent_id} calling tool...")
        new_messages = list(state['messages'])
        last_message = state['messages'][-1]
        tool_map = {t.name: t for t in tools}
        tasks, tools_names = [], []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in tool_map:
                tools_names.append(tool_name)
                tasks.append(tool_map[tool_name].ainvoke(tool_call['args']))
            else:
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"Error: Tool '{tool_name}' not found."))

        tool_responses = await asyncio.gather(*tasks)
        tool_messages = []
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_messages.append(
                ToolMessage(content=str(response), tool_call_id=tool_call['id'], name=tool_call['name'])
            )
        log.info(f"🔚🔚🔚  {persona.agent_id} tool call response: {tool_responses}")
        new_messages.extend(tool_messages)
        return {
            "messages": new_messages,
            "tools_call_str": tools_names
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.add_node("summarize", call_model_summarize)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        after_agent_router,
        {
            "action": "action",
            "summarize": "summarize",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "summarize",
        after_summarize_router,
        {
            "action": "action",
            "end": END,
        }
    )
    workflow.add_conditional_edges(
        "action",
        after_tool_router,
        {
            "agent": "agent",  # Return to main loop
            "end": END,  # Force end
        }
    )

    graph = workflow.compile()

    # Get drawable graph
    graph_viz = graph.get_graph()

    # Generate Mermaid syntax string
    mermaid_text = graph_viz.draw_mermaid()

    with open("public_summarize_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("public_summarize_graph has been successfully compiled")

    return graph
