import asyncio
import logging
from typing import TypedDict, Sequence, List

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from method.utils.get_llm import get_async_llm
from method.utils.token_statistics import token_logger
from config import settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Defines the structure of the agent state.
    """
    messages: Sequence[BaseMessage]


def create_agent_graph(tools: List[BaseTool]):
    """
    Creates and returns the executable graph for LangGraph.
    This graph is now generic and can work with any toolset that conforms to the specification.

    Args:
        tools: A list of BaseTool objects to be used by the agent.
    """
    llm = get_async_llm(settings.model.platform_model)
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

    async def call_model(state: AgentState):
        new_messages = list(state['messages'])
        response = await llm_with_tools.ainvoke(state['messages'])
        token_logger.record(response.usage_metadata)
        log.info(f"Model returned: {response}")
        new_messages.append(response)
        return {
            "messages": new_messages
        }

    async def call_tool(state: AgentState):
        new_messages = list(state['messages'])  # Get message list

        last_message = state['messages'][-1]  # Get the last message

        tool_map = {t.name: t for t in tools}  # Create a lookup map from the passed tool list

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            log.info(f"Preparing to execute tool in parallel: {tool_name}(args={tool_call['args']})")
            if tool_name in tool_map:
                tool_to_call = tool_map[tool_name]
                # Create task and add to the list, but do not await
                task = tool_to_call.ainvoke(tool_call['args'])
                tasks.append(task)
            else:
                # Handle hallucinated tools by creating a coroutine that returns an error immediately
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"Error: Tool '{tool_name}' does not exist."))

        # 2. Use asyncio.gather to execute all tasks simultaneously and in parallel
        tool_responses = await asyncio.gather(*tasks)

        # 3. Collect the results of all parallel tasks
        tool_messages = []
        # Map the results back to the original tool_calls
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_name = tool_call['name']
            log.info(f"Tool {tool_name} returned: {response}")
            tool_messages.append(ToolMessage(content=str(response), tool_call_id=tool_call['id'], name=tool_name))

        new_messages.extend(tool_messages)
        return {
            "messages": new_messages,
        }

    # 1. Create StateGraph instance
    workflow = StateGraph(AgentState)

    # 2. Add all nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

    # 3. Set entry point
    workflow.set_entry_point("agent")

    # 4. Conditional routing
    workflow.add_conditional_edges(
        "agent",  # Start node
        after_agent_router,  # Decision function
        {
            "action": "action",
            "end": END,  # If "end" is returned, terminate
        },
    )

    # 5. Add other fixed edges
    workflow.add_edge("action", "agent")

    graph = workflow.compile()

    # Get the drawable graph
    graph_viz = graph.get_graph()

    # Generate Mermaid syntax string
    mermaid_text = graph_viz.draw_mermaid()

    with open("platform_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("Agent Graph has been successfully compiled")
    return graph
