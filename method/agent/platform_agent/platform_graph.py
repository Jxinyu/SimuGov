import asyncio
import logging
from typing import TypedDict, Sequence, List

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from method.utils.get_llm import get_async_llm
from config import settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    定义代理状态的结构。
    """
    messages: Sequence[BaseMessage]


def create_agent_graph(tools: List[BaseTool]):
    """
    创建并返回 LangGraph 的可执行图。
    这个图现在是通用的，它可以与任何符合规范的工具集一起工作。

    Args:
        tools: 一个包含要给智能体使用的 BaseTool 对象的列表。
    """
    llm = get_async_llm(settings.model.platform_model)
    llm_with_tools = llm.bind_tools(tools)

    def after_agent_router(state: AgentState):
        """
        模型思考后，决定是去行动还是结束。
        """
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "action"        
                       
        return "end"

    async def call_model(state: AgentState):
        new_messages = list(state['messages'])
        response = await llm_with_tools.ainvoke(state['messages'])
        log.info(f"模型返回：{response}")
        new_messages.append(response)
        return {
            "messages": new_messages
        }

    async def call_tool(state: AgentState):
        new_messages = list(state['messages'])          

        last_message = state['messages'][-1]            

        tool_map = {t.name: t for t in tools}                  

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            log.info(f"准备并行执行工具: {tool_name}(args={tool_call['args']})")
            if tool_name in tool_map:
                tool_to_call = tool_map[tool_name]
                                      
                task = tool_to_call.ainvoke(tool_call['args'])
                tasks.append(task)
            else:
                                      
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"错误：工具 '{tool_name}' 不存在。"))

                                            
        tool_responses = await asyncio.gather(*tasks)

                        
        tool_messages = []
                                
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_name = tool_call['name']
            log.info(f"工具 {tool_name} 返回：{response}")
            tool_messages.append(ToolMessage(content=str(response), tool_call_id=tool_call['id'], name=tool_name))

        new_messages.extend(tool_messages)
        return {
            "messages": new_messages,
        }

                         
    workflow = StateGraph(AgentState)

               
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

              
    workflow.set_entry_point("agent")

             
    workflow.add_conditional_edges(
        "agent",        
        after_agent_router,        
        {
            "action": "action",
            "end": END,                  
        },
    )

                 
    workflow.add_edge("action", "agent")

    graph = workflow.compile()

              
    graph_viz = graph.get_graph()

                       
    mermaid_text = graph_viz.draw_mermaid()

    with open("platform_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("Agent Graph 已成功编译")
    return graph
