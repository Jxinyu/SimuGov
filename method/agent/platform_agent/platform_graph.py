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
            return "action"  # 调用工具
        # 如果模型没调用工具, 结束
        return "end"

    async def call_model(state: AgentState):
        new_messages = list(state['messages'])
        response = await llm_with_tools.ainvoke(state['messages'])
        token_logger.record(response.usage_metadata)
        log.info(f"模型返回：{response}")
        new_messages.append(response)
        return {
            "messages": new_messages
        }

    async def call_tool(state: AgentState):
        new_messages = list(state['messages'])  # 获取消息列表

        last_message = state['messages'][-1]  # 获取最后一条消息

        tool_map = {t.name: t for t in tools}  # 从传入的工具列表创建查找映射

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            log.info(f"准备并行执行工具: {tool_name}(args={tool_call['args']})")
            if tool_name in tool_map:
                tool_to_call = tool_map[tool_name]
                # 创建任务并添加到列表，但不要 await
                task = tool_to_call.ainvoke(tool_call['args'])
                tasks.append(task)
            else:
                # 处理幻觉工具，创建一个立即返回错误的协程
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"错误：工具 '{tool_name}' 不存在。"))

        # 2. 使用 asyncio.gather 一次性、并行地执行所有任务
        tool_responses = await asyncio.gather(*tasks)

        # 3. 收集所有并行任务的结果
        tool_messages = []
        # 将结果与原始的 tool_call 对应起来
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_name = tool_call['name']
            log.info(f"工具 {tool_name} 返回：{response}")
            tool_messages.append(ToolMessage(content=str(response), tool_call_id=tool_call['id'], name=tool_name))

        new_messages.extend(tool_messages)
        return {
            "messages": new_messages,
        }

    # 1. 创建 StateGraph 实例
    workflow = StateGraph(AgentState)

    # 2. 添加所有节点
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

    # 3. 设置入口点
    workflow.set_entry_point("agent")

    # 4. 条件路由
    workflow.add_conditional_edges(
        "agent",  # 起始节点
        after_agent_router,  # 判断函数
        {
            "action": "action",
            "end": END,  # 如果返回 "end"，则结束
        },
    )

    # 5. 添加其他固定的边
    workflow.add_edge("action", "agent")

    graph = workflow.compile()

    # 获取可绘制的图谱
    graph_viz = graph.get_graph()

    # 生成 Mermaid 语法的字符串
    mermaid_text = graph_viz.draw_mermaid()

    with open("platform_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("Agent Graph 已成功编译")
    return graph
