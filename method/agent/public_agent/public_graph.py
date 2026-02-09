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
    定义代理状态的结构。
    """
    messages: Sequence[BaseMessage]
    content_already_reacted_ids: Annotated[Set[str], operator.ior]
    content_already_read_ids: Annotated[Set[str], operator.ior]
    step_count: Annotated[int, operator.add]


def create_agent_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    创建并返回 LangGraph 的可执行图。
    这个图现在是通用的，它可以与任何符合规范的工具集一起工作。

    Args:
        tools: 一个包含要给智能体使用的 BaseTool 对象的列表。
        :param persona:
        :param environment:
    """
    llm = get_async_llm(settings.model.public_scan_model)
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

    def after_tool_router(state: AgentState):
        """
        工具执行后，决定是继续循环还是去总结。
        """
        reacted_ids = state.get("content_already_reacted_ids", set())
        current_step = state.get("step_count", 0)
        if len(reacted_ids) >= settings.public_agent.number_of_interactions:  # 已达到3次互动上限，强制结束。
            return "end"
        if current_step > 12:
            return "end"
        if len(state['messages']) > settings.public_agent.number_of_compressions:
            return "memory_compression"
        return "agent"

    async def call_model(state: AgentState):
        new_messages = list(state['messages'])
        log.info(f"🤔🤔🤔  {persona.agent_id} 调用模型思考...➡️➡️➡️  {persona.agent_id} 模型输入")
        async with environment.llm_concurrent_nums_semaphore:
            response = await llm_with_tools.ainvoke(new_messages)

        token_logger.record(response.usage_metadata)
        log.info(f"🔚🔚🔚  {persona.agent_id}  模型思考返回 {response} ...")

        if response.content:
            thought_text = f"【思维链/CoT】{response.content}"

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
        log.info(f"🤔🤔🤔  {persona.agent_id} 调用工具...")

        new_messages = list(state['messages'])  # 获取消息列表
        last_message = state['messages'][-1]  # 获取最后一条消息
        tool_map = {t.name: t for t in tools}  # 从传入的工具列表创建查找映射

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in tool_map:
                tool_to_call = tool_map[tool_name]
                tasks.append(tool_to_call.ainvoke(tool_call['args']))
            else:
                async def _get_error(msg):
                    return msg

                tasks.append(_get_error(f"错误：工具 '{tool_name}' 不存在。"))

        try:
            tool_responses = await asyncio.gather(*tasks)
        except Exception as e:
            # 如果任何一个工具调用失败，记录错误并继续，而不是让整个智能体崩溃
            log.error(f"在并行执行工具时发生错误: {e}")
            # 返回一个错误消息作为工具响应
            tool_responses = [f"工具执行失败: {e}"] * len(last_message.tool_calls)

        # 收集结果
        tool_messages = []
        newly_reacted_ids = set()
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_name = tool_call['name']
            if tool_name == "react_to_content":
                content_id = tool_call['args'].get("content_id")
                if content_id:
                    newly_reacted_ids.add(content_id)

            # 正确地创建 ToolMessage 对象
            tool_messages.append(
                ToolMessage(
                    content=str(response),
                    tool_call_id=tool_call['id'],
                    name=tool_name
                )
            )

        new_messages.extend(tool_messages)
        log.info(f"🔚🔚🔚  {persona.agent_id} 调用工具 响应：{tool_responses}")
        return {
            "messages": new_messages,
            "content_already_reacted_ids": newly_reacted_ids,
            "step_count": 1
        }

    async def memory_compression(state: AgentState):
        """
        智能地压缩消息历史，以节省 token。
        """
        """
               智能地压缩消息历史，只总结最旧的部分，保留最新的工作记忆。
               """
        log.info(f"✂️ {persona.agent_id}  的记忆数量，达到压缩阈值，开始智能剪枝...")

        messages = state["messages"]

        system_message = messages[0]

        # 定义要保留的最新消息数量，这足以维持一个完整的ReAct循环
        NUM_TO_KEEP = settings.public_agent.number_of_keep

        messages_to_prune = messages[1:-NUM_TO_KEEP]
        messages_to_keep = messages[-NUM_TO_KEEP:]

        # 检查需要保留的部分是否包含未完成的 tool_calls
        # 这是一个额外的安全检查
        if isinstance(messages_to_keep[-1], AIMessage) and messages_to_keep[-1].tool_calls:
            # 如果最后一条是未响应的工具调用，那么保留更多历史以确保上下文完整
            messages_to_prune = messages[:- (NUM_TO_KEEP + 1)]
            messages_to_keep = messages[- (NUM_TO_KEEP + 1):]

        # 准备一个 prompt 让 LLM 来做总结
        summarization_prompt = [
            HumanMessage(
                content="""
                # 角色：记忆压缩专家
你现在的任务是协助一个 AI 智能体整理它的短期记忆。它正在执行“浏览社交平台并互动”的任务。
由于上下文窗口有限，你需要将它**过往的思考和操作历史**压缩成一段简练的摘要，以便它能接续工作。

# 压缩原则 (严格遵守)
1.  **数据保留 (最高优先级)**: 必须明确保留所有出现过的 **内容ID**、**主题** 以及 **平台标签**。智能体绝不能忘记它看过哪些 ID，否则会重复浏览。
2.  **状态保留**: 必须保留智能体对这些内容的 **态度** (喜欢/厌恶/无感) 以及 **已执行的操作** (已点赞/已评论/已跳过)。
3.  **意图保留**: 如果智能体在上一步表达了某种意图 (例如"我打算评论内容X") 但尚未执行，必须记录下来。
4.  **去噪**: 移除具体的 JSON 格式细节、重复的系统提示、无意义的寒暄。

# 输出格式
输出一段**第一人称**的叙述性文本。例如："我浏览了内容 [ID: xxx]，主题是科幻，我不喜欢它因为...。接着我浏览了 [ID: yyy]，我打算对它评论。"
                """),
            HumanMessage(content="这是需要压缩的对话历史："),
            *messages_to_prune,
            HumanMessage(content="请根据上述原则，生成压缩后的摘要：")
        ]

        summarizer_llm = get_async_llm(settings.model.dialogue_history_model)
        summary_response = await summarizer_llm.ainvoke(summarization_prompt)

        summary_memory = summary_response.content
        log.info(f"    -> 压缩了 {len(messages_to_prune)} 条消息，摘要为: {summary_memory}")

        # 用一条摘要消息替换掉所有被剪枝的消息
        new_messages = [system_message,
                        HumanMessage(content=f"【我给你压缩了记忆，形成的总结如下】{summary_memory}")] + messages_to_keep

        return {"messages": new_messages}

    # 1. 创建 StateGraph 实例
    workflow = StateGraph(AgentState)

    # 2. 添加所有节点
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.add_node("memory_compression", memory_compression)

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
    workflow.add_conditional_edges(
        "action",  # 起始节点
        after_tool_router,  # 判断函数
        {
            "agent": "agent",
            "memory_compression": "memory_compression",
            "end": END,
        },
    )

    # 5. 添加其他固定的边
    workflow.add_edge("memory_compression", "agent")

    graph = workflow.compile()

    # 获取可绘制的图谱
    graph_viz = graph.get_graph()

    # 生成 Mermaid 语法的字符串
    mermaid_text = graph_viz.draw_mermaid()

    with open("public_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("Public Agent Graph 已成功编译")
    return graph
