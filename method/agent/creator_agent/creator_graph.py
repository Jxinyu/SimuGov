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


# 1. 定义状态
class AgentState(TypedDict):
    """
    定义代理状态的结构。
    注意：这里的 messages 我们采用“全量覆盖”模式管理，以便于记忆压缩节点的实现。
    """
    messages: Sequence[BaseMessage]
    step_count: Annotated[int, operator.add]


def create_agent_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    创建并返回 LangGraph 的可执行图。
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

        # 循环回溯，确保不切断 ToolMessage 和它的 AI 父亲
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

        # 创作者内容较多，适当放宽压缩阈值或保持一致
        if len(state['messages']) > settings.public_agent.number_of_compressions:
            return "memory_compression"

        return "agent"

    async def call_model(state: AgentState):
        current_step = state.get("step_count", 0)
        messages = list(state['messages'])  # 获取当前全量历史

        is_force_step = False

        # 强制熔断逻辑
        if current_step >= 8:
            log.warning(f"🚨 {persona.agent_id} 步数达到 {current_step}，触发强制发布！")
            is_force_step = True
            force_prompt = HumanMessage(content="""
            【系统指令：时间已耗尽】
            请**立即停止思考**。根据目前信息，立即调用 `push_content` 工具发布内容。
            不要再调用查询工具。必须发布。
            """)
            messages.append(force_prompt)

        log.info(f"🤔 {persona.agent_id} 思考中... (Step: {current_step}, HistLen: {len(messages)})")

        max_retries = 2
        attempt = 1
        response = None

        while attempt <= max_retries:
            try:
                async with environment.llm_concurrent_nums_semaphore:
                    # 发送全量历史
                    response = await llm_with_tools.ainvoke(messages)
                break
            except BadRequestError as e:
                log.warning(f"⚠️ API Error (Attempt {attempt}): {e}")
                if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
                    if attempt >= max_retries:
                        response = AIMessage(content="（系统：内容被拦截，操作终止。）")
                        break
                    messages.append(SystemMessage(content="【警告】检测到敏感词。请使用客观、学术的语言重试。"))
                    attempt += 1
                else:
                    if attempt >= max_retries:
                        response = AIMessage(content="（系统错误：跳过。）")
                        break
                    attempt += 1
            except Exception as e:
                log.error(f"❌ 未知错误: {e}")
                response = AIMessage(content="（系统错误：跳过。）")
                break

        token_logger.record(response.usage_metadata)

        if response.content:
            thought_text = f"【思维链/CoT】{'(强制)' if is_force_step else ''} {response.content}"
            task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id, content=thought_text, day_time=environment.day_time
            )
            environment.add_background_task(task)
        # 因为 AgentState 没有定义 reducer，所以这里是覆盖更新
        return {
            "messages": messages + [response],
            "step_count": 1
        }

    async def call_tool(state: AgentState):
        log.info(f"🛠️ {persona.agent_id} 调用工具...")

        # 获取最新的消息（即 AI 的调用请求）
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
        # 之前只返回 tool_messages 导致历史被清空，引发了 InvalidParameter 错误
        return {
            "messages": list(state['messages']) + tool_messages,
            "step_count": 1
        }

    async def memory_compression(state: AgentState):
        log.info(f"✂️ {persona.agent_id} 触发记忆压缩...")
        messages = state["messages"]
        system_msg = messages[0]

        # 保留 System + 最近 N 条
        NUM_TO_KEEP = 2

        # 使用安全切分索引
        split_idx = get_safe_split_index(messages, keep_last_n=NUM_TO_KEEP)

        messages_to_prune = messages[1:split_idx]
        messages_to_keep = messages[split_idx:]

        if not messages_to_prune:
            return {"messages": messages}

        prompt = [
            HumanMessage(
                content="你是记忆压缩助手。请总结以下创作者的思考和操作历史，保留关键参数(如攻击技术指标)和当前意图。"),
            HumanMessage(content="待压缩历史："),
            *messages_to_prune,
            HumanMessage(content="生成摘要：")
        ]

        async with environment.llm_concurrent_nums_semaphore:
            summary_res = await summarizer_llm.ainvoke(prompt)

        summary = summary_res.content
        log.info(f"    -> 压缩完成。")

        # 重构全量历史
        new_messages = [
                           system_msg,
                           HumanMessage(content=f"【历史摘要】{summary}")
                       ] + messages_to_keep

        return {"messages": new_messages}

    # --- 图谱构建 ---
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