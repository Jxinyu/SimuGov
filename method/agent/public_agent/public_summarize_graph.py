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
from config import settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    定义代理状态的结构。
    """
    messages: Sequence[BaseMessage]
    tools_call_str: Annotated[List[str], operator.add]
                                  
    is_finalizing: bool


def create_agent_summarize_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    """
    创建一个带有“最终检查”逻辑的 LangGraph 可执行图。
    """
    llm = get_async_llm(settings.model.public_summarize_model)
    llm_with_tools = llm.bind_tools(tools)

    def after_agent_router(state: AgentState):
        """
        模型在主循环中思考后，决定是继续行动、进入最终检查，还是结束。
        """
        last_message = state['messages'][-1]
        tools_names_str = state.get('tools_call_str', [])

                                              
        if len(tools_names_str) > 3 and (len(set(tools_names_str[-3:])) == 1 or len(tools_names_str) >= 8):
            log.info("检测到循环或达到长度限制，进入最终检查流程...")
            return "summarize"

        if last_message.tool_calls:
            return "action"

        return "end"

    def after_summarize_router(state: AgentState):
        """
        在'summarize'节点之后进行路由。
        如果 summarize 节点要求调用工具，则去 action；否则结束。
        """
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            log.info("最终检查后需要调用工具，路由到 action。")
            return "action"
        else:
            log.info("最终检查完成，无需额外操作，路由到 END。")
            return "end"

    def after_tool_router(state: AgentState):
        """
        在工具调用之后进行路由。
        这是控制循环的关键。
        """
                                                      
        if state.get("is_finalizing"):
            log.info("最终强制工具已执行，流程结束。")
            return "end"

                                  
        return "agent"

    async def call_model(state: AgentState):
        """主循环中的模型思考节点。"""
        log.info(f"🤔🤔🤔  {persona.agent_id} 调用模型思考...➡️➡️➡️  {persona.agent_id}模型输入：{state['messages']}")
        async with environment.llm_concurrent_nums_semaphore:
            response = await llm_with_tools.ainvoke(state['messages'])
        log.info(f"🔚🔚🔚  {persona.agent_id}  模型思考返回 {response} ...")

        if response.content:
            thought_text = f"【思维链/CoT】{response.content}"

            save_thought_task = environment.memories_store.add_agent_think_memory(
                persona_id=persona.agent_id,
                content=thought_text,
                day_time=environment.day_time,
            )
            environment.add_background_task(save_thought_task)

        return {"messages": list(state['messages']) + [response]}

    async def call_model_summarize(state: AgentState):
        """
        最终检查节点。
        检查 'update_persona_data' 是否被调用，如果没有，则强制调用。
        """
        log.info(f"🤔🤔🤔  {persona.agent_id} 进入 summarize 节点进行最终检查...")
        tools_names_str = state.get('tools_call_str', [])

                      
        if "update_persona_data" in tools_names_str:
            log.info("'update_persona_data' 已被调用，流程可以安全结束。")
            return state

                             
        log.warning("'update_persona_data' 从未被调用！强制执行最终操作。")

        messages = list(state['messages'])

                        
                                       
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            log.info(f"清理掉来自上一轮的、未被执行的工具调用请求: {messages[-1].tool_calls}")
            messages = messages[:-1]

        force_prompt = HumanMessage(
            content="""
            重要：你的任务还没有完成。你必须调用 `update_persona_data` 工具来更新个人数据。
            这是你在此次任务中的最后一步，也是必须执行的一步。请立即调用该工具。
            """
        )
                    
        messages_for_force_call = messages + [force_prompt]

        response = await llm_with_tools.ainvoke(messages_for_force_call)
        log.info(f"🔚🔚🔚  {persona.agent_id}  强制调用后，模型返回： {response} ...")

        if response.content:
            thought_text = f"【思维链/CoT】{response.content}"

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
        """工具执行节点（对主循环和最终调用都有效）。"""
        log.info(f"🤔🤔🤔  {persona.agent_id} 调用工具...")
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

                tasks.append(_get_error(f"错误：工具 '{tool_name}' 不存在。"))

        tool_responses = await asyncio.gather(*tasks)
        tool_messages = []
        for response, tool_call in zip(tool_responses, last_message.tool_calls):
            tool_messages.append(
                ToolMessage(content=str(response), tool_call_id=tool_call['id'], name=tool_call['name'])
            )
        log.info(f"🔚🔚🔚  {persona.agent_id} 调用工具 响应：{tool_responses}")
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
            "agent": "agent",         
            "end": END,        
        }
    )

    graph = workflow.compile()

              
    graph_viz = graph.get_graph()

                       
    mermaid_text = graph_viz.draw_mermaid()

    with open("public_summarize_graph.mermaid", "w") as f:
        f.write(mermaid_text)
    log.info("public_summarize_graph 已成功编译")

    return graph
