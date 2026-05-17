import logging
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware, AgentMiddleware, before_model,
)
from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain.tools import BaseTool
from openai import BadRequestError

from method.agent.persona import Persona
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from config import settings

log = logging.getLogger(__name__)


class CreatorPolicyMiddleware(AgentMiddleware):
    """
    负责：
    1. 达到阈值后强制发布
    2. 维护调用流程稳定
    3. 标记是否已经完成 push_content
    4. 处理 BadRequest 降级
    """

    def __init__(
        self,
        environment: Environment,
        persona: Persona,
        force_after_model_calls: int = 8,
        push_tool_name: str = "push_content",
        finish_tool_name: str = "finish_creation",
    ):
        self.environment = environment
        self.persona = persona
        self.force_after_model_calls = force_after_model_calls
        self.push_tool_name = push_tool_name
        self.finish_tool_name = finish_tool_name
                                                  
        self.push_content_llm = get_async_llm("qwen-plus")

    def _extract_tool_name(self, tool_result: Any) -> Optional[str]:
        if hasattr(tool_result, "tool"):
            return getattr(tool_result, "tool", None)
        if hasattr(tool_result, "name"):
            return getattr(tool_result, "name", None)
        if isinstance(tool_result, dict):
            return tool_result.get("tool") or tool_result.get("name")
        return None

    def _extract_tool_names_from_model_response(self, response: Any) -> List[str]:
        names: List[str] = []
        visited: set[int] = set()

        def _walk(node: Any) -> None:
            if node is None:
                return
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)

                                                           
            if isinstance(node, (list, tuple, set)):
                for item in node:
                    _walk(item)
                return

                     
            if isinstance(node, dict):
                                            
                if "tool_calls" in node:
                    _walk(node.get("tool_calls"))
                                    
                if "function_call" in node:
                    fc = node.get("function_call") or {}
                    if isinstance(fc, dict):
                        fc_name = fc.get("name")
                        if fc_name:
                            names.append(str(fc_name))
                         
                call_name = node.get("name")
                if call_name and ("args" in node or "arguments" in node or "function" in node):
                    names.append(str(call_name))
                fn = node.get("function") or {}
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if fn_name:
                        names.append(str(fn_name))
                                      
                if node.get("type") in {"tool_use", "tool_call"} and node.get("name"):
                    names.append(str(node["name"]))

                for v in node.values():
                    _walk(v)
                return

                                            
            tool_calls = getattr(node, "tool_calls", None)
            if tool_calls:
                _walk(tool_calls)

            additional_kwargs = getattr(node, "additional_kwargs", None)
            if additional_kwargs:
                _walk(additional_kwargs)

            content = getattr(node, "content", None)
            if content is not None:
                _walk(content)

                        
            for attr in ("result", "message", "output", "response"):
                v = getattr(node, attr, None)
                if v is not None:
                    _walk(v)

        _walk(response)

                 
        deduped: List[str] = []
        for n in names:
            if n not in deduped:
                deduped.append(n)
        return deduped

    def _get_model_call_count(self, state: dict) -> int:
        if "creator_model_call_count" in state:
            return int(state["creator_model_call_count"])
        messages = state.get("messages", [])
        return sum(1 for m in messages if isinstance(m, AIMessage))

    def _request_with_update(self, request: Any, update: dict) -> Any:
                                           
        if hasattr(request, "model_copy"):
            return request.model_copy(update=update)
        if hasattr(request, "copy"):
            return request.copy(update=update)

                          
        for k, v in update.items():
            setattr(request, k, v)
        return request

    def before_model(self, state: dict, runtime: Any) -> Optional[dict]:
        messages = list(state.get("messages", []))
        model_calls = self._get_model_call_count(state)

                            
        if state.get("creator_published", False) or state.get("creator_finished", False):
            return None

        if model_calls >= self.force_after_model_calls:
            log.warning(
                "🚨 %s 模型调用数达到 %s，触发强制发布",
                self.persona.agent_id,
                model_calls,
            )
            return {"messages": [
                HumanMessage(
                    content=(
                        "【系统指令：时间已耗尽】\n"
                        "请立即停止继续规划或查询。\n"
                        f"优先调用 `{self.push_tool_name}` 发布内容；如果你明确决定今天不发布，"
                        f"必须调用 `{self.finish_tool_name}` 结束流程。\n"
                        "不要再调用查询类工具。"
                    )
                )
            ]}

        return None

    def after_model(self, state: dict) -> Optional[dict]:
        messages = state.get("messages", [])

        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {
                "creator_model_call_count": self._get_model_call_count(state)
            }

        return {
            "creator_model_call_count": self._get_model_call_count(state) + 1
        }

    async def awrap_model_call(self, request: Any, handler: Any) -> AIMessage:
        max_retries = 2
        attempt = 1

        while attempt <= max_retries:
            try:
                response = await handler(request)
                tool_names = self._extract_tool_names_from_model_response(response)
                will_push_content = self.push_tool_name in tool_names

                                                                  
                if will_push_content:
                    log.info(
                        "🔁 %s 检测到即将调用 %s，临时切换 qwen-plus 重跑当前模型轮次",
                        self.persona.agent_id,
                        self.push_tool_name,
                    )
                    original_model = getattr(request, "model", None)
                    plus_request = self._request_with_update(request, {"model": self.push_content_llm})
                    response = await handler(plus_request)
                                             
                    if plus_request is request:
                        setattr(request, "model", original_model)
                elif tool_names:
                    log.info(
                        "ℹ️ %s 本轮工具调用未命中 push_content: %s",
                        self.persona.agent_id,
                        tool_names,
                    )
                else:
                    log.debug(
                        "ℹ️ %s 本轮未检测到工具调用（response_type=%s）",
                        self.persona.agent_id,
                        type(response).__name__,
                    )

                return response
            except BadRequestError as e:
                log.warning("⚠️ API Error (Attempt %s): %s", attempt, e)

                msg = str(e)
                if "data_inspection_failed" in msg or "inappropriate content" in msg:
                    if attempt >= max_retries:
                        return AIMessage(content="（系统：内容被拦截，操作终止。）")

                    patched_messages = list(request.state.get("messages", []))
                    patched_messages.append(
                        SystemMessage(
                            content="【警告】检测到敏感词。请使用客观、学术、中性语言重试。"
                        )
                    )
                    request = self._request_with_update(
                        request,
                        {"state": {**request.state, "messages": patched_messages}},
                    )
                    attempt += 1
                    continue

                if attempt >= max_retries:
                    return AIMessage(content="（系统错误：跳过。）")
                attempt += 1

            except Exception as e:
                log.error("❌ 未知错误: %s", e)
                return AIMessage(content="（系统错误：跳过。）")

        return AIMessage(content="（系统错误：跳过。）")

    def after_tool(self, state: dict, runtime: Any, tool_result: Any) -> Optional[dict]:
        tool_name = self._extract_tool_name(tool_result)
        if tool_name == self.push_tool_name:
            log.info("✅ %s 已调用 %s，发布阶段完成", self.persona.agent_id, self.push_tool_name)
            return {
                "creator_published": True,
                "creator_finished": True,
            }
        if tool_name == self.finish_tool_name:
            log.info("✅ %s 已调用 %s，创作阶段结束", self.persona.agent_id, self.finish_tool_name)
            return {
                "creator_finished": True,
            }
        return None


def build_creator_final_reflection_prompt(state: dict) -> str:
    messages = state.get("messages", [])
    readable_history = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        role = "AI" if isinstance(msg, AIMessage) else getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        else:
            content = str(content)

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            content = f"{content}\n[tool_calls={tool_calls}]"

        if content.strip():
            readable_history.append(f"[{role}] {content.strip()}")

    history_text = "\n\n".join(readable_history[-20:]) if readable_history else "无有效历史。"

    return (
        "你已经完成本轮创作与发布，现在需要做一次最终复盘反思。\n"
        "这是一次独立总结，不允许调用任何工具，不允许继续规划，不允许继续修改内容。\n"
        "你的输出将写入长期记忆。\n\n"
        "请基于以下本轮执行历史，输出一段中文复盘，总结：\n"
        "1. 最终发布了什么\n"
        "2. 为什么这样发布\n"
        "3. 浏览、判断和创作时最关键的依据是什么\n"
        "4. 哪些经验可以复用\n"
        "5. 下次应避免什么问题\n\n"
        f"{history_text}\n\n"
        "要求：直接输出正文，不要加标题，不要寒暄，不要虚构未发生的过程。"
    )


def create_agent_graph(tools: List[BaseTool], environment: Environment, persona: Persona):
    llm = get_async_llm("qwen-flash")
    summarizer_llm = get_async_llm("qwen-flash")

    query_tool_names = [t.name for t in tools if t.name not in {"push_content", "finish_creation"}]

    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[
            SummarizationMiddleware(
                model=summarizer_llm,
                max_tokens_before_summary=1580,
                messages_to_keep=5,
                summary_prompt=(
                    "# 角色：记忆压缩专家\n"
                    "你现在的任务是协助一个 AI 智能体整理它的短期记忆。"
                    "它正在执行“浏览社交平台并互动”的任务。\n\n"
                    "# 压缩原则（严格遵守）\n"
                    "1. 当前目标\n"
                    "2. 已完成的关键操作\n"
                    "3. 未完成的任务\n"
                    "4. 关键状态或策略\n\n"
                    "避免丢失未完成任务和当前意图。\n"
                    "删除重复、细节和无关内容。\n"
                    "用第一人称简要总结。"
                ),
            ),
            ModelCallLimitMiddleware(
                run_limit=6,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=5,
                exit_behavior="end",
            ),
            *[
                ToolCallLimitMiddleware(
                    tool_name=name,
                    run_limit=2,
                    exit_behavior="end",
                )
                for name in query_tool_names
            ],
            ToolCallLimitMiddleware(
                tool_name="finish_creation",
                run_limit=1,
                exit_behavior="end",
            ),
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            CreatorPolicyMiddleware(
                environment=environment,
                persona=persona,
                force_after_model_calls=6,
                push_tool_name="push_content",
                finish_tool_name="finish_creation",
            ),
        ],
    )

    return agent

