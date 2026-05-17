import logging
from typing import Any, List, Set, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.messages import AIMessage, SystemMessage
from langchain.tools import BaseTool

from method.agent.persona import Persona
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from config import settings

log = logging.getLogger(__name__)


class PublicStateMiddleware(AgentMiddleware):
    """
    负责：
    1. 维护公共状态
    2. 维护已读 / 已互动内容 ID
    3. 只做浏览阶段状态维护，不再在 graph 内触发“最终反思”
    """

    def __init__(
        self,
        environment: Environment,
        persona: Persona,
        max_interactions: int,
        max_model_calls: int = 12,
        react_tool_name: str = "react_to_content",
        finish_tool_name: str = "finish_browsing",
        read_tool_names: tuple[str, ...] = (
            "get_content_detail",
            "read_content",
            "view_content",
            "browse_content",
        ),
    ):
        self.environment = environment
        self.persona = persona
        self.max_interactions = max_interactions
        self.max_model_calls = max_model_calls
        self.react_tool_name = react_tool_name
        self.finish_tool_name = finish_tool_name
        self.read_tool_names = set(read_tool_names)

    def before_model(self, state: dict, runtime: Any) -> Optional[dict]:
                              
        if state.get("public_finished", False):
            return {"messages": [SystemMessage(content="本轮浏览已结束，禁止继续调用任何工具。")]}
        return None

    def _get_model_call_count(self, state: dict) -> int:
        if "public_model_call_count" in state:
            return int(state["public_model_call_count"])
        messages = state.get("messages", [])
        return sum(1 for m in messages if isinstance(m, AIMessage))

    def _get_reacted_ids(self, state: dict) -> Set[str]:
        return set(state.get("content_already_reacted_ids", set()) or set())

    def _get_read_ids(self, state: dict) -> Set[str]:
        return set(state.get("content_already_read_ids", set()) or set())

    def _extract_text_content(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()

        return str(content).strip()

    def after_model(self, state: dict) -> Optional[dict]:
        messages = state.get("messages", [])
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {
                "public_model_call_count": self._get_model_call_count(state)
            }

        return {
            "public_model_call_count": self._get_model_call_count(state) + 1
        }

    def after_tool(self, state: dict) -> Optional[dict]:
        messages = state.get("messages", [])
        if not messages:
            return None

        reacted_ids = self._get_reacted_ids(state)
        read_ids = self._get_read_ids(state)

        last_ai = None
        for message in reversed(messages):
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                last_ai = message
                break

        if last_ai is None:
            return None

        public_finished = bool(state.get("public_finished", False))

        for tool_call in last_ai.tool_calls:
            tool_name = tool_call.get("name")
            args = tool_call.get("args", {}) or {}

            if tool_name == self.finish_tool_name:
                public_finished = True
                continue

            content_id = args.get("content_id")
            if not content_id:
                continue

            if tool_name == self.react_tool_name:
                reacted_ids.add(content_id)
                read_ids.add(content_id)
            elif tool_name in self.read_tool_names:
                read_ids.add(content_id)

        return {
            "content_already_reacted_ids": reacted_ids,
            "content_already_read_ids": read_ids,
            "public_finished": public_finished,
        }


class PublicInteractionGuardMiddleware(AgentMiddleware):
    """
    负责：
    1. 防止重复 react 同一条内容
    2. 不再处理反思阶段，因为反思已经移到 graph 外
    """

    def __init__(self, react_tool_name: str = "react_to_content"):
        self.react_tool_name = react_tool_name

    def before_model(self, state: dict) -> Optional[dict]:
        if state.get("public_finished", False):
            return {"messages": [SystemMessage(content="你已经结束本轮浏览。不要再互动或浏览，直接结束。")]}

        reacted_ids = sorted(set(state.get("content_already_reacted_ids", set()) or set()))

        if not reacted_ids:
            return None

        return {"messages": [
            SystemMessage(
                content=(
                    f"你已经互动过这些内容ID：{reacted_ids}。\n"
                    f"不要再次对这些内容调用 {self.react_tool_name}。"
                )
            )
        ]}


class PublicToolAuditMiddleware(AgentMiddleware):
    """
    可选审计日志中间件
    """

    def __init__(self, persona: Persona):
        self.persona = persona

    def after_tool(self, state: dict) -> Optional[dict]:
        try:
            messages = state.get("messages", [])
            last_ai = None
            for message in reversed(messages):
                if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                    last_ai = message
                    break

            log.info(
                "public tool finished: persona=%s last_tool_calls=%s",
                self.persona.agent_id,
                getattr(last_ai, "tool_calls", None),
            )
        except Exception:
            pass
        return None


def build_public_final_reflection_prompt(state: dict) -> str:
    """
    graph 外部使用的最终反思 prompt。
    只做一次、无工具调用。
    """
    messages = state.get("messages", [])
    reacted_ids = sorted(set(state.get("content_already_reacted_ids", set()) or set()))
    read_ids = sorted(set(state.get("content_already_read_ids", set()) or set()))

    readable_history = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        role = None
        if isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = getattr(msg, "type", None) or msg.__class__.__name__

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
        "你现在要对刚刚完成的一轮平台浏览与互动做【最终反思】。\n"
        "这是一次独立的总结，不允许调用任何工具，不允许继续浏览，不允许继续互动。\n"
        "你的输出将被写入长期记忆。\n\n"
        f"本轮已阅读内容ID：{read_ids}\n"
        f"本轮已互动内容ID：{reacted_ids}\n\n"
        "以下是本轮执行过程中的关键信息：\n"
        f"{history_text}\n\n"
        "请基于以上内容，输出一段详细、可复用的复盘总结，覆盖：\n"
        "1. 本轮关注了哪些关键信息\n"
        "2. 对哪些内容做了互动，以及为什么这样互动\n"
        "3. 本轮判断内容价值时采用了什么标准\n"
        "4. 后续可复用的经验或策略\n"
        "5. 这次执行中应避免的问题\n\n"
        "要求：\n"
        "- 直接输出反思正文\n"
        "- 使用中文\n"
        "- 不要说“好的”或“下面是总结”\n"
        "- 不要虚构未发生的阅读、互动或工具结果"
    )


def create_agent_graph(
    tools: List[BaseTool],
    environment: Environment,
    persona: Persona,
):
    """
    执行链路（仅浏览阶段）：
    1. 浏览 / 阅读 / 互动
    2. 命中模型 / 工具上限后自然结束
    3. 最终反思不在 graph 内执行，而是在 graph 外单独调用一次无工具 LLM
    """
    llm = get_async_llm("qwen-flash")
    summarizer_llm = get_async_llm("qwen-flash")

    react_tool_name = "react_to_content"
    finish_tool_name = "finish_browsing"
    max_interactions = settings.public_agent.number_of_interactions
    keep_messages = settings.public_agent.number_of_keep
    max_model_calls = 8

    tool_names = [tool.name for tool in tools]
    non_react_tools = [name for name in tool_names if name not in {react_tool_name, finish_tool_name}]

    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[
            SummarizationMiddleware(
                model=summarizer_llm,
                max_tokens_before_summary=1580,
                messages_to_keep=keep_messages,
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
                run_limit=max_model_calls,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=10,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                tool_name=react_tool_name,
                run_limit=max_interactions,
                exit_behavior="end",
            ),
            *[
                ToolCallLimitMiddleware(
                    tool_name=name,
                    run_limit=3,
                    exit_behavior="end",
                )
                for name in non_react_tools
            ],
            ToolCallLimitMiddleware(
                tool_name=finish_tool_name,
                run_limit=1,
                exit_behavior="end",
            ),
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            PublicStateMiddleware(
                environment=environment,
                persona=persona,
                max_interactions=max_interactions,
                max_model_calls=max_model_calls,
                react_tool_name=react_tool_name,
                finish_tool_name=finish_tool_name,
            ),
            PublicInteractionGuardMiddleware(
                react_tool_name=react_tool_name,
            ),
            PublicToolAuditMiddleware(
                persona=persona,
            ),
        ],
    )
    return agent
