import threading
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr, SecretStr

from config import settings

_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _load_api_keys() -> list[str]:
    keys: list[str] = []
    llm_config_dict = settings.llm.model_dump()

    for _, value in llm_config_dict.items():
        if not value:
            continue
        real_key = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        if len(real_key) > 5:
            keys.append(real_key)

    if not keys:
        raise ValueError("No API key was found in the LLM settings.")
    return keys


class _KeyManager:
    def __init__(self, keys: list[str]) -> None:
        self._keys = list(keys)
        self._idx = 0
        self._lock = threading.Lock()

    def snapshot_size(self) -> int:
        with self._lock:
            return len(self._keys)

    def get_next(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("No available API key.")
            key = self._keys[self._idx % len(self._keys)]
            self._idx = (self._idx + 1) % len(self._keys)
            return key


_key_manager: _KeyManager | None = None


def _get_key_manager() -> _KeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = _KeyManager(_load_api_keys())
    return _key_manager


def get_api_key_count() -> int:
    return len(_load_api_keys())


def _build_chat_model(api_key: str, model: str, temperature: float, request_timeout: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=_BASE_URL,
        request_timeout=request_timeout,
        max_retries=0,
        temperature=temperature,
    )


def _invoke_with_key_rotation(
    input_data: Any,
    model: str,
    temperature: float,
    request_timeout: int,
    structured_schema: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> Any:
    key_manager = _get_key_manager()
    max_attempts = key_manager.snapshot_size()
    if max_attempts <= 0:
        raise RuntimeError("No available API key.")

    last_error: Exception | None = None
    for _ in range(max_attempts):
        current_key = key_manager.get_next()
        base_llm = _build_chat_model(current_key, model, temperature, request_timeout)
        try:
            if structured_schema is None:
                return base_llm.invoke(input_data, config=config, **kwargs)

            structured_llm = base_llm.with_structured_output(structured_schema, include_raw=True)
            raw_bundle = structured_llm.invoke(input_data, config=config, **kwargs)
            if isinstance(raw_bundle, dict):
                parsing_error = raw_bundle.get("parsing_error")
                if parsing_error:
                    raise parsing_error
                return raw_bundle.get("parsed")

            return raw_bundle
        except Exception as exc:
            last_error = exc
            print("[LLM Key] current key failed; trying the next configured key.")
            continue

    raise RuntimeError(f"All configured API keys failed. Last error: {last_error}")


async def _ainvoke_with_key_rotation(
    input_data: Any,
    model: str,
    temperature: float,
    request_timeout: int,
    structured_schema: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> Any:
    key_manager = _get_key_manager()
    max_attempts = key_manager.snapshot_size()
    if max_attempts <= 0:
        raise RuntimeError("No available API key.")

    last_error: Exception | None = None
    for _ in range(max_attempts):
        current_key = key_manager.get_next()
        base_llm = _build_chat_model(current_key, model, temperature, request_timeout)
        try:
            if structured_schema is None:
                return await base_llm.ainvoke(input_data, config=config, **kwargs)

            structured_llm = base_llm.with_structured_output(structured_schema, include_raw=True)
            raw_bundle = await structured_llm.ainvoke(input_data, config=config, **kwargs)
            if isinstance(raw_bundle, dict):
                parsing_error = raw_bundle.get("parsing_error")
                if parsing_error:
                    raise parsing_error
                return raw_bundle.get("parsed")

            return raw_bundle
        except Exception as exc:
            last_error = exc
            print("[LLM Key] current key failed; trying the next configured key.")
            continue

    raise RuntimeError(f"All configured API keys failed. Last error: {last_error}")


class RotatingChatOpenAI(ChatOpenAI):
    _rot_model: str = PrivateAttr(default="qwen-flash")
    _rot_temperature: float = PrivateAttr(default=0.5)
    _rot_timeout: int = PrivateAttr(default=60)
    _structured_schema: Any = PrivateAttr(default=None)

    def setup_rotation(
        self,
        model: str,
        temperature: float,
        request_timeout: int,
        structured_schema: Any = None,
    ) -> "RotatingChatOpenAI":
        self._rot_model = model
        self._rot_temperature = temperature
        self._rot_timeout = request_timeout
        self._structured_schema = structured_schema
        return self

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        return _invoke_with_key_rotation(
            input_data=input_data,
            model=self._rot_model,
            temperature=self._rot_temperature,
            request_timeout=self._rot_timeout,
            structured_schema=self._structured_schema,
            config=config,
            **kwargs,
        )

    async def ainvoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        return await _ainvoke_with_key_rotation(
            input_data=input_data,
            model=self._rot_model,
            temperature=self._rot_temperature,
            request_timeout=self._rot_timeout,
            structured_schema=self._structured_schema,
            config=config,
            **kwargs,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return _build_patched_llm(
            model=self._rot_model,
            temperature=self._rot_temperature,
            request_timeout=self._rot_timeout,
            structured_schema=schema,
        )


def _build_patched_llm(
    model: str,
    temperature: float,
    request_timeout: int,
    structured_schema: Any = None,
):
    seed_key = _get_key_manager().get_next()
    llm = RotatingChatOpenAI(
        model=model,
        api_key=seed_key,
        base_url=_BASE_URL,
        request_timeout=request_timeout,
        max_retries=0,
        temperature=temperature,
    )
    return llm.setup_rotation(
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        structured_schema=structured_schema,
    )


def get_async_llm(model: str = "qwen-flash", temperature: float = 0.5):
    return _build_patched_llm(model=model, temperature=temperature, request_timeout=60)


def get_async_flash_llm(model: str = "qwen-flash", temperature: float = 0.5):
    return _build_patched_llm(model=model, temperature=temperature, request_timeout=30)
