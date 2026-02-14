from itertools import cycle

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import settings


def _load_api_keys():
    """
    Iterate through all dynamic fields in settings.llm and extract all API Keys.
    """
    keys = []

    llm_config_dict = settings.llm.model_dump()

    for field_name, value in llm_config_dict.items():
        if value:
            if isinstance(value, SecretStr):
                real_key = value.get_secret_value()
            else:
                real_key = str(value)

            if len(real_key) > 5:
                keys.append(real_key)

    if not keys:
        raise ValueError("❌ No API Keys found in the configuration file (.env)! Please ensure LLM__KEY... is configured.")
    return keys


_all_keys = _load_api_keys()
_key_cycle = cycle(_all_keys)


def get_async_llm(model="qwen-plus", temperature=0.5):
    llm = ChatOpenAI(
        model=model,
        api_key=next(_key_cycle),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        request_timeout=60,  # Set request timeout to 60 seconds
        max_retries=3,
        temperature=temperature,
    )

    return llm


def get_async_flash_llm(model="qwen-flash", temperature=0.5):
    llm = ChatOpenAI(
        model=model,
        # If environment variables are not configured, please replace the next line with your API Key: api_key="sk-xxx",
        api_key=next(_key_cycle),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        request_timeout=30,  # Set request timeout to 30 seconds
        max_retries=3,
        temperature=temperature,
    )

    return llm
