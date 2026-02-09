from itertools import cycle

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import settings


def _load_api_keys():
    """
    遍历 settings.llm 中的所有动态字段，提取出所有的 API Key。
    """
    keys = []

    llm_config_dict = settings.llm.model_dump()

    for field_name, value in llm_config_dict.items():
        # 简单的过滤：值不能为空
        if value:
            # 处理类型：extra='allow' 读进来的可能是 str，也可能是 SecretStr
            if isinstance(value, SecretStr):
                real_key = value.get_secret_value()
            else:
                real_key = str(value)

            # 简单的验证：确保 key 看起来像是一个 key (可选)
            if len(real_key) > 5:
                keys.append(real_key)

    if not keys:
        raise ValueError("❌ 未在配置文件(.env)中找到任何 API Key！请确保配置了 LLM__KEY... ")
    return keys


_all_keys = _load_api_keys()
_key_cycle = cycle(_all_keys)


def get_async_llm(model="qwen-plus", temperature=0.5):
    llm = ChatOpenAI(
        model=model,
        api_key=next(_key_cycle),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        request_timeout=60,  # 设置请求超时为30秒
        max_retries=3,
        temperature=temperature,
    )

    return llm


def get_async_flash_llm(model="qwen-flash", temperature=0.5):
    llm = ChatOpenAI(
        model=model,
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=next(_key_cycle),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        request_timeout=30,  # 设置请求超时为30秒
        max_retries=3,
        temperature=temperature,
    )

    return llm
