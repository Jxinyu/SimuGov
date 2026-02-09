# method/store/ollama_embedding.py (或 method/utils/ollama_client.py)

import asyncio
from ollama import Client, AsyncClient  # <-- 导入两个客户端
from langchain_core.embeddings import Embeddings
from typing import List

# --- 创建两个全局的、可重用的客户端实例 ---
# 异步客户端，用于 aembed_* 方法
ollama_async_client = AsyncClient()
# 同步客户端，用于 embed_* 方法
ollama_sync_client = Client()


class OllamaEmbeddings(Embeddings):
    """
    一个使用Ollama本地模型并兼容LangChain的自定义Embedding类。
    【最终修正版：分离同步和异步实现】
    """
    model_name: str = 'qwen3-embedding:4b'  # 你可以把它放到config.yaml中

    # --- 异步方法 ---
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步地为一组文档生成向量"""
        # 使用异步客户端
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    async def aembed_query(self, text: str) -> List[float]:
        """异步地为单个查询文本生成向量"""
        # 使用异步客户端
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]

    # --- 同步方法 ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步地为一组文档生成向量"""
        # 【核心修复】: 使用同步客户端，不再调用 asyncio.run()
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    def embed_query(self, text: str) -> List[float]:
        """同步地为单个查询文本生成向量"""
        # 【核心修复】: 使用同步客户端，不再调用 asyncio.run()
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]


# --- 导出一个可直接使用的实例 (保持不变) ---
default_ollama_embedding_function = OllamaEmbeddings()
