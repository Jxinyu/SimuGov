import asyncio
from ollama import Client, AsyncClient  # <-- Import both clients
from langchain_core.embeddings import Embeddings
from typing import List

ollama_async_client = AsyncClient()
ollama_sync_client = Client()


class OllamaEmbeddings(Embeddings):
    """
    A custom Embedding class that uses the Ollama local model and is compatible with LangChain.
    """
    model_name: str = 'qwen3-embedding:4b'

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously generate vectors for a set of documents"""
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously generate a vector for a single query text"""
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]

    # --- Synchronous methods ---
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronously generate vectors for a set of documents"""
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    def embed_query(self, text: str) -> List[float]:
        """Synchronously generate a vector for a single query text"""
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]


default_ollama_embedding_function = OllamaEmbeddings()
