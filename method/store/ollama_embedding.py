                                                                    

import asyncio
from ollama import Client, AsyncClient               
from langchain_core.embeddings import Embeddings
from typing import List

                           
                      
ollama_async_client = AsyncClient()
                     
ollama_sync_client = Client()


class OllamaEmbeddings(Embeddings):
    """
    一个使用Ollama本地模型并兼容LangChain的自定义Embedding类。
    【最终修正版：分离同步和异步实现】
    """
    model_name: str = 'qwen3-embedding:4b'                       

                  
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步地为一组文档生成向量"""
                 
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    async def aembed_query(self, text: str) -> List[float]:
        """异步地为单个查询文本生成向量"""
                 
        response = await ollama_async_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]

                  
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步地为一组文档生成向量"""
                                            
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=texts
        )
        return response['embeddings']

    def embed_query(self, text: str) -> List[float]:
        """同步地为单个查询文本生成向量"""
                                            
        response = ollama_sync_client.embed(
            model=self.model_name,
            input=text
        )
        return response['embeddings'][0]


                             
default_ollama_embedding_function = OllamaEmbeddings()
