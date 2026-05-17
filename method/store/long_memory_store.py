import json
import math
import os
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from config import settings
from method.utils.psychological_parameter_mapping_table import psycho_numeric_for_recall
from method.store.ollama_embedding import default_ollama_embedding_function

log = logging.getLogger(__name__)

DEFAULT_EMBEDDING_FUNCTION = default_ollama_embedding_function


class MemoryType(Enum):
    """
    定义智能体记忆的三种核心类型，代表了从具体经验到抽象信念的认知层次。
    """
    EXPERIENCE = 'experience'
    """
    **类型：经验 **
    """

                       
    BELIEF = 'experience'
    """
    **类型：信念 **
    """

                             
    SUMMARIZE = 'experience'
    """
    **类型：总结 **
    """


class MemoryStore:
    """
    一个用于管理智能体记忆的向量存储库。
    """

    def __init__(self,
                 collection_name: str = "simulation_memories",
                 content_collection_name: str = "platform_contents",          
                 embedding_function: Embeddings = DEFAULT_EMBEDDING_FUNCTION):
        log.info(f"正在初始化 MemoryStore...")

                   
        current_now = datetime.now()
        current_time = current_now.strftime("%H_%M_%S")

                                                
        unique_suffix = uuid.uuid4().hex[:8]

                   
        persist_directory = (
            f"{settings.file_load_path.base_store_file}/"
            f"chromadb/{current_time}_{unique_suffix}_db"
        )

                            
        self.persist_directory = persist_directory
        log.info(f"📂 本次运行的独立数据库路径: {self.persist_directory}")

                    
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
        )

                   
                                           
        self.content_vectorstore = Chroma(
            collection_name=content_collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

                              
        self.comments_vectorstore = Chroma(
            collection_name="platform_comments",
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

        log.info("✅ MemoryStore 初始化成功 (包含记忆库与内容库)。")

    async def add_content_to_db(self, content_obj) -> bool:
        """
        将内容向量化并存入 ChromaDB 的内容集合中。
        Args:
            content_obj: Content 对象
        """
        try:
                                
                                
            embed_text = f"主题: {content_obj.topic}\n内容: {content_obj.content_detail}"

                   
            metadata = {
                "content_id": content_obj.id,
                "author_id": content_obj.author_id,
                "time": content_obj.time,            
                "topic": content_obj.topic,
                "type": "content",
            }

            doc = Document(page_content=embed_text, metadata=metadata)

                                 
            await self.content_vectorstore.aadd_documents(documents=[doc], ids=[content_obj.id])
            return True
        except Exception as e:
            log.error(f"❌ 存储内容向量失败: {e}")
            return False

    async def add_comment_to_db(self, *, content_obj, comment_text: str, author_id: str, day_time: int) -> bool:
        """
        将评论向量化并存入 ChromaDB 的评论集合中。

        Args:
            content_obj: 评论所属的 Content 对象
            comment_text: 评论文本
            author_id: 评论者 persona_id
            day_time: 仿真时间步
        """
        try:
            if not comment_text:
                return False

                                                       
            topic = getattr(content_obj, "topic", "") or ""
            embed_text = f"主题: {topic}\n评论: {comment_text}"

                   
            comment_id = f"cmt_{content_obj.id}_{author_id}_{uuid.uuid4().hex[:8]}"
            metadata = {
                "comment_id": comment_id,
                "content_id": content_obj.id,
                "author_id": author_id,
                "day_time": day_time,
                "topic": topic,
                "type": "comment",
            }

            doc = Document(page_content=embed_text, metadata=metadata)
            await self.comments_vectorstore.aadd_documents(documents=[doc], ids=[comment_id])
            return True
        except Exception as e:
            log.error(f"❌ 存储评论向量失败: {e}")
            return False

    async def recommend_contents(self,
                                 persona,
                                 interest_content: str,
                                 current_day: int,
                                 limit: int = 5,
                                 recall_multiplier: int = 10) -> List[str]:
        """
        从全量内容池中，为指定 persona 生成当前时刻可见的内容集合。
        返回的不是全量客观信息，而是经过兴趣、社交关系、时间衰减、
        立场一致性和确认偏误共同作用后的曝光结果。

        公式：Score = MaxSim(User, Content) * lambda^(time_diff) + phi * I(social)
        这一条内容要么是用户极其感兴趣且刚发布的，要么是用户关注的好友发送的。

        Args:
            persona: 请求推荐的智能体对象 (包含 beliefs, description, social_relationships)
            interest_content: 用户想要获取的内容
            current_day: 当前仿真时间步 (用于计算时间衰减)
            limit: 最终推荐的内容数量
            recall_multiplier: 初筛倍率 (先从向量库召回 limit * N 条，再进行重排序)

        Returns:
            List[str]: 推荐的内容 ID 列表
        """
                                          
        if settings.platform.ablation_validation:
            try:
                raw_contents = self.content_vectorstore.get(include=["metadatas"], limit=max(limit * 20, 1000))
            except Exception as e:
                error_str = str(e)
                if "Nothing found on disk" in error_str or "segment" in error_str:
                    log.warning(f"⚠️ 内容库尚未就绪或为空 (Day {current_day})，跳过本次推送。")
                    return []
                log.error(f"❌ 普通推送获取内容失败: {e}")
                return []

            metadatas = raw_contents.get("metadatas") or []
            push_ids: List[str] = []
            seen_in_this_round = set()

            for meta in metadatas:
                if not meta:
                    continue
                content_id = meta.get("content_id")
                if not content_id:
                    continue
                                 
                if content_id in persona.viewed_content or content_id in seen_in_this_round:
                    continue
                push_ids.append(content_id)
                seen_in_this_round.add(content_id)
                if len(push_ids) >= limit:
                    break

            return push_ids

                             
        if interest_content:
            query_text = interest_content
        else:
            query_text = persona.description
        beliefs = getattr(persona, "beliefs", []) or []
                                            
        user_interest_text = f"{query_text}\n我的信念: {', '.join(beliefs)}"

                                                
        gamma = psycho_numeric_for_recall(getattr(persona, "gamma", None), default=0.5)
        stance_query_text = f"我的信念: {', '.join(beliefs)}"

                        
                                             
                          
        search_k = limit * recall_multiplier

        try:
                    
            candidates = await self.content_vectorstore.asimilarity_search_with_score(
                query=user_interest_text,
                k=search_k
            )
        except Exception as e:
                                     
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ 向量数据库尚未就绪或为空 (Day {current_day})，跳过本次搜索。")
                return []                   
            else:
                                     
                log.error(f"❌ 向量搜索发生未知异常: {e}")
                return []

                                                                              
                                                          
        stance_sim_by_content_id = {}
        try:
            belief_only_candidates = await self.content_vectorstore.asimilarity_search_with_score(
                query=stance_query_text,
                k=search_k
            )
            for doc, score in belief_only_candidates:
                meta = getattr(doc, "metadata", {}) or {}
                cid = meta.get("content_id")
                if not cid:
                    continue
                                            
                similarity_score = 1.0 - (float(score) / 2.0)
                similarity_score = max(0.0, min(1.0, similarity_score))
                stance_sim_by_content_id[cid] = similarity_score
        except Exception:
                                             
            stance_sim_by_content_id = {}

                           
                                                         
                                                          
        normalized_candidates = [(doc, 1.0 - (score / 2.0)) for doc, score in candidates]

                        
        ranked_candidates = []

              
        lambda_decay = 0.8                                 
        phi_social = 0.5             

        is_case_validation = settings.platform.case_validation

        for doc, similarity_score in normalized_candidates:
                   
            meta = doc.metadata
            content_id = meta.get("content_id")
            author_id = meta.get("author_id")
            topic = meta.get("topic", "")
            pub_time = meta.get("time", current_day)

                       
            if author_id == persona.agent_id:
                continue

                      
            if content_id in persona.viewed_content:
                continue

                       
                                 
            time_diff = max(0, current_day - pub_time)
            time_factor = math.pow(lambda_decay, time_diff)

                       
                            
            social_bonus = 0.0
            if author_id in persona.social_relationships:
                                               
                strength = persona.social_relationships[author_id]
                if strength > 0:
                    social_bonus = phi_social * strength             

                                                    
            viral_bonus = 0.0
            if is_case_validation:
                               
                if "NO AI" in topic.upper() or "PROTEST" in topic.upper():
                                                   
                    viral_bonus = 0.6
                                                          
                                               
                if "AI Generation" in doc.page_content:
                                              
                    viral_bonus = 0.3

                          
            stance_alignment = float(stance_sim_by_content_id.get(content_id, 0.0))
                                                         
                                          
            similarity_score = max(0.0, min(1.0, float(similarity_score)))
            final_sim = ((1.0 - gamma) * similarity_score) + (gamma * stance_alignment)

                     
                                                                      
            final_score = (final_sim * time_factor) + social_bonus + viral_bonus

            ranked_candidates.append({
                "content_id": content_id,
                "score": final_score,
                "debug_info": f"InterestSim:{similarity_score:.2f}, StanceSim:{stance_alignment:.2f}, CBmix:{final_sim:.2f}, Time:{time_diff}, Soc:{social_bonus:.2f}"
            })

                          
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        final_ids = [item["content_id"] for item in ranked_candidates[:limit]]

        if ranked_candidates:
            top_debug = ranked_candidates[0]
            log.info(
                f"推荐Top1给 {persona.name}: ID={top_debug['content_id']}, Score={top_debug['score']:.3f} ({top_debug['debug_info']})")

        return final_ids

    async def get_content_by_id(self, content_id: Optional[str], author_id: Optional[str]):
        content = None
        if content_id:
            content = await self.content_vectorstore.aget_by_ids(content_id)
        if author_id:
            content = self.content_vectorstore.get(
                where={"author_id": author_id},
                include=["metadatas", "documents"]
            )
        print(content)

    async def add_memory(self, persona_id: str, content: str, day_time: int, memory_type: MemoryType,
                         important_score: Optional[float]) -> str:
        """
        存储记忆
        :param persona_id:
        :param content:
        :param day_time:
        :param memory_type:
        :param important_score: 该条记忆的重要性分数
        :return:
        """

        final_metadata = {
            "persona_id": persona_id,
            "day_time": day_time,
            "memory_type": memory_type.value,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "important_score": important_score or 0
        }

                  
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"          

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

        return memory_id

    async def add_agent_think_memory(self, content: str, persona_id: str, day_time: int):
        """
        存储智能体思考记忆
        :param content:
        :param persona_id:
        :param day_time:
        :return:
        """
        final_metadata = {
            "persona_id": persona_id,
            "day_time": day_time,
            "memory_type": MemoryType.EXPERIENCE.value,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "important_score": 0
        }

                  
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"          

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

    async def recall_memories(self,
                              persona_id: str,
                              query: str = None,
                              top_k: int = 5,
                              memory_type: Optional[MemoryType] = None,
                              day_time: Optional[int] = None,
                              reflection: Optional[bool] = False,
                              gamma: float = 0.5,
                              ) -> List[Document]:
        """
        根据语义相似性，为特定智能体回忆最相关的记忆，并可选地模拟基于时间的记忆衰减。

        Args:
            persona_id (str):
            query (str):
            top_k (int):
            memory_type (Optional[MemoryType]):
            day_time (Optional[int]):
            reflection: False 不是反思的时候查的；True  是反思的时候查的
            gamma: 确认偏误系数 (0.0-1.0)，用于记忆过滤
        Returns:
            List[Document]: ...
        """

                            
        final_filter_conditions = {'persona_id': persona_id}
        if memory_type:
            final_filter_conditions['memory_type'] = memory_type.value
        if day_time is not None:                                      
            final_filter_conditions['day_time'] = day_time

                                        
        if not final_filter_conditions:
            chroma_filter = None
        elif len(final_filter_conditions) == 1:
            key, value = list(final_filter_conditions.items())[0]
            chroma_filter = {key: {"$eq": value}}                
        else:
            chroma_filter = {
                "$and": [
                    {key: {"$eq": value}} for key, value in final_filter_conditions.items()
                ]
            }
                                                
                                      
        gamma = max(0.0, min(1.0, float(gamma)))

        try:
            if reflection:
                                                   
                results = self.vectorstore.get(where=chroma_filter, limit=top_k)
            else:
                                                     
                if settings.platform.ablation_validation:
                    if not query:
                        return await self.vectorstore.asimilarity_search(
                            query="",
                            k=top_k,
                            filter=chroma_filter,
                        )
                    return await self.vectorstore.asimilarity_search(
                        query=query,
                        k=top_k,
                        filter=chroma_filter,
                    )

                                                  
                pool_k = max(top_k * 5, 20)
                if not query:
                                                        
                    results = await self.vectorstore.asimilarity_search(
                        query="",
                        k=top_k,
                        filter=chroma_filter,
                    )
                    return results

                candidates_with_scores = await self.vectorstore.asimilarity_search_with_score(
                    query=query, k=pool_k, filter=chroma_filter
                )

                                                                                  
                scored_candidates = []
                for doc, distance in candidates_with_scores:
                    similarity_score = 1.0 - (float(distance) / 2.0)
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    scored_candidates.append((doc, similarity_score))

                if not scored_candidates:
                    return []

                                            
                import hashlib
                import random

                mtype_val = memory_type.value if memory_type else ""
                seed_key = f"{persona_id}|{query}|{top_k}|{day_time}|{mtype_val}|{gamma}"
                seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
                rng = random.Random(seed)

                                                 
                exponent = 1.0 + 4.0 * gamma                                  

                weights = []
                for _, similarity_score in scored_candidates:
                    weights.append(max(1e-12, similarity_score ** exponent))

                                    
                remaining = list(range(len(scored_candidates)))
                selected_indices = []
                for _ in range(min(top_k, len(scored_candidates))):
                    total_w = sum(weights[i] for i in remaining)
                    if total_w <= 0:
                                            
                        selected_indices = remaining[:top_k]
                        break
                    r = rng.random() * total_w
                    cum = 0.0
                    chosen = remaining[-1]
                    for i in remaining:
                        cum += weights[i]
                        if cum >= r:
                            chosen = i
                            break
                    selected_indices.append(chosen)
                    remaining.remove(chosen)

                                        
                selected = [(scored_candidates[i][0], scored_candidates[i][1]) for i in selected_indices]
                selected.sort(key=lambda x: x[1], reverse=True)
                results = [doc for doc, _ in selected]
        except Exception as e:
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ 记忆库尚未初始化或为空 (查询: '{query}')，返回空结果。这在第一天是正常的。")
                return []
            log.error(f"❌ 在向量搜索期间发生未预期错误。查询: '{query}'. 错误: {e}")
            traceback.TracebackException.from_exception(e).print()
            return []
        return results

    def export_day_to_json(self, environment, day_number: int = 1,
                           additional_str: str = "", simple: bool = False):
        """
        将在指定日期添加到向量数据库的所有记忆导出到一个JSON文件中。

        这个函数应该在每天的仿真循环结束后调用，用于数据备份和离线分析。
        **前提**: 添加记忆时，元数据中必须包含 'day' 字段。

        Args:
            :param output_directory:
            :param day_number:
            :param simple:
            :param additional_str:
            :param environment:
            day_number (int): 需要导出记忆的日期（天数）。
            output_directory (str): 存放导出JSON文件的目录路径。
        """
        log.info(f"📄 开始导出第 {day_number} 天的数据...")
        if simple:
            output_directory = settings.file_load_path.base_store_file + "/low"
        else:
            output_directory = settings.file_load_path.base_store_file + "/high"
                  
        if additional_str:
            output_directory = output_directory + f"/{additional_str}/day_time_{day_number}"
        else:
            output_directory = output_directory + f"/day_time_{day_number}"
        os.makedirs(output_directory, exist_ok=True)

              
        output_contents = output_directory + f"/output_contents.json"
        output_personas = output_directory + f"/output_personas.json"
        output_platform = output_directory + f"/output_platform.json"
        output_policy = output_directory + f"/output_policy.json"
        output_system_kpi = output_directory + f"/output_system_kpi.json"
        output_memories = output_directory + f"/output_memories.json"

                
        contents = [content.model_dump() for content in environment.contents.get_all_contents_dict()]
                
        personas = [v.model_dump() for k, v in environment.personas.items()]
                
        policy = {
            "e_edu": environment.policy.e_edu,
            "ai_threshold": environment.policy.ai_threshold,
            "f_penalty": environment.policy.f_penalty,
        }
                
        platforms = {
            "w": environment.platform.w,
            "mu": environment.platform.mu,
            "eta": environment.platform.eta,
            "tau_tech": environment.platform.tau_tech,
            "steep": environment.platform.steep,
            "policy": policy,
            "today_theta": environment.platform.theta,

            "platform_theta_change": environment.platform.platform_theta_change,
            "public_loss_data": environment.platform.public_loss_data,
            "public_change_role_data": environment.platform.public_change_role_data,
            "kpi_change_data": environment.platform.kpi_change_data,
            "fn_data": environment.platform.fn_data,
            "fp_data": environment.platform.fp_data,
                                                                                                  

            "creator_data": environment.platform.creator_data,
                                                                            

        }
                   
        system_kpi = {
            "safety": environment.system_kpi.safety,
            "satisfaction": environment.system_kpi.satisfaction,
            "creativity": environment.system_kpi.creativity,
            "theta": environment.system_kpi.theta,
        }

                                        
        try:
            results = self.vectorstore.get(
                where={
                    "$and": [
                        {"day_time": {"$eq": day_number}},
                        {"memory_type": {"$eq": MemoryType.EXPERIENCE.value}}
                    ]
                }
            )
        except Exception as e:
            log.error(f"从ChromaDB获取第 {day_number} 天的数据时出错: {e}")
            results = {"ids": [], "documents": [], "metadatas": []}
                           
        daily_memories_list = []
        if results['ids']:
            for mem_id, content, metadata in zip(results['ids'], results['documents'], results['metadatas']):
                memory_obj = {
                    "memory_id": mem_id,
                    "content": content,
                    "metadata": metadata
                }
                daily_memories_list.append(memory_obj)

                   
        results_agent_think_memory = {}
        for persona in environment.personas.values():
            try:
                agent_think_memory = self.vectorstore.get(where={
                    "$and": [
                        {"memory_type": {"$eq": "agent_think"}},                  
                        {"persona_id": {"$eq": persona.agent_id}},
                    ]
                })
            except Exception as e:
                log.error(f"获取智能体 {persona.agent_id} 的思考记忆时出错: {e}")
                agent_think_memory = {"ids": [], "documents": [], "metadatas": []}
                            
            if agent_think_memory['ids']:
                res = [persona.model_dump(exclude=["viewed_content", "reacted_content"])]

                metadatas = agent_think_memory.get('metadatas', [])
                if metadatas is None: metadatas = []

                for mem_id, content, metadata in zip(agent_think_memory['ids'],
                                                     agent_think_memory['documents'],
                                                     metadatas):
                    memory_obj = {
                        "memory_id": mem_id,
                        "content": content,
                        "metadata": metadata
                    }
                    res.append(memory_obj)

                        
                results_agent_think_memory[persona.agent_id] = res

                     
        try:
            with open(output_contents, 'w', encoding='utf-8') as f:
                json.dump(contents, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 内容 已成功导出到: {output_contents}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_contents} 时出错: {e}")

                       
        try:
            with open(output_personas, 'w', encoding='utf-8') as f:
                json.dump(personas, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 人物数据 已成功导出到: {output_personas}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_personas} 时出错: {e}")

                       
        try:
            with open(output_platform, 'w', encoding='utf-8') as f:
                json.dump(platforms, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 平台数据 已成功导出到: {output_platform}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_platform} 时出错: {e}")

                       
        try:
            with open(output_policy, 'w', encoding='utf-8') as f:
                json.dump(policy, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 政策参数 已成功导出到: {output_policy}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_policy} 时出错: {e}")

                        
        try:
            with open(output_system_kpi, 'w', encoding='utf-8') as f:
                json.dump(system_kpi, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 系统KPI 已成功导出到: {output_system_kpi}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_system_kpi} 时出错: {e}")

                     
        try:
            with open(output_memories, 'w', encoding='utf-8') as f:
                json.dump(daily_memories_list, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 {len(daily_memories_list)} 条记忆已成功导出到: {output_memories}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_memories} 时出错: {e}")


                    
def _flatten_metadata_fully(metadata: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    一个辅助函数，递归地将嵌套字典扁平化，通过连接键名。
    例如: {"a": {"b": 1}} -> {"a_b": 1}
    """
    items = []
    for k, v in metadata.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_metadata_fully(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
