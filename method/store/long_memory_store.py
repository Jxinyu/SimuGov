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

from method.utils.calculation_token_nums import calculate_token_nums, calculate_token_nums_simplt
from config import settings
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

    # BELIEF = 'belief'
    BELIEF = 'experience'
    """
    **类型：信念 **
    """

    # SUMMARIZE = 'summarize'
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
                 content_collection_name: str = "platform_contents",  # 内容专用集合
                 persist_directory: str = None,
                 embedding_function: Embeddings = DEFAULT_EMBEDDING_FUNCTION):
        log.info(f"正在初始化 MemoryStore...")
        if persist_directory is None:
            # 1. 获取当前时间
            current_now = datetime.now()
            current_date = current_now.strftime("%Y-%m-%d")
            current_time = current_now.strftime("%H_%M_%S")

            # 2. 生成一个随机后缀 (8位 UUID)，确保即使同一秒启动，路径也不一样
            unique_suffix = uuid.uuid4().hex[:8]

            # 3. 拼接完整路径
            # 格式示例: .../2025-12-27/21_30_05_a1b2c3d4_db
            persist_directory = (
                f"{settings.file_load_path.chroma_db_file}/"
                f"{current_date}/{current_time}_{unique_suffix}_db"
            )

            # 记录这次运行使用的实际路径，方便调试
        self.persist_directory = persist_directory
        log.info(f"📂 本次运行的独立数据库路径: {self.persist_directory}")

        # 1. 智能体记忆存储
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

        # 2. 平台内容存储
        # 使用独立的 collection，防止搜索新闻时搜到别人的私密记忆
        self.content_vectorstore = Chroma(
            collection_name=content_collection_name,
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
            # 构造用于 Embedding 的文本
            # 包含主题和详情，有助于计算语义相似度
            embed_text = f"主题: {content_obj.topic}\n内容: {content_obj.content_detail}"

            # 构造元数据
            metadata = {
                "content_id": content_obj.id,
                "author_id": content_obj.author_id,
                "time": content_obj.time,  # 用于时间衰减计算
                "topic": content_obj.topic,
                "type": "content",
            }

            doc = Document(page_content=embed_text, metadata=metadata)

            # 使用内容专用的 vectorstore
            await self.content_vectorstore.aadd_documents(documents=[doc], ids=[content_obj.id])
            log.debug(f"✅ 内容 '{content_obj.id}' 已向量化存入数据库。")
            return True
        except Exception as e:
            log.error(f"❌ 存储内容向量失败: {e}")
            return False

    async def recommend_contents(self,
                                 persona,
                                 interest_content: str,
                                 current_day: int,
                                 limit: int = 5,
                                 recall_multiplier: int = 10) -> List[str]:
        """
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
        # --- 1. 构造用户兴趣向量 ---
        if interest_content:
            query_text = interest_content
        else:
            query_text = persona.description
        # 对应公式中的 τ^i (Agent的过去观点/兴趣)  “兴趣向量”
        user_interest_text = f"{query_text}\n我的信念: {', '.join(persona.beliefs)}"

        # --- 2. 向量召回---
        # 先召回较多候选集 (例如需要5条，先查50条)，然后在内存中做精细排序
        # 对应公式中的 MaxSim 部分
        search_k = limit * recall_multiplier

        try:
            # 尝试进行搜索
            candidates = await self.content_vectorstore.asimilarity_search_with_relevance_scores(
                query=user_interest_text,
                k=search_k
            )
        except Exception as e:
            # 捕获 ChromaDB 的空库/索引未就绪错误
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ 向量数据库尚未就绪或为空 (Day {current_day})，跳过本次搜索。")
                return []  # 优雅返回空列表，表示“没内容”
            else:
                # 如果是其他严重错误，打印日志但尽量不崩
                log.error(f"❌ 向量搜索发生未知异常: {e}")
                return []

        # --- 3. 重排序 ---
        ranked_candidates = []

        # 参数设置
        lambda_decay = 0.8  # 时间衰减系数 (0~1)，越小衰减越快。表示新内容的权重。
        phi_social = 0.5  # 社交关系加分项权重

        is_case_validation = settings.platform.case_validation

        for doc, similarity_score in candidates:
            # 获取元数据
            meta = doc.metadata
            content_id = meta.get("content_id")
            author_id = meta.get("author_id")
            topic = meta.get("topic", "")
            pub_time = meta.get("time", current_day)

            # 过滤掉自己发的内容
            if author_id == persona.agent_id:
                continue

            # 过滤掉已经看过的
            if content_id in persona.viewed_content:
                continue

            # A. 计算时间衰减
            # time_diff = d - d_k
            time_diff = max(0, current_day - pub_time)
            time_factor = math.pow(lambda_decay, time_diff)

            # B. 计算社交加权
            # phi * I (是否关注)
            social_bonus = 0.0
            if author_id in persona.social_relationships:
                # 获取关注强度 (假设 range -1.0 to 1.0)
                strength = persona.social_relationships[author_id]
                if strength > 0:
                    social_bonus = phi_social * strength  # 关系越好，加分越多

            # C. [关键修改] 热门事件/抗议内容的强制提权 (Viral Boost)
            viral_bonus = 0.0
            if is_case_validation:
                # 识别抗议内容或AI刷屏内容
                if "NO AI" in topic.upper() or "PROTEST" in topic.upper():
                    # 给予巨大的加分，模拟“全站热搜”，确保能冲破个人的兴趣茧房
                    viral_bonus = 0.6
                # 识别是否是AI刷屏内容 (通过 content_detail 或 注入时的特征)
                # 这里假设我们在注入时，Topic设置为了特定的AI相关词汇
                if "AI Generation" in doc.page_content:
                    # AI 内容也有一定的热度，确保被看到从而引发愤怒
                    viral_bonus = 0.3

            # C. 综合打分
            # Score = Sim * TimeDecay + SocialBonus + viral_bonus
            final_score = (similarity_score * time_factor) + social_bonus + viral_bonus

            ranked_candidates.append({
                "content_id": content_id,
                "score": final_score,
                "debug_info": f"Sim:{similarity_score:.2f}, Time:{time_diff}, Soc:{social_bonus:.2f}"
            })

        # --- 4. 排序并截断 ---
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        final_ids = [item["content_id"] for item in ranked_candidates[:limit]]

        if ranked_candidates:
            top_debug = ranked_candidates[0]
            log.info(
                f"推荐Top1给 {persona.name}: ID={top_debug['content_id']}, Score={top_debug['score']:.3f} ({top_debug['debug_info']})")

        return final_ids

    async def get_content_by_id(self, content_id: str | None, author_id: str | None):
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
                         important_score: float | None) -> str:
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

        # 自动扁平化元数据
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"  # 生成唯一ID

        log.info(f"➕ 正在为 '{persona_id}' 添加记忆 (ID: {memory_id}): '{content[:30]}...'")

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

        log.debug(f"✅ 记忆 '{memory_id}' 已存入，元数据: {flattened_metadata}")

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
            "memory_type": "agent_think",
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "important_score": 0
        }

        # 自动扁平化元数据
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"  # 生成唯一ID

        log.info(f"➕ 正在为 '{persona_id}' 添加记忆 (ID: {memory_id}): '{content[:30]}...'")

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

        log.debug(f"✅ 记忆 '{memory_id}' 已存入，元数据: {flattened_metadata}")

    async def recall_memories(self,
                              persona_id: str,
                              query: str,
                              top_k: int = 5,
                              memory_type: Optional[MemoryType] = None) -> List[Document]:
        """
        根据语义相似性，为特定智能体回忆最相关的记忆，并可选地模拟基于时间的记忆衰减。

        Args:
            persona_id (str):
            query (str):
            top_k (int):
            memory_type (Optional[MemoryType]):

        Returns:
            List[Document]: ...
        """

        # --- 1. 数据过滤器 ---
        final_filter_conditions = {'persona_id': persona_id}
        if memory_type:
            final_filter_conditions['memory_type'] = memory_type.value

        if not final_filter_conditions:
            chroma_filter = None
        elif len(final_filter_conditions) == 1:
            # 如果只有一个条件，直接使用
            key, value = list(final_filter_conditions.items())[0]
            chroma_filter = {key: {"$eq": value}}
        else:
            # 如果有多个条件，才使用 $and 操作符
            chroma_filter = {
                "$and": [
                    {key: {"$eq": value}} for key, value in final_filter_conditions.items()
                ]
            }

        log.info(f"🔍 正在为 '{persona_id}' 查询记忆...")
        try:
            # --- 2. 执行查询 ---
            results = await self.vectorstore.asimilarity_search(query=query, k=top_k, filter=chroma_filter)
        except Exception as e:
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ 记忆库尚未初始化或为空 (查询: '{query}')，返回空结果。这在第一天是正常的。")
                return []
            log.error(f"❌ 在向量搜索期间发生未预期错误。查询: '{query}'. 错误: {e}")
            traceback.TracebackException.from_exception(e).print()
            return []

        log.info(f"✅  - 检索到 {len(results)} 条相关记忆。")
        return results

    def export_day_to_json(self, environment, day_number: int = 1,
                           output_directory: str = settings.file_load_path.daily_memory_exports_file,
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

        # 确保输出目录存在
        if additional_str:
            output_directory = output_directory + f"/{environment.log_output_dir}/{additional_str}/day_time_{day_number}"
        else:
            output_directory = output_directory + f"/{environment.log_output_dir}/day_time_{day_number}"
        os.makedirs(output_directory, exist_ok=True)

        # 输出目录
        output_contents = output_directory + f"/output_contents.json"
        output_personas = output_directory + f"/output_personas.json"
        output_platform = output_directory + f"/output_platform.json"
        output_policy = output_directory + f"/output_policy.json"
        output_system_kpi = output_directory + f"/output_system_kpi.json"
        output_memories = output_directory + f"/output_memories.json"
        output_agent_think_memories = output_directory + f"/output_agent_think_memories.json"

        # 构造内容数据
        contents = [content.model_dump() for content in environment.contents.get_all_contents_dict()]
        # 构造用户数据
        personas = [v.model_dump() for k, v in environment.personas.items()]
        # 构造策略数据
        policy = {
            "e_edu": environment.policy.e_edu,
            "ai_threshold": environment.policy.ai_threshold,
            "f_penalty": environment.policy.f_penalty,
        }
        # 构造平台数据
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
            # "cost_calculation_details_data": environment.platform.cost_calculation_details_data,

            "creator_data": environment.platform.creator_data,
            # "personas_call_tool": environment.platform.personas_call_tool,


        }
        # 构造系统KPI数据
        system_kpi = {
            "safety": environment.system_kpi.safety,
            "satisfaction": environment.system_kpi.satisfaction,
            "creativity": environment.system_kpi.creativity,
            "all_tokens": calculate_token_nums_simplt() if simple else calculate_token_nums(),
            "theta": environment.system_kpi.theta,
        }

        # 使用 get 方法和 where 过滤器来获取当天的所有记忆
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
        # 将返回的数据构造成一个更清晰的列表
        daily_memories_list = []
        if results['ids']:
            for mem_id, content, metadata in zip(results['ids'], results['documents'], results['metadatas']):
                memory_obj = {
                    "memory_id": mem_id,
                    "content": content,
                    "metadata": metadata
                }
                daily_memories_list.append(memory_obj)

        # 获取智能体思考记忆
        results_agent_think_memory = {}
        for persona in environment.personas.values():
            try:
                agent_think_memory = self.vectorstore.get(where={
                    "$and": [
                        {"memory_type": {"$eq": "agent_think"}},  # 确保存储时也是用的这个字符串
                        {"persona_id": {"$eq": persona.agent_id}},
                    ]
                })
            except Exception as e:
                log.error(f"获取智能体 {persona.agent_id} 的思考记忆时出错: {e}")
                agent_think_memory = {"ids": [], "documents": [], "metadatas": []}
            # 只有当查找到数据时才进行处理
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

                # 存入结果字典
                results_agent_think_memory[persona.agent_id] = res

        # 内容 写入JSON文件
        try:
            with open(output_contents, 'w', encoding='utf-8') as f:
                json.dump(contents, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 内容 已成功导出到: {output_contents}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_contents} 时出错: {e}")

        # 人物数据 写入JSON文件
        try:
            with open(output_personas, 'w', encoding='utf-8') as f:
                json.dump(personas, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 人物数据 已成功导出到: {output_personas}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_personas} 时出错: {e}")

        # 平台数据 写入JSON文件
        try:
            with open(output_platform, 'w', encoding='utf-8') as f:
                json.dump(platforms, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 平台数据 已成功导出到: {output_platform}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_platform} 时出错: {e}")

        # 政策参数 写入JSON文件
        try:
            with open(output_policy, 'w', encoding='utf-8') as f:
                json.dump(policy, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 政策参数 已成功导出到: {output_policy}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_policy} 时出错: {e}")

        # 系统KPI 写入JSON文件
        try:
            with open(output_system_kpi, 'w', encoding='utf-8') as f:
                json.dump(system_kpi, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 系统KPI 已成功导出到: {output_system_kpi}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_system_kpi} 时出错: {e}")

        # 记忆 写入JSON文件
        try:
            with open(output_memories, 'w', encoding='utf-8') as f:
                json.dump(daily_memories_list, f, ensure_ascii=False, indent=4)
            log.info(f"✅ 第 {day_number} 天的 {len(daily_memories_list)} 条记忆已成功导出到: {output_memories}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_memories} 时出错: {e}")

        try:
            with open(output_agent_think_memories, 'w', encoding='utf-8') as f:
                json.dump(results_agent_think_memory, f, ensure_ascii=False, indent=4)
            log.info(
                f"✅ 第 {day_number} 天的 {len(results_agent_think_memory)} 条思考记忆已成功导出到: {output_agent_think_memories}")
        except Exception as e:
            log.error(f"写入JSON文件 {output_agent_think_memories} 时出错: {e}")


# --- 推荐的扁平化辅助函数 ---
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
