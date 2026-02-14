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
    Defines the three core types of agent memory, representing the cognitive hierarchy from specific experiences to abstract beliefs.
    """
    EXPERIENCE = 'experience'
    """
    ** Type: Experience **
    """

    # BELIEF = 'belief'
    BELIEF = 'experience'
    """
    ** Type: Belief **
    """

    # SUMMARIZE = 'summarize'
    SUMMARIZE = 'experience'
    """
    ** Type: Summarize **
    """


class MemoryStore:
    """
    A vector store for managing agent memories.
    """

    def __init__(self,
                 collection_name: str = "simulation_memories",
                 content_collection_name: str = "platform_contents",  # Collection dedicated to content
                 persist_directory: str = None,
                 embedding_function: Embeddings = DEFAULT_EMBEDDING_FUNCTION):
        log.info(f"Initializing MemoryStore...")
        if persist_directory is None:
            # 1. Get current time
            current_now = datetime.now()
            current_date = current_now.strftime("%Y-%m-%d")
            current_time = current_now.strftime("%H_%M_%S")

            # 2. Generate a random suffix (8-character UUID) to ensure unique paths even if started in the same second
            unique_suffix = uuid.uuid4().hex[:8]

            # 3. Concatenate the full path
            # Example: .../2025-12-27/21_30_05_a1b2c3d4_db
            persist_directory = (
                f"{settings.file_load_path.chroma_db_file}/"
                f"{current_date}/{current_time}_{unique_suffix}_db"
            )

            # Record the actual path used for this run to facilitate debugging
        self.persist_directory = persist_directory
        log.info(f"📂 Independent database path for this run: {self.persist_directory}")

        # 1. Agent memory storage
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

        # 2. Platform content storage
        # Use a separate collection to prevent private memories from appearing in news searches
        self.content_vectorstore = Chroma(
            collection_name=content_collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )

        log.info("✅ MemoryStore initialization successful (includes memory and content stores).")

    async def add_content_to_db(self, content_obj) -> bool:
        """
        Vectorize content and store it in the ChromaDB content collection.
        Args:
            content_obj: Content object
        """
        try:
            # Construct text for Embedding
            # Includes topic and details to help calculate semantic similarity
            embed_text = f"Topic: {content_obj.topic}\nContent: {content_obj.content_detail}"

            # Construct metadata
            metadata = {
                "content_id": content_obj.id,
                "author_id": content_obj.author_id,
                "time": content_obj.time,  # Used for time decay calculation
                "topic": content_obj.topic,
                "type": "content",
            }

            doc = Document(page_content=embed_text, metadata=metadata)

            # Use the content-specific vectorstore
            await self.content_vectorstore.aadd_documents(documents=[doc], ids=[content_obj.id])
            log.debug(f"✅ Content '{content_obj.id}' has been vectorized and stored in the database.")
            return True
        except Exception as e:
            log.error(f"❌ Failed to store content vector: {e}")
            return False

    async def recommend_contents(self,
                                 persona,
                                 interest_content: str,
                                 current_day: int,
                                 limit: int = 5,
                                 recall_multiplier: int = 10) -> List[str]:
        """
        Formula: Score = MaxSim(User, Content) * lambda^(time_diff) + phi * I(social)
        This content is either something the user is extremely interested in and just released,
        or sent by a followed friend.

        Args:
            persona: Agent object requesting recommendation (contains beliefs, description, social_relationships)
            interest_content: Content the user wants to get
            current_day: Current simulation time step (used for time decay calculation)
            limit: Number of contents finally recommended
            recall_multiplier: Recall multiplier (recall limit * N items from the vector store first, then re-rank)

        Returns:
            List[str]: List of recommended content IDs
        """
        # --- 1. Construct user interest vector ---
        if interest_content:
            query_text = interest_content
        else:
            query_text = persona.description
        # Corresponds to τ^i in the formula (Agent's past views/interests) "Interest Vector"
        user_interest_text = f"{query_text}\nMy Beliefs: {', '.join(persona.beliefs)}"

        # --- 2. Vector Recall ---
        # Recall a larger candidate set first (e.g., if 5 are needed, search 50), then do fine-grained ranking in memory
        # Corresponds to the MaxSim part in the formula
        search_k = limit * recall_multiplier

        try:
            # Attempt search
            candidates = await self.content_vectorstore.asimilarity_search_with_relevance_scores(
                query=user_interest_text,
                k=search_k
            )
        except Exception as e:
            # Catch ChromaDB empty database/index not ready error
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ Vector database is not ready or is empty (Day {current_day}), skipping this search.")
                return []  # Gracefully return empty list
            else:
                # For other serious errors, log them but try not to crash
                log.error(f"❌ Unknown exception occurred during vector search: {e}")
                return []

        # --- 3. Re-ranking ---
        ranked_candidates = []

        # Parameter settings
        lambda_decay = 0.8  # Time decay coefficient (0~1), smaller means faster decay. Represents weight of new content.
        phi_social = 0.5  # Weight for social relationship bonus

        is_case_validation = settings.platform.case_validation

        for doc, similarity_score in candidates:
            # Get metadata
            meta = doc.metadata
            content_id = meta.get("content_id")
            author_id = meta.get("author_id")
            topic = meta.get("topic", "")
            pub_time = meta.get("time", current_day)

            # Filter out own content
            if author_id == persona.agent_id:
                continue

            # Filter out viewed content
            if content_id in persona.viewed_content:
                continue

            # A. Calculate time decay
            # time_diff = d - d_k
            time_diff = max(0, current_day - pub_time)
            time_factor = math.pow(lambda_decay, time_diff)

            # B. Calculate social weighting
            # phi * I (whether following)
            social_bonus = 0.0
            if author_id in persona.social_relationships:
                # Get following strength (assume range -1.0 to 1.0)
                strength = persona.social_relationships[author_id]
                if strength > 0:
                    social_bonus = phi_social * strength  # Better relationship, more points

            # C. [Key Modification] Viral boost for hot events/protest content
            viral_bonus = 0.0
            if is_case_validation:
                # Identify protest content or AI spam content
                if "NO AI" in topic.upper() or "PROTEST" in topic.upper():
                    # Give a huge bonus to simulate "site-wide trending," ensuring it breaks through personal interest bubbles
                    viral_bonus = 0.6
                # Identify if it's AI spam content (via content_detail or injection features)
                if "AI Generation" in doc.page_content:
                    # AI content also gets some heat to ensure it's seen to trigger anger
                    viral_bonus = 0.3

            # C. Comprehensive scoring
            # Score = Sim * TimeDecay + SocialBonus + viral_bonus
            final_score = (similarity_score * time_factor) + social_bonus + viral_bonus

            ranked_candidates.append({
                "content_id": content_id,
                "score": final_score,
                "debug_info": f"Sim:{similarity_score:.2f}, Time:{time_diff}, Soc:{social_bonus:.2f}"
            })

        # --- 4. Sort and truncate ---
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        final_ids = [item["content_id"] for item in ranked_candidates[:limit]]

        if ranked_candidates:
            top_debug = ranked_candidates[0]
            log.info(
                f"Recommended Top 1 to {persona.name}: ID={top_debug['content_id']}, Score={top_debug['score']:.3f} ({top_debug['debug_info']})")

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
        Store memory
        :param persona_id:
        :param content:
        :param day_time:
        :param memory_type:
        :param important_score: Importance score of this memory
        :return:
        """

        final_metadata = {
            "persona_id": persona_id,
            "day_time": day_time,
            "memory_type": memory_type.value,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "important_score": important_score or 0
        }

        # Automatically flatten metadata
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"  # Generate unique ID

        log.info(f"➕ Adding memory for '{persona_id}' (ID: {memory_id}): '{content[:30]}...'")

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

        log.debug(f"✅ Memory '{memory_id}' stored with metadata: {flattened_metadata}")

        return memory_id

    async def add_agent_think_memory(self, content: str, persona_id: str, day_time: int):
        """
        Store agent thought memory
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

        # Automatically flatten metadata
        flattened_metadata = _flatten_metadata_fully(final_metadata)

        memory_id = f"mem_{persona_id}_{uuid.uuid4().hex[:8]}"  # Generate unique ID

        log.info(f"➕ Adding memory for '{persona_id}' (ID: {memory_id}): '{content[:30]}...'")

        doc = Document(page_content=content, metadata=flattened_metadata)
        await self.vectorstore.aadd_documents(documents=[doc], ids=[memory_id])

        log.debug(f"✅ Memory '{memory_id}' stored with metadata: {flattened_metadata}")

    async def recall_memories(self,
                              persona_id: str,
                              query: str,
                              top_k: int = 5,
                              memory_type: Optional[MemoryType] = None) -> List[Document]:
        """
        Recall the most relevant memories for a specific agent based on semantic similarity,
        optionally simulating time-based memory decay.

        Args:
            persona_id (str):
            query (str):
            top_k (int):
            memory_type (Optional[MemoryType]):

        Returns:
            List[Document]: ...
        """

        # --- 1. Data Filter ---
        final_filter_conditions = {'persona_id': persona_id}
        if memory_type:
            final_filter_conditions['memory_type'] = memory_type.value

        if not final_filter_conditions:
            chroma_filter = None
        elif len(final_filter_conditions) == 1:
            # If only one condition, use it directly
            key, value = list(final_filter_conditions.items())[0]
            chroma_filter = {key: {"$eq": value}}
        else:
            # Use $and operator only if multiple conditions exist
            chroma_filter = {
                "$and": [
                    {key: {"$eq": value}} for key, value in final_filter_conditions.items()
                ]
            }

        log.info(f"🔍 Querying memories for '{persona_id}'...")
        try:
            # --- 2. Execute Query ---
            results = await self.vectorstore.asimilarity_search(query=query, k=top_k, filter=chroma_filter)
        except Exception as e:
            error_str = str(e)
            if "Nothing found on disk" in error_str or "segment" in error_str:
                log.warning(f"⚠️ Memory store is not initialized or is empty (Query: '{query}'), returning empty results. This is normal on day one.")
                return []
            log.error(f"❌ Unexpected error occurred during vector search. Query: '{query}'. Error: {e}")
            traceback.TracebackException.from_exception(e).print()
            return []

        log.info(f"✅ - Retrieved {len(results)} relevant memories.")
        return results

    def export_day_to_json(self, environment, day_number: int = 1,
                           output_directory: str = settings.file_load_path.daily_memory_exports_file,
                           additional_str: str = "", simple: bool = False):
        """
        Export all memories added to the vector database on a specified date to a JSON file.

        This function should be called after the simulation loop each day for data backup and offline analysis.
        **Prerequisite**: Metadata must include the 'day' field when adding memories.

        Args:
            :param output_directory:
            :param day_number:
            :param simple:
            :param additional_str:
            :param environment:
            day_number (int): The date (day number) to export memories for.
            output_directory (str): The directory path to store exported JSON files.
        """
        log.info(f"📄 Starting export of Day {day_number} data...")

        # Ensure output directory exists
        if additional_str:
            output_directory = output_directory + f"/{environment.log_output_dir}/{additional_str}/day_time_{day_number}"
        else:
            output_directory = output_directory + f"/{environment.log_output_dir}/day_time_{day_number}"
        os.makedirs(output_directory, exist_ok=True)

        # Output paths
        output_contents = output_directory + f"/output_contents.json"
        output_personas = output_directory + f"/output_personas.json"
        output_platform = output_directory + f"/output_platform.json"
        output_policy = output_directory + f"/output_policy.json"
        output_system_kpi = output_directory + f"/output_system_kpi.json"
        output_memories = output_directory + f"/output_memories.json"
        output_agent_think_memories = output_directory + f"/output_agent_think_memories.json"

        # Construct content data
        contents = [content.model_dump() for content in environment.contents.get_all_contents_dict()]
        # Construct user data
        personas = [v.model_dump() for k, v in environment.personas.items()]
        # Construct policy data
        policy = {
            "e_edu": environment.policy.e_edu,
            "ai_threshold": environment.policy.ai_threshold,
            "f_penalty": environment.policy.f_penalty,
        }
        # Construct platform data
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
        # Construct system KPI data
        system_kpi = {
            "safety": environment.system_kpi.safety,
            "satisfaction": environment.system_kpi.satisfaction,
            "creativity": environment.system_kpi.creativity,
            "all_tokens": calculate_token_nums_simplt() if simple else calculate_token_nums(),
            "theta": environment.system_kpi.theta,
        }

        # Use get method and where filter to retrieve all memories of the day
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
            log.error(f"Error retrieving Day {day_number} data from ChromaDB: {e}")
            results = {"ids": [], "documents": [], "metadatas": []}
        # Construct a clearer list from returned data
        daily_memories_list = []
        if results['ids']:
            for mem_id, content, metadata in zip(results['ids'], results['documents'], results['metadatas']):
                memory_obj = {
                    "memory_id": mem_id,
                    "content": content,
                    "metadata": metadata
                }
                daily_memories_list.append(memory_obj)

        # Retrieve agent thought memories
        results_agent_think_memory = {}
        for persona in environment.personas.values():
            try:
                agent_think_memory = self.vectorstore.get(where={
                    "$and": [
                        {"memory_type": {"$eq": "agent_think"}},  # Ensure string matches storage
                        {"persona_id": {"$eq": persona.agent_id}},
                    ]
                })
            except Exception as e:
                log.error(f"Error retrieving thought memory for agent {persona.agent_id}: {e}")
                agent_think_memory = {"ids": [], "documents": [], "metadatas": []}
            # Process only if data is found
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

                # Store into result dictionary
                results_agent_think_memory[persona.agent_id] = res

        # Write Content to JSON file
        try:
            with open(output_contents, 'w', encoding='utf-8') as f:
                json.dump(contents, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number} Content exported successfully to: {output_contents}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_contents}: {e}")

        # Write Persona data to JSON file
        try:
            with open(output_personas, 'w', encoding='utf-8') as f:
                json.dump(personas, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number} Persona data exported successfully to: {output_personas}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_personas}: {e}")

        # Write Platform data to JSON file
        try:
            with open(output_platform, 'w', encoding='utf-8') as f:
                json.dump(platforms, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number} Platform data exported successfully to: {output_platform}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_platform}: {e}")

        # Write Policy parameters to JSON file
        try:
            with open(output_policy, 'w', encoding='utf-8') as f:
                json.dump(policy, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number} Policy parameters exported successfully to: {output_policy}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_policy}: {e}")

        # Write System KPI to JSON file
        try:
            with open(output_system_kpi, 'w', encoding='utf-8') as f:
                json.dump(system_kpi, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number} System KPI exported successfully to: {output_system_kpi}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_system_kpi}: {e}")

        # Write Memories to JSON file
        try:
            with open(output_memories, 'w', encoding='utf-8') as f:
                json.dump(daily_memories_list, f, ensure_ascii=False, indent=4)
            log.info(f"✅ Day {day_number}'s {len(daily_memories_list)} memories exported successfully to: {output_memories}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_memories}: {e}")

        try:
            with open(output_agent_think_memories, 'w', encoding='utf-8') as f:
                json.dump(results_agent_think_memory, f, ensure_ascii=False, indent=4)
            log.info(
                f"✅ Day {day_number}'s {len(results_agent_think_memory)} thought memories exported successfully to: {output_agent_think_memories}")
        except Exception as e:
            log.error(f"Error writing to JSON file {output_agent_think_memories}: {e}")


# --- Recommended flattening helper function ---
def _flatten_metadata_fully(metadata: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    A helper function to recursively flatten nested dictionaries by joining keys.
    Example: {"a": {"b": 1}} -> {"a_b": 1}
    """
    items = []
    for k, v in metadata.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_metadata_fully(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
