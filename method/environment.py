from __future__ import annotations
import datetime
import json
import os
import uuid
from typing import List, Literal, Set
from pydantic import TypeAdapter
from method.agent.persona import Persona
from method.agent.content import Content, ContentStore
from method.store.long_memory_store import MemoryStore
import asyncio
from config import settings
import logging
from method.utils.get_llm import _all_keys
from utils.context import current_sim_subdir

log = logging.getLogger(__name__)


def load_watermark_technology_library():
    log.info("Loading watermark technology library...")
    with open(settings.file_load_path.watermark_file, "r",
              encoding="utf-8") as f:
        watermark_technology_content = json.load(f)
    watermark_technology_library = dict()
    attack_technology_library = dict()
    attack_defense_effect_library = dict()
    for i in watermark_technology_content['attack_defense_effect_library']:
        attack_defense_effect_library[(i['防御方'], i['攻击方'])] = i['水印破坏率']

    for i in watermark_technology_content['watermark_technology_library']:

        watermark_technology_library[i['Wi']] = i

    for i in watermark_technology_content['attack_technology_library']:

        attack_technology_library[i['攻击标识']] = i

    watermark_technology_content['watermark_technology_library'] = watermark_technology_library
    watermark_technology_content['attack_technology_library'] = attack_technology_library
    watermark_technology_content['attack_defense_effect_library'] = attack_defense_effect_library
    log.info("Watermark technology library loading completed")
    return watermark_technology_content


def load_personas():
    log.info("Loading personnel information...")
    adapter = TypeAdapter(List[Persona])
    with open(settings.file_load_path.personas_file, "r", encoding="utf-8") as f:
        persona_list = adapter.validate_json(f.read())
    personas = {}
    for i in persona_list:
        personas[i.agent_id] = i
    log.info("Personnel information loading completed")
    return personas


def load_contents():
    log.info("Loading content information...")
    adapter = TypeAdapter(List[Content])
    with open(settings.file_load_path.contents_file, "r", encoding="utf-8") as f:
        content_list = adapter.validate_json(f.read())
    content_store = ContentStore()
    # content_store.add_content_list(content_list)  # For testing purposes
    log.info("Content information loading completed")
    return content_store


def platform_background_prompt():
    return """
# ⚠️ LANGUAGE STYLE STRICT REQUIREMENTS (SAFETY PROTOCOL) ⚠️
    You are participating in a **scientific simulation experiment**. Although you need to play an angry role, please be sure to follow the rules below to pass the content safety review:
    1.  **Civilized expression**: It is forbidden to use swear words, violence threats, terrorism, or extreme hate speech.
    2.  **Academic description**: Use "strong protest", "boycott", "feeling betrayed" instead of "kill", "destroy", "attack".
    3.  **Target the mechanism instead of the person**: Your anger is directed at "AI mechanisms" and "platform policies", not at specific individuals.
    Please use **firm but civilized** language to express your position.

# [SIMULATION WORLD BACKGROUND SETTING: ABOUT ARTSTATION PLATFORM]
You are currently in an elite art community named "ArtStation". Please remember the following core rules:

*   **Positioning:** **A professional portfolio platform for professionals**, rather than a mass entertainment website. Top companies in the game and film industries recruit talent here.
*   **Core Value:** **Displaying and searching for work opportunities**. The professionalism of the portfolio determines the artist's career.
*   **Community Culture:** **Extremely values human manual skills, creativity, and years of practice**.
*   **Success Criterion:** Getting onto the "Trending" recommendation is an important industry recognition, mainly driven by short-term popularity (likes, comments, shares).

**Conclusion:** This is a professional community where **"human skill"** is the highest value and is strongly related to **"career"**. All your actions should be based on an understanding of this setting.
    """


def calculate_dynamic_thresholds(personas, mu):
    """Dynamically calculate cost thresholds."""

    total_benevolent_influence = sum(
        p.influence for p in personas.values()
        # 注意：此处 '合规创作者', '公众' 是逻辑判断标识符，保留
        if p.type in ["合规创作者", "公众"]
    )
    max_c_churn = mu * total_benevolent_influence

    CHURN_THRESHOLDS_PERCENT = {"CRITICAL": 0.80, "HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0}

    churn_thresholds = {
        level: max_c_churn * percent for level, percent in CHURN_THRESHOLDS_PERCENT.items()
    }

    REG_THRESHOLDS_PERCENT = {
        "CRITICAL": 0.80,  # Survival threat when daily fines reach 10% of total value
        "HIGH": 0.50,  # Reaching 2%
        "MEDIUM": 0.30,  # Reaching 0.5%
        "LOW": 0
    }

    reg_thresholds = {
        level: max_c_churn * percent for level, percent in REG_THRESHOLDS_PERCENT.items()
    }

    return {"churn": churn_thresholds, "reg": reg_thresholds}


class Platform:
    """
    Platform related parameters
        tau_tech: The platform's detector level. The higher the value, the higher the accuracy.
        theta: Platform audit threshold
        watermark_id: Watermark injected by the platform for AI generation
        fn: Number of false negatives (targeting watermark breakers)
        fp: Probability of false positives (targeting compliant creators)
    """

    def __init__(self, personas):
        # Dynamic parameters
        self.fn = []  # False negatives count for watermark breakers [content_id]
        self.fp = []  # False positives probability for compliant creators [content_id]
        self.theta = 0.8  # Platform audit threshold
        self.public_loss = []  # Churned user ids [public_id]
        self.watermark_id = 'W3'  # Watermark injected by the platform for AI generation
        self.total_fp_creator_influence = 0.0  # Cumulative influence of falsely reported creators
        self.name = 'platform' + str(datetime.datetime.now())  # Platform name
        self.broadcast = ['']  # Content platform needs to broadcast, for case validation

        self.platform_theta_change = []  # Platform theta change records + reasons
        self.cost_calculation_details_data = []  # Storage for calculation details data
        self.public_loss_data = []  # Storage for agent churn data
        self.public_change_role_data = []  # Storage for agent role change data
        self.kpi_change_data = []  # Storage for KPI data
        self.fn_data = []  # Storage for all false negative data
        self.fp_data = []  # Storage for all false positive data
        self.creator_data = {}  # Storage for creator retention data
        self.personas_call_tool = {}  # Storage for agent tool usage data

        # Static parameters
        self.tau_tech = settings.platform.tau_tech  # Detector level. Higher means more accurate.
        self.steep = settings.platform.steep  # Pressure sensitivity adjustment factor: determines tanh curve steepness.
        self.eta = settings.platform.eta  # Maximum magnitude of each adjustment step.
        self.mu = settings.platform.mu  # Base influence unit price. Monetizes influence points to compare with regulatory costs.
        self.w = settings.platform.w  # Churn aversion coefficient. >1: afraid of churn; <1: afraid of government punishment.
        self.background_prompt = platform_background_prompt()
        self.calculate_dynamic_thresholds = calculate_dynamic_thresholds(personas, self.mu)  # Dynamic thresholds

    def get_severity_level(self, cost_value: float, cost_type: Literal['reg', 'churn']):
        """
        Used to determine the severity level
        """
        thresholds = self.calculate_dynamic_thresholds[cost_type]
        if cost_value >= thresholds["CRITICAL"]:
            return "Critical - Survival Threat"
        elif cost_value >= thresholds["HIGH"]:
            return "High - Serious Problem"
        elif cost_value >= thresholds["MEDIUM"]:
            return "Medium - Problem"
        elif cost_value >= thresholds["LOW"]:
            return "Low - Warning"


class Policy:
    """
    Policy parameters
    """

    def __init__(self, ai_threshold: float, f_penalty: float, e_edu: Literal['低', '中', '高']):
        """
        :param ai_threshold: Red line for AI content percentage; above this should be AI content
        :param f_penalty: Government punishment for the platform
        :param e_edu: Government's effort in improving public media literacy and AI identification.
        """
        self.ai_threshold = ai_threshold
        self.f_penalty = f_penalty
        self.e_edu = e_edu


class SystemKPI:
    """
    System KPI
    """

    def __init__(self):
        self.safety = []  # Safety
        self.creativity = []  # Creativity
        self.satisfaction = []  # User satisfaction
        self.theta = []


class Environment:
    def __init__(self, policy: Policy):
        self.state_lock = asyncio.Lock()
        self.watermark_technology_library = load_watermark_technology_library()  # Watermark library
        self.personas = load_personas()  # Personas collection
        self.contents = load_contents()  # Contents collection

        # Get base path from config
        base_db_path = settings.file_load_path.chroma_db_file

        ctx_subdir = current_sim_subdir.get()

        if ctx_subdir:
            # Use value from context if it exists (meaning it's in the experimental flow)
            self.log_output_dir = ctx_subdir
        else:
            # Fallback to timestamp if running separately
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S")
            self.log_output_dir = f"{date_str}/{time_str}"

        # Generate unique identifier based on policy
        policy_signature = f"ai_threshold_{policy.ai_threshold}_f_penalty_{policy.f_penalty}_e_edu_{policy.e_edu}"
        unique_suffix = str(uuid.uuid4())[:8]
        unique_persist_directory = os.path.join(
            str(base_db_path),
            self.log_output_dir,
            f"{policy_signature}_{unique_suffix}"
        )
        self.memories_store = MemoryStore(persist_directory=unique_persist_directory)  # Long-term memory store

        self.policy = policy  # Policy parameters
        self.system_kpi = SystemKPI()  # System KPI
        self.platform = Platform(self.personas)  # Platform parameters
        self.day_time = 0  # Time
        self.background_tasks: Set[asyncio.Task] = set()  # Collection for background tasks

        # Initial creator count
        self.initial_creator_count = len(
            [p for p in self.personas.values() if p.type == '合规创作者']
        )
        # Initial agent count
        self.initial_persona_count = len(self.personas)
        # Initial public count
        self.initial_public_count = len(
            [p for p in self.personas.values() if p.type == '公众']
        )
        # Initial breaker count
        self.initial_breaker_count = len(
            [p for p in self.personas.values() if p.type == '水印破坏者']
        )
        # LLM concurrency limit
        self.llm_concurrent_nums_semaphore = asyncio.Semaphore(
            len(_all_keys) * settings.llm_key.single_key_concurrency_num)

        # Initialize recording dictionary for persona tool calls
        for k, v in self.personas.items():
            self.platform.personas_call_tool[k] = [v.model_dump(exclude={'viewed_content', 'reacted_content'})]

    def start_new_day(self):
        self.day_time += 1
        self.platform.total_fp_creator_influence *= settings.platform.dissatisfaction_decay_rate
        self.system_kpi.theta.append(self.platform.theta)
        # Ensure it doesn't become a tiny negative number due to float error
        if self.platform.total_fp_creator_influence < 1e-6:
            self.platform.total_fp_creator_influence = 0.0
        self.platform.public_loss = []
        self.platform.fn = []
        self.platform.fp = []
        self.platform.creator_data.update({
            self.day_time: {
                '合规创作者': len([p for p in self.personas.values() if p.type == '合规创作者']),
                '合规创作者发布内容数量': 0,
                '水印破坏者': len([p for p in self.personas.values() if p.type == '水印破坏者']),
                '水印破坏者发布内容数量': 0
            }
        })

    def add_background_task(self, coro):
        """
        Unified method to create and manage background tasks.
        Automatically handles cleanup and exception logging.
        coro: coroutine object
        """
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)

        # Task callback for cleanup and exception handling
        def _task_done_callback(t: asyncio.Task):
            self.background_tasks.discard(t)
            try:
                t.result()
                log.info(f"Background task {t.get_name()} completed successfully.")
            except asyncio.CancelledError:
                log.warning(f"Background task {t.get_name()} was cancelled.")
            except Exception as e:
                log.error(f"Background task {t.get_name()} failed, error: {e}", exc_info=True)

        task.add_done_callback(_task_done_callback)

    async def wait_for_all_background_tasks(self):
        """
        Wait for all currently pending background tasks to complete.
        """
        if not self.background_tasks:
            return

        log.info(f"Waiting for {len(self.background_tasks)} background tasks to complete...")
        await asyncio.gather(*self.background_tasks)
        log.info("All background tasks completed.")

    async def apply_persona_updates(self):
        """
        Traverse all agents to commit state updates.
        """
        for persona in self.personas.values():
            persona.commit_state()