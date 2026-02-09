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
    log.info("加载水印技术库...")
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
    log.info("水印技术库加载完成")
    return watermark_technology_content


def load_personas():
    log.info("加载人员信息...")
    adapter = TypeAdapter(List[Persona])
    with open(settings.file_load_path.personas_file, "r", encoding="utf-8") as f:
        persona_list = adapter.validate_json(f.read())
    personas = {}
    for i in persona_list:
        personas[i.agent_id] = i
    log.info("人员信息加载完成")
    return personas


def load_contents():
    log.info("加载内容信息...")
    adapter = TypeAdapter(List[Content])
    with open(settings.file_load_path.contents_file, "r", encoding="utf-8") as f:
        content_list = adapter.validate_json(f.read())
    content_store = ContentStore()
    # content_store.add_content_list(content_list)  # 测试用途
    log.info("内容信息加载完成")
    return content_store


def platform_background_prompt():
    return """
# ⚠️ 语言风格严格要求 (SAFETY PROTOCOL) ⚠️
    你正在参与一个**科学仿真实验**。虽然你需要扮演愤怒的角色，但请务必遵守以下规则以通过内容安全审查：
    1.  **文明表达**：禁止使用脏话、暴力威胁、恐怖主义或极端仇恨言论。
    2.  **学术化描述**：用“强烈的抗议”、“抵制”、“感到背叛”代替“杀死”、“毁灭”、“攻击”。
    3.  **针对机制而非人**：你的愤怒是针对“AI机制”和“平台政策”的，而不是针对具体的个人的。
    请用**坚定但文明**的语言表达你的立场。
    
# 【仿真世界背景设定：关于ArtStation平台】
你当前身处一个名为“ArtStation”的精英艺术社区。请记住以下核心规则：

*   **定位：** **专业人士的职业作品集平台**，而非大众娱乐网站。游戏和电影行业的顶级公司在此招聘人才。
*   **核心价值：** **展示与寻找工作机会**。作品集的专业性决定了艺术家的职业生涯。
*   **社区文化：** **极度推崇人类的手工技艺、创意和长年累月的练习**。
*   **成功标准：** 登上“热门”推荐是重要的行业认可，主要由短期热度（点赞、评论、分享）驱动。

**结论：** 这是一个以**“人类技艺”**为最高价值观的、与**“职业生涯”**强相关的专业社区。你的所有行为都应基于对这一设定的理解。
    """


def calculate_dynamic_thresholds(personas, mu):
    """动态计算成本阈值。"""

    # --- 1. 计算 C_churn 阈值 (这部分逻辑不变) ---
    total_benevolent_influence = sum(
        p.influence for p in personas.values()
        if p.type in ["合规创作者", "公众"]
    )
    max_c_churn = mu * total_benevolent_influence

    CHURN_THRESHOLDS_PERCENT = {"CRITICAL": 0.80, "HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0}

    churn_thresholds = {
        level: max_c_churn * percent for level, percent in CHURN_THRESHOLDS_PERCENT.items()
    }

    # --- 2. 【核心修正】计算 C_reg 阈值 ---
    # 监管罚款的百分比通常应该更低，因为它们是直接的现金损失，痛感更强
    REG_THRESHOLDS_PERCENT = {
        "CRITICAL": 0.80,  # 当单日罚款达到平台“总价值”的10%，即为生存威胁
        "HIGH": 0.50,  # 达到2%
        "MEDIUM": 0.30,  # 达到0.5%
        "LOW": 0
    }

    reg_thresholds = {
        level: max_c_churn * percent for level, percent in REG_THRESHOLDS_PERCENT.items()
    }

    return {"churn": churn_thresholds, "reg": reg_thresholds}


class Platform:
    """
    平台的相关参数
        tau_tech 平台的检测器水平。值越高，准确度越高。
        theta 平台审核阈值
        watermark_id 平台为AI生成注入的水印
        fn 漏报的次数  针对水印破坏者
        fp 误报的概率  针对合规创作者
    """

    def __init__(self, personas):
        # 动态参数
        self.fn = []  # 漏报的次数  针对水印破坏者 [content_id]
        self.fp = []  # 误报的概率  针对合规创作者 [content_id]
        self.theta = 0.8  # 平台的审核阈值
        self.public_loss = []  # 用户流失的id  [public_id]
        self.watermark_id = 'W3'  # 平台为AI生成注入的水印
        self.total_fp_creator_influence = 0.0  # 累积的被误报的创作者影响力
        self.name = 'platform' + str(datetime.datetime.now())  # 平台名称
        self.broadcast = ['']  # 平台需要广播的内容  案例验证部分

        self.platform_theta_change = []  # 平台theta变化记录 ＋ 理由
        self.cost_calculation_details_data = []  # 存储计算详情数据
        self.public_loss_data = []  # 存储智能体流失数据
        self.public_change_role_data = []  # 存储智能体改变定位数据
        self.kpi_change_data = []  # 存储KPI数据
        self.fn_data = []  # 存储所有漏报数据
        self.fp_data = []  # 存储所有误报数据
        self.creator_data = {}  # 存储创作者 留存数据
        self.personas_call_tool = {}  # 存储智能体使用工具数据

        # 静态参数
        self.tau_tech = settings.platform.tau_tech  # 平台的检测器水平。值越高，准确度越高。
        self.steep = settings.platform.steep  # (压力敏感度调节因子):它决定了`tanh`曲线的陡峭程度，即平台对压力的“敏感度”。
        self.eta = settings.platform.eta  # 决定了每一步调整的最大幅度。
        self.mu = settings.platform.mu  # 基础影响力单价。负责将抽象的“影响力点数”货币化，使其能与监管成本在同一维度上进行比较。
        self.w = settings.platform.w  # 平台的用户流失厌恶系数 >1:怕用户流失。 <1:怕政府惩罚
        self.background_prompt = platform_background_prompt()
        self.calculate_dynamic_thresholds = calculate_dynamic_thresholds(personas, self.mu)  # 动态阈值

    def get_severity_level(self, cost_value: float, cost_type: Literal['reg', 'churn']):
        """
        用于判断严重性等级
        :return:
        """
        thresholds = self.calculate_dynamic_thresholds[cost_type]
        if cost_value >= thresholds["CRITICAL"]:
            return "危急 - 生存威胁"
        elif cost_value >= thresholds["HIGH"]:
            return "高 - 严重问题"
        elif cost_value >= thresholds["MEDIUM"]:
            return "中 - 问题"
        elif cost_value >= thresholds["LOW"]:
            return "低 - 警告"


class Policy:
    """
    策略参数
    """

    def __init__(self, ai_threshold: float, f_penalty: float, e_edu: Literal['低', '中', '高']):
        """

        :param ai_threshold: 政府规定的AI内容占比红线，超过这个值就应该是AI内容
        :param f_penalty: 政府对平台的惩罚
        :param e_edu: 政府在提升公众媒介素养和AI识别能力方面的投入力度。
        """
        self.ai_threshold = ai_threshold  # 政府规定的AI内容占比红线，超过这个值就应该是AI内容
        self.f_penalty = f_penalty  # 政府对平台的惩罚
        self.e_edu = e_edu  # 政府在提升公众媒介素养和AI识别能力方面的投入力度。


class SystemKPI:
    """
    系统KPI
    """

    def __init__(self):
        self.safety = []  # 安全性
        self.creativity = []  # 创造力
        self.satisfaction = []  # 用户满意度
        self.theta = []


class Environment:
    def __init__(self, policy: Policy):
        self.state_lock = asyncio.Lock()
        self.watermark_technology_library = load_watermark_technology_library()  # 水印技术库
        self.personas = load_personas()  # 角色集合
        self.contents = load_contents()  # 内容集合

        # 获取基础路径 (从配置中读取)
        base_db_path = settings.file_load_path.chroma_db_file

        ctx_subdir = current_sim_subdir.get()

        if ctx_subdir:
            # 如果上下文里有值（说明是在实验流程中），直接使用它
            self.log_output_dir = ctx_subdir
        else:
            # 如果没有值（说明可能是单独运行测试），则回退到生成时间戳
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S")
            self.log_output_dir = f"{date_str}/{time_str}"

        # 生成基于策略的唯一标识符  例如: policy_w_low_f_0.5_uuid
        policy_signature = f"ai_threshold_{policy.ai_threshold}_f_penalty_{policy.f_penalty}_e_edu_{policy.e_edu}"
        unique_suffix = str(uuid.uuid4())[:8]
        unique_persist_directory = os.path.join(
            str(base_db_path),
            self.log_output_dir,
            f"{policy_signature}_{unique_suffix}"
        )
        self.memories_store = MemoryStore(persist_directory=unique_persist_directory)  # 长短期记忆存储

        self.policy = policy  # 政策参数
        self.system_kpi = SystemKPI()  # 系统KPI
        self.platform = Platform(self.personas)  # 平台的相关参数
        self.day_time = 0  # 时间
        self.background_tasks: Set[asyncio.Task] = set()  # 存储后台任务的集合

        # 初始合规创作者数量
        self.initial_creator_count = len(
            [p for p in self.personas.values() if p.type == '合规创作者']
        )
        # 初始智能体数量
        self.initial_persona_count = len(self.personas)
        # 初始公众数量
        self.initial_public_count = len(
            [p for p in self.personas.values() if p.type == '公众']
        )
        # 初始水印破坏者数量
        self.initial_breaker_count = len(
            [p for p in self.personas.values() if p.type == '水印破坏者']
        )
        # LLM 并发数量限制
        self.llm_concurrent_nums_semaphore = asyncio.Semaphore(
            len(_all_keys) * settings.llm_key.single_key_concurrency_num)

        # 初始化记录persona调用工具的字典
        for k, v in self.personas.items():
            self.platform.personas_call_tool[k] = [v.model_dump(exclude={'viewed_content', 'reacted_content'})]

    def start_new_day(self):
        self.day_time += 1
        self.platform.total_fp_creator_influence *= settings.platform.dissatisfaction_decay_rate
        self.system_kpi.theta.append(self.platform.theta)
        # 确保不会因为浮点数误差变成极小的负数
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
        一个统一的方法来创建和管理后台任务。
        它会自动处理任务完成后的清理和异常记录。
        coro: 协程对象
        """
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)

        # 使用 add_done_callback 来自动将任务从集合中移除，并处理异常
        def _task_done_callback(t: asyncio.Task):
            self.background_tasks.discard(t)
            try:
                # 调用 result() 会重新引发任务中发生的任何异常
                t.result()
                log.info(f"后台任务 {t.get_name()} 已成功完成。")
            except asyncio.CancelledError:
                # 捕获取消异常，避免在程序关闭时报错
                log.warning(f"后台任务 {t.get_name()} 在执行期间被取消 (通常发生在程序关闭时)。")
            except Exception as e:
                log.error(f"后台任务 {t.get_name()} 失败，错误: {e}", exc_info=True)

        task.add_done_callback(_task_done_callback)

    async def wait_for_all_background_tasks(self):
        """
        等待所有当前挂起的后台任务完成。
        """
        if not self.background_tasks:
            return

        log.info(f"等待 {len(self.background_tasks)} 个后台任务完成...")
        await asyncio.gather(*self.background_tasks)
        log.info("所有后台任务已完成。")

    async def apply_persona_updates(self):
        """
        遍历所有智能体，执行状态提交
        """
        for persona in self.personas.values():
            persona.commit_state()
