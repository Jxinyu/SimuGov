from __future__ import annotations
import datetime
import json
import os
import uuid
from typing import Any, List, Literal, Set
from pydantic import TypeAdapter
from method.agent.persona import Persona
from method.agent.content import Content, ContentStore
from method.store.long_memory_store import MemoryStore
import asyncio
from config import settings
import logging
from method.utils.get_llm import get_api_key_count
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


def _coerce_top_level_persona_data(data: Any) -> List[dict]:
    """支持顶层数组，或 {'personas'|'agents'|'data': [...]} 包装。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("personas", "agents", "data"):
            inner = data.get(key)
            if isinstance(inner, list):
                return inner
    raise ValueError(
        "人设文件须为 JSON 数组，或包含 personas / agents / data 列表的对象"
    )


def _normalize_persona_dict(raw: dict) -> dict:
    """补齐历史字段，兼容 trend 类文件缺少 attack_resource 等情况。"""
    out = dict(raw)
    if out.get("attack_resource") is None:
        out["attack_resource"] = 2
    return out


def load_personas():
    log.info("加载人员信息...")
    with open(settings.file_load_path.personas_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_list = _coerce_top_level_persona_data(data)
    persona_list = [
        Persona.model_validate(_normalize_persona_dict(item))
        for item in raw_list
    ]
    personas = {p.agent_id: p for p in persona_list}
    log.info("人员信息加载完成")
    return personas


def load_contents():
    log.info("加载内容信息...")
    adapter = TypeAdapter(List[Content])
    with open(settings.file_load_path.contents_file, "r", encoding="utf-8") as f:
        content_list = adapter.validate_json(f.read())
    content_store = ContentStore()
                                                          
    log.info("内容信息加载完成")
    return content_store


def platform_background_prompt():
    return """
# 语言风格要求（科学仿真）
- 文明表达：禁止脏话、暴力、极端仇恨言论
- 学术化描述：用“抗议”“抵制”“感到背叛”代替攻击性词汇
- 针对机制而非个人：愤怒指向AI机制与平台政策

# 世界背景：ArtStation平台
- 专业作品集平台，行业招聘核心渠道
- 核心价值：人类手工技艺、创意、长期练习
- 成功标准：热门推荐（点赞/评论/分享）决定职业认可
- 社区文化：极度推崇人类创作，反对AI生成内容

(使用中文回答)
    """


def calculate_dynamic_thresholds(personas, mu):
    """动态计算成本阈值。"""

                                        
    total_benevolent_influence = sum(
        p.influence for p in personas.values()
        if p.type in ["合规创作者", "公众"]
    )
    max_c_churn = mu * total_benevolent_influence

    CHURN_THRESHOLDS_PERCENT = {"CRITICAL": 0.80, "HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0}

    churn_thresholds = {
        level: max_c_churn * percent for level, percent in CHURN_THRESHOLDS_PERCENT.items()
    }

                                  
                                      
    REG_THRESHOLDS_PERCENT = {
        "CRITICAL": 0.80,                             
        "HIGH": 0.50,        
        "MEDIUM": 0.30,          
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
              
        self.fn = []                               
        self.fp = []                               
        self.theta = 0.8           
        self.public_loss = []                        
        self.watermark_id = 'W3'                
        self.total_fp_creator_influence = 0.0                 
        self.name = 'platform' + str(datetime.datetime.now())        
        self.broadcast = ['']                     

        self.platform_theta_change = []                    
        self.cost_calculation_details_data = []            
        self.public_loss_data = []             
        self.public_change_role_data = []               
        self.kpi_change_data = []           
        self.fn_data = []            
        self.fp_data = []            
        self.creator_data = {}              
        self.personas_call_tool = {}               

              
        self.tau_tech = settings.platform.tau_tech                       
        self.steep = settings.platform.steep                                               
        self.eta = settings.platform.eta                  
        self.mu = settings.platform.mu                                                
        self.w = settings.platform.w                                  
        self.background_prompt = platform_background_prompt()
        self.calculate_dynamic_thresholds = calculate_dynamic_thresholds(personas, self.mu)        

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
        self.ai_threshold = ai_threshold                               
        self.f_penalty = f_penalty            
        self.e_edu = e_edu                              


class SystemKPI:
    """
    系统KPI
    """

    def __init__(self):
        self.safety = []       
        self.creativity = []       
        self.satisfaction = []         
        self.theta = []


class Environment:
    def __init__(self, policy: Policy):
        self.state_lock = asyncio.Lock()
        self.watermark_technology_library = load_watermark_technology_library()         
        self.personas = load_personas()        
        self.contents = load_contents()        

        self.memories_store = MemoryStore()           

        self.policy = policy        
        self.system_kpi = SystemKPI()         
        self.platform = Platform(self.personas)           
        self.day_time = 0      
        self.background_tasks: Set[asyncio.Task] = set()             

                   
        self.initial_creator_count = len(
            [p for p in self.personas.values() if p.type == '合规创作者']
        )
                 
        self.initial_persona_count = len(self.personas)
                
        self.initial_public_count = len(
            [p for p in self.personas.values() if p.type == '公众']
        )
                   
        self.initial_breaker_count = len(
            [p for p in self.personas.values() if p.type == '水印破坏者']
        )
                    
        self.llm_concurrent_nums_semaphore = asyncio.Semaphore(
            get_api_key_count() * settings.llm_key.single_key_concurrency_num)

                             
        for k, v in self.personas.items():
            self.platform.personas_call_tool[k] = [v.model_dump(exclude={'viewed_content', 'reacted_content'})]

    def start_new_day(self):
        self.day_time += 1
        self.platform.total_fp_creator_influence *= settings.platform.dissatisfaction_decay_rate
        self.system_kpi.theta.append(self.platform.theta)
                            
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

                                                 
        def _task_done_callback(t: asyncio.Task):
            self.background_tasks.discard(t)
            try:
                                             
                t.result()
                log.info(f"后台任务 {t.get_name()} 已成功完成。")
            except asyncio.CancelledError:
                                   
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
