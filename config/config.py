from typing import List

import yaml
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PlatformSettings(BaseModel):
    """平台的动态博弈与策略调整相关参数"""
    tau_tech: float = Field(
        ge=0,
        description="平台的检测器水平。值越高，准确度越高。"
    )
    steep: float = Field(
        ge=0,
        description="压力敏感度调节因子：决定tanh曲线的陡峭程度，即平台对压力的敏感度。"
    )
    eta: float = Field(
        ge=0,
        description="策略调整速率：决定了每一步调整的最大幅度。"
    )
    mu: float = Field(
        ge=0,
        description="基础影响力单价：将抽象的“影响力”货币化，与监管成本进行比较。"
    )
    w: float = Field(
        ge=0,
        description="用户流失厌恶系数：>1表示怕用户流失，<1表示怕政府惩罚。"
    )
    simple_run_days: int = Field(
        ge=0,
        description="简化框架运行天数。"
    )
    complete_run_days: int = Field(
        ge=0,
        description="完整框架运行天数。"
    )
    simple_batch_size: int = Field(
        ge=0,
        description="简化框架中，一次批量处理的数量。"
    )
    dissatisfaction_per_fp: float = Field(
        ge=0,
        description="每一次误报产生的不满点数。"
    )
    dissatisfaction_decay_rate: float = Field(
        ge=0,
        description="每日不满值的衰减率 (5%的不满会被“遗忘”)。"
    )
    dissatisfaction_to_churn_cost_factor: float = Field(
        ge=0,
        description="将累积不满值转化为潜在用户流失成本的因子。"
    )
    post_wish_threshold: float = Field(
        ge=-10,
        description="当智能体对平台的满意度低于这个阈值，就不想再发布内容。"
    )
    is_active_threshold: float = Field(
        ge=-10,
        description="当智能体对平台的满意度低于这个阈值，就脱离平台。"
    )
    import_policy_day_time: int = Field(
        ge=-1,
        description="导入策略的开始时间。"
    )
    kpi_window_size: int = Field(
        ge=-1,
        description="KPI窗口大小。"
    )
    critical_survival_count: int = Field(
        default=6,
        ge=1,
        description="群体最小生存阈值。人数小于等于该值时，全员晋升为代表。"
    )
    representative_sampling_mode: str = Field(
        default="mixed",
        description="代表选择模式，可选 mixed / random / influence。mixed 为影响力优先+随机补位。"
    )
    public_daily_mood_std: float = Field(
        default=0.08,
        ge=0,
        description="Daily Gaussian mood noise std for public follower satisfaction updates."
    )
    ai_proportion_noise_std: float = Field(
        default=0.2,
        ge=0,
        description="Gaussian noise std for shadow creator AI proportion around representative base value."
    )
    case_validation: bool = Field(
        default=False,
        description="是否开启案例验证"
    )
    efficiency_validation: bool = Field(
        default=False,
        description="是否开启效率验证"
    )
    ablation_validation: bool = Field(
        default=False,
        description="是否开启消融验证"
    )


class PublicAgentSettings(BaseModel):
    """公众智能体的行为参数"""
    number_of_interactions: int = Field(
        gt=0,
        description="智能体在浏览内容后，互动的次数达到这个限制，就结束浏览。"
    )
    number_of_compressions: int = Field(
        gt=0,
        description="智能体ReAct流程中，消息累积到这个数量，就触发消息压缩。"
    )
    number_of_keep: int = Field(
        gt=0,
        description="智能体react流程中，消息压缩后，保留消息的数量"
    )


class FileLoadPathSettings(BaseModel):
    """所有外部文件的加载路径"""
    contents_file: str = Field(description="内容文件路径")
    watermark_file: str = Field(description="水印技术知识库文件路径")
    personas_file: str = Field(description="智能体人设文件路径")
    token_file: str = Field(description="用于存储Token消耗记录的CSV文件路径")
    chroma_db_file: str = Field(description="Chroma向量数据库的存储目录")
    daily_memory_exports_file: str = Field(description="每日导出记忆的存储目录")
    base_store_file: str = Field(description="每次运行实验时，存储结果数据的目录.他是动态创建的")


class NsgeSettings(BaseModel):
    """NSGA 算法参数"""
    population_size: int = Field(description="nsga算法初始种群大小")
    generations: int = Field(description="nsga迭代轮数")


class ModelSettings(BaseModel):
    """不同任务所使用的LLM模型名称"""
    public_scan_model: str = Field(description="浏览内容模型")
    public_summarize_model: str = Field(description="总结内容模型")
    creator_model: str = Field(description="创作者驱动模型")
    platform_model: str = Field(description="平台驱动模型")
    compression_memory_model: str = Field(description="消息压缩模型")
    simple_model: str = Field(description="简化仿真框架驱动的模型")
    dialogue_history_model: str = Field(description="对话历史总结模型")


class LLMKeySettings(BaseModel):
    """LLM模型密钥"""
    qwen: str = Field(description="选择qwen的哪个key使用")
    single_key_concurrency_num: int = Field(description="单key并发数")


class LLMSettings(BaseModel):
    """存储敏感的API密钥"""
                                    
    model_config = ConfigDict(extra='allow')


                                                                     
                                
                                                                     

class AppSettings(BaseSettings):
    """
    项目总配置模型。
    它会首先从 YAML 文件加载结构化数据，
    然后自动从 .env 文件或环境变量中查找并覆盖 LLM 密钥等敏感信息。
    """
                              
    model_config = SettingsConfigDict(
        env_file='.env',                   
        env_file_encoding='utf-8',             
        env_nested_delimiter='__',                             
        extra='ignore'                  
    )

             
    platform: PlatformSettings
    public_agent: PublicAgentSettings
    nsga: NsgeSettings
    file_load_path: FileLoadPathSettings
    model: ModelSettings
    llm: LLMSettings
    llm_key: LLMKeySettings


                                                                     
                                    
                                                                     

def load_settings(config_path: str | Path) -> AppSettings:
    """
    从指定的YAML配置文件路径加载配置。

    Args:
        config_path: YAML配置文件的路径。

    Returns:
        一个经过完整校验和填充的 AppSettings 实例。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parents[1]
    file_paths = config_data.get("file_load_path", {})
    for key, value in list(file_paths.items()):
        if isinstance(value, str) and not Path(value).is_absolute():
            file_paths[key] = str(project_root / value)

                               
                                           
    return AppSettings.model_validate(config_data)


if __name__ == '__main__':
    settings = load_settings("config.yaml")
    print('2')
