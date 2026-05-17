from pydantic import BaseModel, Field
from typing import Literal, Optional


class PlatformDailyReport(BaseModel):
    """定义每日平台运营报告的数据结构，作为LLM决策的输入。"""
    day: int = Field(description="当前日期。")
    current_theta: float = Field(description="平台当前的审核阈值。")

              
    regulatory_cost: float = Field(description="因漏报AI内容产生的监管成本/罚款风险。")
    regulatory_cost_severity: str = Field(description="监管成本的严重性评级。")

                
    total_churn_cost: float = Field(description="用户流失成本(总计)，反映社区健康度的核心指标。")
    explicit_churn_cost: float = Field(description="由今天实际用户流失事件产生的显性成本。")
    potential_churn_cost: float = Field(description="由累积的误报(FP)产生的潜在社区不满成本，是关键预警信号。")
    churn_cost_severity: str = Field(description="用户流失总成本的严重性评级。")

              
    fp_today: int = Field(description="今日新增的误报(FP)数量。")
    grievance_total: float = Field(description="当前累积的社区不满值。")

    net_pressure: float = Field(description="净压力。正值偏向收紧，负值偏向放松。")
    system_recommendation: float = Field(description="算法基于当前数据推荐的新阈值。")


class PlatformDecision(BaseModel):
    """定义平台智能体的最终决策，作为LLM的输出。"""
    new_theta: Optional[float] = Field(
        ge=0.05, le=0.95,
        description="此字段必须包含新的阈值(0.05-0.95)。"
    )
    reason: str = Field(
        min_length=20,
        description="做出此决策的详细、深思熟虑的理由，至少20字。"
    )
