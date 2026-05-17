import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from method.agent.simple_process.platform.platform_logic import get_platform_daily_report, apply_platform_decision
from method.agent.simple_process.platform.platform_models import PlatformDecision
from method.environment import Environment
from method.utils.get_llm import get_async_llm
from config import settings

log = logging.getLogger(__name__)


async def linear_platform_main(environment: Environment):
    """
    平台智能体线性的每日决策流程。
    """
    log.info(f"⚡️ 开始为平台智能体 'Aura' 执行快速线性决策流程...")

                                                
    report = await get_platform_daily_report(environment)
                                                                       
                              
                                                                       
    decision_data = {
        "new_theta": report.system_recommendation,
        "reason": (f"【系统自动执行】基于净压力 {report.net_pressure:.2f} "
                   f"(监管={report.regulatory_cost:.2f}, 流失={report.total_churn_cost:.2f}) "
                   f"自动调整阈值。"),
        "net_pressure": report.net_pressure,
    }
              
    log.info(f"🤖 [Auto-Platform] 自动调整: {report.current_theta:.3f} -> {report.system_recommendation:.3f}")
                      
    await apply_platform_decision(environment, decision_data)


                                
          
                                                               
     
                                                                      
                                                   
                                                 
                     
           
                                                                                                                    
                                                          
     
                            
                                              
                
     
                
                                                                 
     
                           
                                       
                                                           
     
                      
                                           
                                                 
     
               
                                                                   
                                                                           
                                     
     
                                   
                                                                    
                                                                     
     
                                 
                                                      
                                                                             
                                                                           
                                                         
                                                                       
                                                                  
                                               
     
                 
                
     
              
                         
             
                           
                                   
                                                  
     
                           
         
     
                                                
                                   
                                                                                     
       
                                                                          
     
          
                                           
                                                                   
                                                  
                                        
                                                            
                                                                
                                                                                  
                                                                  
                                                                        
                                                                          
                                                                        
                                                  
                                                                
                                                          
                                                                            
                                               
                                                                
                                                                     
                    
                                                                                       
                                                  
                                                      
     
                   
                                                              
     
                            
                                                


async def platform_main_simple(environment: Environment):
    """ 平台智能体流程的总入口"""
    log.info("=" * 60)
    log.info("==========   开始 平台智能体 [快速线性] 流程   ==========")
    log.info("=" * 60)

    await linear_platform_main(environment)

    log.info("=" * 60)
    log.info("==========   平台智能体 [快速线性] 流程已完成   ==========")
    log.info("=" * 60)
