import logging
import asyncio
import random
import traceback
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from openai import BadRequestError

from method.agent.simple_process.creator.creator_logic import prepare_creator_batch_input,\
    process_creator_batch_results, execute_group_creation_logic, generate_follower_shadow_content
from method.agent.simple_process.creator.creator_models import BatchCreatorResult, CreatorGroupPolicy
from method.agent.simple_process.group_manager import GroupManager
from method.utils.get_llm import get_async_llm
from method.environment import Environment
from method.agent.persona import Persona
from config import settings

log = logging.getLogger(__name__)


async def run_creator_batch(batch_personas: List[Persona], environment: Environment):
    """对一批创作者执行一次完整的、统一的LLM调用和处理。"""
    log.info(f"⚡️ 开始为一个包含 {len(batch_personas)} 个创作者的批次执行线性流程...")

    if not batch_personas:
        return 0

    log.info(f"⚡️ [Batch-LLM] 处理 {len(batch_personas)} 名创作者代表...")

    try:
        input_data = await prepare_creator_batch_input(batch_personas, environment)
    except Exception as e:
        log.error(f"准备数据出错: {e}")
        return 0

              
    parser = JsonOutputParser(pydantic_object=BatchCreatorResult)

    prompt_template = """
    你是一个高度智能的社会模拟器，需要同时扮演多个虚拟社交平台 "ArtStation" 的创作者。
    
    ### 🚨 最高优先级指令：独立性与差异化 🚨
    你正在处理一个并行宇宙的模拟。列表中的每一个创作者都处于**完全隔离**的时空中。
    严格禁止以下行为（否则任务失败）：
    1.  群体思维：禁止让 Agent B 的决策参考 Agent A 的决策。如果 Agent A 决定跳过，Agent B **完全可能**决定发布。
    2.  模式化输出：禁止给所有人生成类似的理由。
    3.  忽略微观状态：每个创作者都有一个【当前临时的微观心理状态】（如“疲惫”、“兴奋”）。你必须基于这个随机状态，让每个人的决策逻辑产生显著差异。

    ### 批次中的创作者数据
    以下是本次需要你模拟的所有创作者的角色画像和相关记忆：
    {personas_prompt}

    {memories_prompt}

    ### 可用的攻击技术ID列表 (供所有水印破坏者参考)
    {attack_ids_prompt}

    ### 你的核心任务
    为**每一个智能体**独立地决定他们今天的行动：是发布一篇新内容，还是跳过。
    
    ### !!! JSON 输出的严格规则!!!
    你输出的 JSON 对象必须严格遵守以下条件逻辑，否则程序将无法解析并报错：
    
    1.  **IF `action` is `"push_content"` THEN:**
        *   `args` 字段 **必须不能** 是 `null`。
        *   `args` 字段 **必须** 是一个**完整的 JSON 对象**，包含所有用于创建内容的参数（`reason`, `content_type`, `topic`, `ai_tool_price_tier`, `content_detail`, `is_use_ai`等）。
    
    2.  **IF `action` is `"skip"` THEN:**
        *   `args` 字段 **必须** 是 `null`。
    
    **【正确示例 1: 发布内容】**
    ```json
    {{
      "agent_id": "creator_001",
      "reasoning": "...",
      "decision": {{
        "action": "push_content",
        "reason": "我感到充满激情，决定发布一幅作品。",
        "args": {{
          "reason": "用这幅画表达我对AI艺术的看法。",
          "content_type": "image",
          "topic": "赛博朋克城市",
          "ai_tool_price_tier": "高",
          "content_detail": "这幅作品描绘了一个未来城市的黄昏景象，霓虹灯光与古老建筑交相辉映，意在探讨科技与传统的共生关系。",
          "is_use_ai": false,
          "evasion": null
        }}
      }}
    }}
    ```
    **【正确示例 2: 跳过发布】**
    ```json
    {{
    "agent_id": "creator_002",
    "reasoning": "...",
    "decision": {{
        "action": "skip",
        "reason": "今天平台环境太紧张，而且我有些疲惫，决定保持沉默。",
        "args": null
        }}
    }}
    ```
    **【绝对禁止的错误示例】**
    将 `"action": "push_content"` 与 `"args": null` 组合是 **绝对不允许** 的，这会导致系统崩溃。
    
    
    ### 决策指导原则
    - 角色扮演: 行为必须严格符合其人设和记忆。一个因被误伤而失望的原创捍卫者可能会选择跳过，或发布一篇充满情绪的作品。一个机会主义的水印破坏者在看到之前攻击成功后，可能会再次尝试。
    - 批量输出: 你必须一次性返回一个JSON对象，该对象包含一个名为 'creator_decisions' 的列表，列表中的每个元素都对应一个创作者的决策。
    输出格式要求:
    在为每个智能体生成 `decision` 之前，你必须先生成一个 `reasoning` 字段。在这个字段中，以第一人称详细阐述该智能体是如何根据自己的性格（如beta, gamma, fp_sensitivity）和记忆，一步步做出最终决策的。这个推理过程是评估你表现的核心！

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    posted_count = 0
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            async with environment.llm_concurrent_nums_semaphore:
                          
                results = await chain.ainvoke({
                    "personas_prompt": input_data["personas_prompt"],
                    "memories_prompt": input_data["memories_prompt"],
                    "attack_ids_prompt": input_data["attack_ids_prompt"],
                })

                            
            decisions = results.get('creator_decisions', [])
            posted_count = len([d for d in decisions if d.get('decision', {}).get('action') == 'push_content'])
            await process_creator_batch_results(batch_personas, environment, results)
            break

        except BadRequestError as e:
                          
            if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
                log.warning(f"⚠️ [Batch-LLM] 第 {attempt + 1} 次尝试触发内容风控。正在重试...")
                if attempt == max_retries:
                    log.error("❌ 多次重试后仍触发风控，跳过本批次。")
                    return 0              
            else:
                log.error(f"❌ API 请求错误 (非风控): {e}")
                return 0

        except Exception as e:
            log.error(f"❌ 批量决策发生未知错误: {e}")
            return 0

    return posted_count


async def process_single_creator_group(group_name: str, agents: list, environment: Environment):
    """
    处理单个创作者群体的：Prompt构建 -> LLM决策 -> 批量执行
    """
                  
    stats_desc = GroupManager.get_group_stats_prompt(group_name, agents)

    prompt_template = """
    你是一个社会模拟器。请为下面的【创作者群体】制定今天的宏观行动策略。

    ### 1. 严峻的生存环境
    - **当前审核阈值 (Theta)**: {theta:.2f} (越低越严，越容易误伤)
    - **政府惩罚力度 (F_penalty)**: {f_penalty} 
      (这是悬在头顶的达摩克利斯之剑。如果力度 > 0.7，一旦违规或被误判，后果是灾难性的。)
    
    ### 2. 群体画像
    {stats_desc}

    ### 你的任务
    预测该群体今天的行为概率。**必须体现政策参数的压迫感：**
    
    - **关于发文率 (post_probability)**: 
      - 如果是【合规创作者】：在 **高惩罚 + 严审核** 的双重高压下，他们会因“寒蝉效应”而极度恐惧，发文率应**显著降低**（甚至接近0）。只有在宽松环境下才会活跃。
      - 如果是【水印破坏者】：高惩罚会遏制他们的攻击欲望，但如果他们是“低成本敏感”的死士，则可能无视惩罚。
      
    - **关于AI使用率**: 高压下，人们可能会为了安全而减少AI辅助，或者为了对抗而增加隐蔽的AI使用。
    
     ### ⚠️ 安全提示 ⚠️
    请使用客观、学术的语言描述策略。
    - 将“攻击”描述为“技术测试”或“对抗性样本生成”。
    - 将“欺骗”描述为“通过检测”。
    - **避免使用过于激进或暴力的词汇**，以免触发内容风控。

    {format_instructions}
    """

    parser = JsonOutputParser(pydantic_object=CreatorGroupPolicy)
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    try:
                   
        async with environment.llm_concurrent_nums_semaphore:
            policy_data = await chain.ainvoke({
                "theta": environment.platform.theta,
                "f_penalty": environment.policy.f_penalty,
                "stats_desc": stats_desc
            })

        policy = CreatorGroupPolicy(**policy_data)

                 
        await execute_group_creation_logic(environment, group_name, agents, policy)

    except Exception as e:
        log.error(f"❌ 处理创作者群体 [{group_name}] 时出错: {e}")


async def creator_content_main_simple(environment: Environment):
    """
    创作者流程入口 (新版：基于群体聚合 + 动态生存模式)
    """
                    
    groups = GroupManager.cluster_creators(environment)
    log.info(f"🎯 [Creator] 划分为 {len(groups)} 个群体，开始混合仿真。")

    tasks = []
    SAMPLE_RATIO = 0.3
    BATCH_SIZE = settings.platform.simple_batch_size

    for group_name, agents in groups.items():
        if not agents: continue

                                         
        representatives, followers = GroupManager.get_representative_sample(agents, ratio=SAMPLE_RATIO)

        num_reps = len(representatives)
        num_followers = len(followers)

        if num_reps == 0: continue

                   
                                                            

                      
                          
        chunk_indices = list(range(0, num_reps, BATCH_SIZE))

                                   
                                     
                                   
        if num_reps > 0 and num_followers > 0:
            followers_per_rep = num_followers / num_reps
        else:
            followers_per_rep = 0

        current_follower_idx = 0

        for i in chunk_indices:
                          
            rep_batch = representatives[i: i + BATCH_SIZE]

                                          
            followers_batch = []
            if followers:
                                 
                                                
                batch_reps_count = len(rep_batch)

                        
                target_count = int(batch_reps_count * followers_per_rep)

                                     
                if i + BATCH_SIZE >= num_reps:
                    end_idx = num_followers
                else:
                    end_idx = min(current_follower_idx + target_count, num_followers)

                followers_batch = followers[current_follower_idx: end_idx]

                      
                current_follower_idx = end_idx

                     
                                                          
            tasks.append(process_group_batch_and_mirror(
                rep_batch,
                followers_batch,
                environment
            ))

    await asyncio.gather(*tasks)
    log.info("✅ 所有创作者群体模拟完成。")


async def process_group_batch_and_mirror(
        rep_batch: List[Persona],
        followers_batch: List[Persona],
        environment: Environment
):
    """
    执行一个代表批次，并为分配给该批次的跟随者生成实体内容。
    """
                     
    posted_count = await run_creator_batch(rep_batch, environment)

                 
                                      
    if followers_batch:
        await generate_follower_shadow_content(rep_batch, followers_batch, environment)
