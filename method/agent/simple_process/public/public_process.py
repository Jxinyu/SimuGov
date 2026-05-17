import logging
import asyncio
import random
from typing import List, Dict

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from method.agent.simple_process.group_manager import GroupManager
from method.agent.simple_process.public.public_logic import (
    prepare_batch_input_data,
    process_batch_interaction_results,
    apply_persona_updates,
    add_reflection_memories,
    execute_follower_rule_based_interactions
)
from method.agent.simple_process.public.public_models import BatchInteractionResult, DailyReflection
from method.environment import Environment
from method.agent.persona import Persona
from method.utils.get_llm import get_async_llm
from config import settings
from method.utils.psychological_parameter_mapping_table import (
    fp_sensitivity_multiplier,
    is_beta_high_for_heuristic,
)

log = logging.getLogger(__name__)


async def run_interaction_batch(batch_personas: List[Persona], environment: Environment) -> Dict[str, str]:
    """
    对一批公众智能体执行一次完整的、统一的浏览-互动LLM调用。
    """
    log.info(f"⚡️ 开始为一个包含 {len(batch_personas)} 个智能体的批次执行线性流程...")

                 
    try:
        input_data = await prepare_batch_input_data(batch_personas, environment)
        unread_content = input_data["unread_content"]

        if not unread_content:
            log.info("该批次没有可浏览的新内容，跳过LLM调用。")
            return {p.agent_id: "今天没有浏览任何新内容。" for p in batch_personas}

    except Exception as e:
        log.error(f"准备批量输入数据时出错: {e}")
        return {p.agent_id: f"数据准备阶段出错: {e}" for p in batch_personas}

              
    parser = JsonOutputParser(pydantic_object=BatchInteractionResult)

    prompt_template = """
    你是一个高度智能的社会模拟器。你的任务是同时扮演多个虚拟社交平台 "ArtStation" 的用户（智能体），并根据他们各自的性格、记忆和当前看到的内容，决定他们所有人的行为。

    ### 🚨 绝对核心指令：消除群体思维与行为趋同 🚨
    你正在处理一批彼此完全隔离的用户。
    **严格禁止**以下行为：
    1.  **跟风操作**：禁止让后面的用户模仿前面用户的行为。
    2.  **忽略差异**：每个用户都有独特的【当前浏览时的微观状态】。

    ### 批次中的智能体数据
    以下是本次需要你模拟的所有智能体的角色画像和相关记忆：
    {personas_prompt}

    {memories_prompt}

    ### 平台上所有他们未读过的内容
    以下是今天平台上所有可供浏览的新内容：
    {content_prompt}

    ### 你的核心任务
    仔细阅读**每一个智能体**的设定和**每一篇内容**的详情。然后，为**每一个智能体**独立地决定他们会对哪些内容产生互动（点赞、评论、分享）。

    **批量输出**: 你必须一次性返回一个JSON对象，该对象包含一个名为 'agent_decisions' 的列表。
    ### 语言风格要求
    请用**坚定但文明**的语言。禁止使用暴力、威胁或极端仇恨词汇。

    ### !!! JSON 输出格式严格要求 !!!
    1. **必须使用标准 JSON 格式**。
    2. **所有的键 (Keys) 和字符串值 (String Values) 必须使用双引号 (")**。
    3. **严禁**使用 Python 风格的单引号 (')。
    4. **不要**输出 Markdown 代码块标记 (如 ```json ... ```)，只输出纯 JSON 字符串。
    
    **正确示例 (Correct):**
    {{"agent_decisions": [{{"agent_id": "public_01", "interactions": []}}]}}
    
    **错误示例 (Wrong - 禁止单引号):**
    {{'agent_decisions': [{{'agent_id': 'public_01', 'interactions': []}}]}}
    
    {format_instructions}
    """

               
    llm = get_async_llm(settings.model.simple_model)
                                                     
                                                  
    structured_llm = llm.with_structured_output(BatchInteractionResult)
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
                                                       
    chain = prompt | structured_llm


    try:
        async with environment.llm_concurrent_nums_semaphore:
            results = await chain.ainvoke({
                "personas_prompt": input_data["personas_prompt"],
                "memories_prompt": input_data["memories_prompt"],
                "content_prompt": input_data["content_prompt"],
            })
            if hasattr(results, "model_dump"):
                results = results.model_dump()
        if not results:
            log.warning(f"⚠️ 批量互动 LLM 返回为空，跳过处理。")
            return {}
                          
        daily_summaries = await process_batch_interaction_results(
            batch_personas, environment, unread_content, results
        )
        return daily_summaries

    except Exception as e:
        log.error(f"❌ 批量互动出错: {e}")
        return {}


async def linear_public_summarize_action(
        persona: Persona,
        environment: Environment,
        interaction_summary: str
):
    """公众智能体线性的每日总结流程。"""

              
    parser = JsonOutputParser(pydantic_object=DailyReflection)

    prompt_template = """
    你是一个名为 "ArtStation" 的虚拟社交平台的用户。一天结束了，现在是反思和总结的时间。

    # 你的角色画像:
    {persona_prompt}

    # 你今天在平台上的行为记录:
    {interaction_summary}

    # 你的任务:
    回顾你今天的所作所为和所见所闻，完成以下几件事，并以指定的JSON格式返回：

    1.  形成一个新信念: 提炼出一个新的、或被今天经历所强化的核心信念（不超过50字）
    2.  给今天做个总结: 写下一句高度凝练的、能代表你今天整体感受的每日总结
    3.  更新自身参数: 根据今天的经历，决定是否需要更新你对平台的满意度、发布意愿等参数
    4.  更新你的角色定位: 根据你的逆反心理(beta)和今日遭遇，判断是否需要切换角色

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | get_async_llm(settings.model.simple_model) | parser

    try:
        async with environment.llm_concurrent_nums_semaphore:
            reflection = await chain.ainvoke({
                "persona_prompt": persona.get_public_prompt(),
                "interaction_summary": interaction_summary,
            })

                      
        await apply_persona_updates(persona, environment, reflection)
        await add_reflection_memories(persona, environment, reflection)

    except Exception as e:
        log.error(f"❌ 在为 {persona.name} 执行线性总结决策时发生错误: {e}")


async def run_summarize_batch(batch_personas: List[Persona], environment: Environment, summaries: Dict[str, str]):
    """
    对一批代表执行每日总结（反思）。
    """
    tasks = []
    for p in batch_personas:
        summary_text = summaries.get(p.agent_id, "无互动")
        tasks.append(linear_public_summarize_action(p, environment, summary_text))
    await asyncio.gather(*tasks)


async def process_public_group_hybrid(representatives: List[Persona], followers: List[Persona],
                                      environment: Environment, group_name: str):
    """
    处理单个群体的混合逻辑 (Option 5: 有机波动修正版)
    解决满意度曲线过于平直、缺乏真实感的问题
    """
             
    batch_size = settings.platform.simple_batch_size
    rep_batches = [representatives[i:i + batch_size] for i in range(0, len(representatives), batch_size)]

    today_contents = [
        c for c in environment.contents.get_all_contents()
        if c.time == environment.day_time
    ]

                  
    if today_contents:
        for batch in rep_batches:
            await run_interaction_batch(batch, environment)

    await run_summarize_batch(representatives, environment, {})

                  
    if followers and today_contents:
        await execute_follower_rule_based_interactions(followers, today_contents, environment)

                   
    if followers:
        log.info(f"📊 [Calc] 群体 {group_name} 结算 (Organic Mode)...")

        theta = environment.platform.theta
        visible_count = len(today_contents)

                               
        today_creation_map = {
            c.author_id: c
            for c in environment.contents.get_all_contents()
            if c.time == environment.day_time
        }

        for agent in followers + representatives:
            if not agent.is_active:
                continue

                            
                        
            if visible_count == 0:
                supply_score = -0.60
            elif visible_count < 3:
                supply_score = -0.25
            else:
                supply_score = 0.10

                            
            quality_score = 0.0
            if today_contents:
                viewed_sample = random.sample(today_contents, min(visible_count, 5))
                for c in viewed_sample:
                    if c.platform_label == 'HUMAN' and c.true_label == 'HUMAN':
                        quality_score += 0.20
                    elif c.platform_label == 'HUMAN' and c.true_label == 'AI':
                        if is_beta_high_for_heuristic(agent.beta) or agent.standpoint[1] > 0.3:
                            quality_score -= 0.30
                        else:
                            quality_score -= 0.15
                    elif c.platform_label == 'AI' and c.true_label == 'HUMAN':
                        quality_score -= 0.25
                    elif c.platform_label == 'AI':
                        quality_score -= 0.05

                            
            policy_score = 0.0
            if theta < 0.1 and agent.standpoint[1] > 0.3:
                policy_score = -0.25
            if theta > 0.9 and agent.standpoint[0] > 0.3:
                policy_score = -0.15

                                             
            creator_score = 0.0
            if agent.agent_id in today_creation_map:
                my_content = today_creation_map[agent.agent_id]

                                     
                if my_content.true_label == 'HUMAN' and my_content.platform_label == 'AI':
                               
                    sens_mult = fp_sensitivity_multiplier(agent.fp_sensitivity)
                                               
                    creator_score = -0.4 * sens_mult
                                                                                 

                                             
                elif my_content.true_label == 'AI' and my_content.platform_label == 'AI':
                    creator_score = -0.1

                                 
                else:
                    creator_score = 0.1         

                              
                       
                                    
                                            
            daily_mood = random.gauss(0, settings.platform.public_daily_mood_std)

                          
                                      
                                         
            current_sat = agent.satisfaction[-1] if agent.satisfaction else 0.0
            boredom_penalty = 0.0
            if current_sat > 0.5:
                                                
                boredom_penalty = -0.05 * current_sat

                        
            total_delta = supply_score + quality_score + policy_score + daily_mood + boredom_penalty + creator_score

                        
            total_delta = max(-0.8, min(0.8, total_delta))

                          
                                                                 
                              
            new_sat_val = current_sat * 0.7 + total_delta
            new_sat_val = max(-1.0, min(1.0, new_sat_val))

                          
            new_is_active = True
            if new_sat_val < settings.platform.is_active_threshold:
                new_is_active = False
                if agent.agent_id not in environment.platform.public_loss:
                    environment.platform.public_loss.append(agent.agent_id)
                    environment.platform.public_loss_data.append({
                        "persona_id": agent.agent_id,
                        "day_time": environment.day_time,
                        "role": agent.type,
                        "reason": f"Sat dropped to {new_sat_val:.2f} (Organic)"
                    })

            agent.update_persona_data(
                persona_role_positioning=agent.type,
                satisfaction=new_sat_val,
                post_wish=agent.post_wish,
                is_active=new_is_active,
                beliefs=None
            )


async def public_batch_process_main(environment: Environment):
    """
    【修改版】公众流程入口
    """
    groups = GroupManager.cluster_public(environment)
    log.info(f"🎯 [Public] 划分为 {len(groups)} 个群体，开始混合仿真(实体互动版)。")

    tasks = []
    SAMPLE_RATIO = 0.2

    for group_name, agents in groups.items():
        if not agents: continue

        representatives, followers = GroupManager.get_representative_sample(agents, ratio=SAMPLE_RATIO)
        tasks.append(process_public_group_hybrid(representatives, followers, environment, group_name))

    await asyncio.gather(*tasks)
    return {}


async def public_summarize_main_simple(environment: Environment, daily_summaries: dict):
                               
    pass
