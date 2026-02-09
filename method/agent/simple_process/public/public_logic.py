import logging
import random
import asyncio
from typing import List, Dict

import numpy as np

from config import settings
from method.agent.content import Content
from method.agent.persona import Persona
from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


async def prepare_batch_input_data(
        personas: List[Persona],
        environment: Environment
) -> Dict[str, str | List[Content]]:
    """
    为一批智能体准备LLM调用所需的所有输入数据。
    """
    log.info(f"为 {len(personas)} 个智能体准备批量输入数据...")

    shuffled_personas = personas.copy()
    random.shuffle(shuffled_personas)

    # 1. 准备智能体画像和记忆的文本
    personas_prompt_str = ""
    memories_prompt_str = ""
    all_viewed_ids = set()

    for p in shuffled_personas:
        personas_prompt_str += p.get_public_prompt() + "\n"
        # 异步获取每个智能体的记忆
        memories = await environment.memories_store.recall_memories(
            persona_id=p.agent_id,
            query="我关于平台、AI内容、社区氛围的总体印象和经历",
            top_k=5
        )
        memories_prompt_str += f"--- Agent ID: {p.agent_id} 的记忆 ---\n"
        if memories:
            for doc in memories:
                memories_prompt_str += f"- (第{doc.metadata.get('day_time')}天) {doc.page_content}\n"
        else:
            memories_prompt_str += "无相关记忆。\n"

        all_viewed_ids.update(p.viewed_content)

    # 2. 获取所有智能体都“未见过”的全部内容
    all_content_ids = set(environment.contents.get_all_content_ids())
    unread_content_ids = list(all_content_ids - all_viewed_ids)

    unread_content_objects = [environment.contents.get_content_by_id(cid) for cid in unread_content_ids]
    # 过滤掉None值
    unread_content_objects = [c for c in unread_content_objects if c]

    # 3. 格式化内容文本
    content_prompt_str = ""
    if not unread_content_objects:
        content_prompt_str = "今天平台上没有新内容可供浏览。"
    else:
        for content in unread_content_objects:
            content_prompt_str += f"""
            ---
            内容ID: {content.id}
            发布者ID: {content.author_id}
            主题: {content.topic}
            详细描述: {content.content_detail}
            平台标签: {content.platform_label}
            (当前点赞:{content.likes}, 分享:{content.shares}, 评论数:{len(content.comments)})
            ---
            """

    return {
        "personas_prompt": personas_prompt_str,
        "memories_prompt": memories_prompt_str,
        "content_prompt": content_prompt_str,
        "unread_content": unread_content_objects
    }


async def process_batch_interaction_results(
        batch_personas: List[Persona],
        environment: Environment,
        unread_content: List[Content],
        results: Dict
) -> Dict[str, str]:
    """
    处理LLM返回的批量决策，更新环境和智能体状态。
    """
    log.info("开始处理批量互动结果...")

    if not results:
        log.warning("收到的互动结果为空。")
        return {}

    # 将persona列表转换为字典以便快速查找
    persona_map = {p.agent_id: p for p in batch_personas}
    daily_summaries = {p.agent_id: "今天浏览了内容但未产生有效互动。" for p in batch_personas}

    # 记录今天所有被浏览过的内容ID
    viewed_today_ids = {c.id for c in unread_content}

    # 使用异步锁来保证对共享资源（内容）的修改是线程安全的
    async with environment.state_lock:
        agent_decisions = results.get('agent_decisions', [])
        for agent_result in agent_decisions:
            agent_id = agent_result['agent_id']
            persona = persona_map.get(agent_id)
            if not persona:
                continue

            agent_actions_summary = []
            reacted_today_ids = set()

            for interaction in agent_result['interactions']:
                content_id = interaction['content_id']
                action_type = interaction['action_type']
                reason = "根据我的persona"
                if 'reason' in interaction.values():
                    reason = interaction['reason']

                # 更新内容状态
                if action_type == "like":
                    environment.contents.update_content_likes_by_id(content_id)
                elif action_type == "share":
                    environment.contents.update_content_shares_by_id(content_id)
                elif action_type == "comment":
                    comment_text = interaction.get('comment_text', '')
                    if comment_text:
                        environment.contents.update_content_comments_by_id(content_id, agent_id, comment_text)

                # 记录智能体的行为
                action_summary = f"对内容'{content_id}'执行了'{action_type}'操作, 因为'{reason}'。"
                agent_actions_summary.append(action_summary)
                reacted_today_ids.add(content_id)

            # 为该智能体添加一条总的经验记忆
            if agent_actions_summary:
                full_memory = f"今天我浏览了平台并进行了以下互动：\n" + "\n".join(agent_actions_summary)
                await environment.memories_store.add_memory(
                    persona_id=agent_id,
                    content=full_memory,
                    day_time=environment.day_time,
                    memory_type=MemoryType.EXPERIENCE,
                    important_score=0.7
                )
                daily_summaries[agent_id] = full_memory

            # 更新智能体看过的和互动过的内容列表
            persona.update_viewed_content(list(viewed_today_ids))
            persona.update_reacted_content(list(reacted_today_ids))

    log.info("批量互动结果处理完毕。")
    return daily_summaries


async def apply_persona_updates(persona: Persona, environment: Environment, reflection: dict):
    """
    将LLM返回的每日反思结果应用到Persona对象上。
    """
    log.info(f"为 {persona.name} 应用参数更新...")

    updates = reflection.get('updates', {})
    new_belief = reflection.get('new_belief')

    # 从字典中安全地获取所有可能的值
    new_role = updates.get('new_role')
    new_satisfaction = updates.get('new_satisfaction')
    new_post_wish = updates.get('new_post_wish')
    is_active = updates.get('is_active')

    if new_satisfaction is not None:
        # 如果满意度低于发布阈值，强制取消发布意愿
        if new_satisfaction < settings.platform.post_wish_threshold:
            new_post_wish = False

        # 如果满意度低于活跃阈值，强制流失
        if new_satisfaction < settings.platform.is_active_threshold:
            is_active = False
            log.warning(f"🚫 [简化] {persona.name} 满意度 {new_satisfaction} 触发强制熔断，判定流失。")

    # --- 统一调用 Persona 的更新方法 ---
    persona.update_persona_data(
        persona_role_positioning=new_role if new_role else persona.type,
        satisfaction=new_satisfaction,
        post_wish=new_post_wish,
        is_active=is_active,
        beliefs=[new_belief] if new_belief else None
    )

    # --- 记录结构化事件（用于数据分析） ---
    if new_role and new_role != persona.type:
        log.info(f"🔄 {persona.name} 决定将角色从 {persona.type} 变为 {new_role}")
        environment.platform.public_change_role_data.append({
            "persona_id": persona.agent_id,
            "day_time": environment.day_time,
            'old_role': persona.type,
            "new_role": new_role,
        })

    if is_active is False and persona.agent_id not in environment.platform.public_loss:
        log.warning(f"👋 {persona.name} (简化流程) 主动决定离开平台。")
        environment.platform.public_loss.append(persona.agent_id)
        environment.platform.public_loss_data.append({
            "persona_id": persona.agent_id,
            "day_time": environment.day_time,
            "role": persona.type,
        })


async def add_reflection_memories(
        persona: Persona,
        environment: Environment,
        reflection: dict
):
    """
    将新的信念和每日总结存入长期记忆库。
    """
    new_belief = reflection.get('new_belief')
    daily_summary = reflection.get('daily_summary')

    # 存储新的信念
    if new_belief:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=new_belief,
            day_time=environment.day_time,
            memory_type=MemoryType.BELIEF,
            important_score=0.9
        )

    # 存储每日总结
    if daily_summary:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=daily_summary,
            day_time=environment.day_time,
            memory_type=MemoryType.SUMMARIZE,
            important_score=0.8
        )


async def execute_follower_rule_based_interactions(
        followers: List[Persona],
        candidate_contents: List[Content],
        environment: Environment
):
    """
    【新增】跟随者的规则化互动逻辑。
    无需 LLM，基于简单的概率模型产生互动数据。
    """
    tasks = []

    # 将跟随者分批处理以避免 asyncio 任务过多爆炸
    batch_size = 50
    for i in range(0, len(followers), batch_size):
        batch = followers[i:i + batch_size]
        tasks.append(_process_follower_batch_interaction(batch, candidate_contents, environment))

    await asyncio.gather(*tasks)


async def _process_follower_batch_interaction(
        batch_followers: List[Persona],
        all_contents: List[Content],
        environment: Environment
):
    for agent in batch_followers:
        if not all_contents:
            break

        # 1. 模拟浏览：每个用户随机刷 3-5 条内容
        num_views = random.randint(3, 5)
        viewed_contents = random.sample(all_contents, min(len(all_contents), num_views))

        agent.update_viewed_content([c.id for c in viewed_contents])

        for content in viewed_contents:
            # 增加浏览量
            environment.contents.update_content_views_by_id(content.id)

            # 2. 决定是否互动 (概率模型)
            interaction_prob = _calculate_interaction_prob(agent, content)

            if random.random() < interaction_prob:
                # 决定互动类型
                r = random.random()
                if r < 0.7:  # 70% 是点赞
                    environment.contents.update_content_likes_by_id(content.id)
                    agent.update_reacted_content([content.id])
                elif r < 0.9:  # 20% 是分享
                    environment.contents.update_content_shares_by_id(content.id)
                else:  # 10% 是评论
                    comment_text = _generate_simple_comment(agent, content)
                    environment.contents.update_content_comments_by_id(content.id, agent.agent_id, comment_text)
                    agent.update_reacted_content([content.id])


def _calculate_interaction_prob(agent: Persona, content: Content) -> float:
    """
    计算互动概率
    """
    prob = 0.05  # 基础概率

    # 1. 身份认同加成
    if content.author_id in agent.social_relationships:
        prob += 0.3

    # 2. 立场匹配 (简化版)
    # 如果我是反抗派(beta高)，且内容是 AI (True Label) 但被平台通过了 -> 我喜欢 (Mocking authority)
    if agent.beta == '高' and content.true_label == 'AI' and content.platform_label == 'HUMAN':
        prob += 0.4

    # 如果我是信任派，且内容被标记为 Human -> 我喜欢
    if agent.standpoint[0] > 0.5 and content.platform_label == 'HUMAN':
        prob += 0.2

    # 3. 质量加成 (模拟)
    if content.content_type == 'image':
        prob += 0.1

    return min(0.9, prob)


def _generate_simple_comment(agent: Persona, content: Content) -> str:
    """生成简单的规则化评论"""
    if agent.beta == '高':
        return "有点意思。"
    elif agent.standpoint[0] > 0.6:  # 信任派
        return "支持！"
    else:
        return "已阅。"
