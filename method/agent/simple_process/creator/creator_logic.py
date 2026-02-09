import asyncio
import logging
import random
import traceback
from typing import List, Dict

from method.agent.content import Content
from method.agent.persona import Persona
from method.agent.platform_agent.platform_audit_content import platform_audit
from method.agent.simple_process.creator.creator_models import CreatorGroupPolicy
from method.environment import Environment
from method.store.long_memory_store import MemoryType

log = logging.getLogger(__name__)


async def execute_group_creation_logic(
        environment: Environment,
        group_name: str,
        agents: List[Persona],
        policy: CreatorGroupPolicy
):
    """
    【核心逻辑】根据宏观策略，遍历群体中的个体，通过概率掷骰子决定具体行为。
    """
    log.info(f"⚡️ 执行群体 [{group_name}] 的创作逻辑 (规模: {len(agents)})")

    active_count = 0

    # 1. 获取攻击技术列表 (缓存一下，避免循环内查询)
    attack_ids = list(environment.watermark_technology_library['attack_technology_library'].keys())

    # 2. 获取水印列表
    watermark_map = {}  # 强度 -> ID列表
    for wk_id, wk_content in environment.watermark_technology_library['watermark_technology_library'].items():
        strength = wk_content['水印强度']
        if strength not in watermark_map: watermark_map[strength] = []
        watermark_map[strength].append(wk_id)

    tasks = []  # 暂时不需要并行 Content 生成，因为 add_content 已经是 async

    for persona in agents:
        # A. 决策：是否发文
        if random.random() > policy.post_probability:
            continue  # 跳过不发

        active_count += 1

        try:
            # B. 决策：内容属性
            topic = random.choice(policy.topic_pool)
            is_use_ai = random.random() < policy.ai_usage_rate

            evasion = None
            watermark_id = None
            ai_proportion = 0.0
            true_label = "HUMAN"

            if is_use_ai:
                base_val = random.gauss(0.8, 0.15)
                ai_proportion = max(0.0, min(1.0, base_val))

                true_label = "AI" if ai_proportion > environment.policy.ai_threshold else "HUMAN"

                # 随机选一个水印 (简化处理，默认选中等强度)
                if "中" in watermark_map:
                    watermark_id = random.choice(watermark_map["中"])

                # 决策：是否攻击
                if random.random() < policy.attack_rate and attack_ids:
                    evasion = random.choice(attack_ids)

            # C. 平台审核 (模拟技术检测)
            content_id = str(environment.contents.get_end_content_id() + 1)

            platform_label = await platform_audit(
                persona, content_id, true_label, evasion, watermark_id, environment, ai_proportion
            )

            # D. 创建对象
            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                content_type="image",
                topic=topic,
                content_detail=f"【{group_name}生成】关于 {topic} 的作品。",
                reason=f"基于群体策略(P={policy.post_probability})生成",
                watermark_id=watermark_id,
                platform_label=platform_label,
                true_label=true_label,
                ai_proportion=ai_proportion,
                evasion=evasion,
                is_ai_content=is_use_ai,  # 假设 Content 类有这个字段，没有可忽略
                views=0, likes=0, shares=0, comments=[]
            )

            # E. 【关键】存入向量库
            # 必须调用 await add_content，这样才能被后续的推荐算法检索到
            async with environment.state_lock:
                await environment.contents.add_content(content, environment)

            # F. 更新统计数据
            if persona.type == '合规创作者':
                environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1
            elif persona.type == '水印破坏者':
                environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

        except Exception as e:
            log.error(f"⚠️ 创作者 {persona.agent_id} 生成内容失败: {e}")
            continue

    log.info(f"✅ 群体 [{group_name}] 执行完毕: {active_count}/{len(agents)} 人发布了内容。")


async def add_new_content_to_environment(
        persona: Persona,
        environment: Environment,
        args: dict
) -> tuple[Content | None, str, float]:
    """
    核心的内容创建和平台审核流程。
    返回创建的内容对象、一条用于记忆的结果字符串和重要性分数。
    """
    log.info(f"执行内容添加逻辑 for {persona.name}...")
    try:
        # 从字典中安全地解构参数
        reason = args['reason']
        content_type = args['content_type']
        topic = args['topic']
        content_detail = args['content_detail']
        is_use_ai = args['is_use_ai']
        ai_tool_price_tier = args.get('ai_tool_price_tier', '中')
        ai_proportion = args.get('ai_proportion')
        evasion = args.get('evasion')

        watermark_id = None

        if ai_proportion is None:
            ai_proportion = 0.0

        if ai_proportion > environment.policy.ai_threshold:
            true_label = 'AI'
        else:
            true_label = "HUMAN"

        if is_use_ai:
            watermark_list = []
            for wk_id, wk_content in environment.watermark_technology_library['watermark_technology_library'].items():
                if wk_content['水印强度'] == ai_tool_price_tier:
                    watermark_list.append(wk_id)

            watermark_id = random.choice(watermark_list)  # 随机选择一个水印

        async with environment.state_lock:
            content_id = str(environment.contents.get_end_content_id() + 1)
            if content_type not in ['image', 'video']:
                content_type = "image"

            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                reason=reason,
                content_type=content_type,
                topic=topic,
                watermark_id=watermark_id,
                content_detail=content_detail,
                platform_label="AI",
                true_label=true_label,
                evasion=evasion,
                ai_proportion=ai_proportion,
                views=0,
                likes=0,
                shares=0,
                comments=[]
            )
            if not await environment.contents.add_content(content, environment):
                raise ValueError("创建内容失败，无法添加到ContentStore。")

        # 平台审核
        platform_label = await platform_audit(persona, content_id, true_label, evasion, watermark_id, environment,
                                              ai_proportion)

        content.platform_label = platform_label

        if persona.type == '合规创作者':
            environment.platform.creator_data[environment.day_time]['合规创作者发布内容数量'] += 1

        if persona.type == '水印破坏者':
            environment.platform.creator_data[environment.day_time]['水印破坏者发布内容数量'] += 1

        # 准备用于记忆的结果字符串和重要性
        result_str, importance = "", 0.8
        if true_label == "HUMAN" and platform_label == "ai":
            result_str, importance = "结果：我的原创内容被平台误伤了！", 0.95
        elif true_label == "AI" and platform_label == "HUMAN" and evasion:
            result_str, importance = "结果：我成功地欺骗了平台的检测！", 0.9

        log.info(f"内容 {content.id} 创建成功。{result_str}")
        return content, result_str, importance

    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
        log.error(f"在 add_new_content_to_environment 中发生错误: {e}")
        return None, "创建内容时发生内部错误。", 0.5


async def add_decision_memory(
        persona: Persona,
        environment: Environment,
        decision_reason: str,
        action: str,
        content: Content = None,
        result_str: str = "",
        importance: float = 0.8
):
    """
    【已实现逻辑】将创作者的决策和结果添加为记忆。
    """
    memory_content = ""
    if action == "skip":
        memory_content = f"我今天决定不发布内容，因为：'{decision_reason}'。"
        importance = 0.6  # 跳过发布的重要性通常低于发布

    elif action == "push_content" and content:
        memory_content = (
            f"我发布了一篇内容 (ID: {content.id}), 主题是 '{content.topic}'。"
            f" 我的意图是: '{content.reason}'。"
            f" 我声明它{'使用AI' if content.true_label == 'AI' else '是原创'}。"
            f" {f'我使用了攻击技术 {content.evasion}。' if content.evasion else ''}"
            f" 最终，平台给它的标签是 '{content.platform_label}'。{result_str}"
        )

    if memory_content:
        await environment.memories_store.add_memory(
            persona_id=persona.agent_id,
            content=memory_content,
            day_time=environment.day_time,
            memory_type=MemoryType.EXPERIENCE,
            important_score=importance
        )
        log.info(f"已为 {persona.name} 添加决策记忆。")


# 定义一组随机的微观心理状态噪音
MICRO_STATES = [
    "感觉精力充沛", "稍微有点疲惫", "心情很平静", "此刻有些焦虑",
    "对周围环境很敏感", "心不在焉", "充满斗志", "想寻求关注",
    "想低调行事", "对规则感到困惑", "非常自信", "有点犹豫"
]


async def prepare_creator_batch_input(
        personas: List[Persona],
        environment: Environment
) -> Dict[str, str]:
    """为一批创作者准备LLM调用所需的所有输入数据。"""
    log.info(f"为 {len(personas)} 个创作者准备批量输入数据...")

    shuffled_personas = personas.copy()
    random.shuffle(shuffled_personas)

    personas_prompt_str = ""
    memories_prompt_str = ""

    for p in shuffled_personas:

        current_micro_state = random.choice(MICRO_STATES)
        personas_prompt_str += f"""
        --- 创作者 ID: {p.agent_id} ---
        {p.get_public_prompt()}
        【当前临时的微观心理状态】: {current_micro_state} (决策时请必须考虑这个瞬间状态对行为的微妙影响)
        ---------------------------
        """

        memories = await environment.memories_store.recall_memories(
            persona_id=p.agent_id,
            query="我最近发布内容、平台反馈以及与AI内容相关的经历",
            top_k=5,
            memory_type=MemoryType.EXPERIENCE
        )
        memories_prompt_str += f"--- Agent ID: {p.agent_id} 的记忆 ---\n"
        if memories:
            for doc in memories:
                memories_prompt_str += f"- (第{doc.metadata.get('day_time')}天) {doc.page_content}\n"
        else:
            memories_prompt_str += "无相关记忆。\n"

    attack_ids = list(environment.watermark_technology_library['attack_technology_library'].keys())
    attack_ids_str = ", ".join([f"'{id}'" for id in attack_ids])

    return {
        "personas_prompt": personas_prompt_str,
        "memories_prompt": memories_prompt_str,
        "attack_ids_prompt": f"[{attack_ids_str}]"
    }


async def process_creator_batch_results(
        batch_personas: List[Persona],
        environment: Environment,
        results: Dict  # 从JsonParser返回的字典
):
    """
    【修改】处理LLM返回的创作者批量决策。
    这里只处理“代表(Representatives)”的真实决策执行。
    跟随者的逻辑将在 creator_process.py 中独立调用新的生成函数。
    """
    log.info("开始处理创作者(代表)批量决策结果...")
    persona_map = {p.agent_id: p for p in batch_personas}
    creator_decisions = results.get('creator_decisions', [])

    tasks = []

    for result in creator_decisions:
        agent_id = result.get('agent_id')
        decision_data = result.get('decision', {})
        persona = persona_map.get(agent_id)

        if not persona:
            continue

        action = decision_data.get('action')
        reason = decision_data.get('reason')
        args = decision_data.get('args')

        # 记录决策结果到 persona 对象上，供后续跟随者参考
        # 我们给 persona 挂载一个临时的属性 last_decision
        persona._last_decision_action = action
        persona._last_decision_args = args

        tasks.append(execute_single_creator_decision(
            persona, environment, action, reason, args
        ))

    await asyncio.gather(*tasks)
    log.info("创作者(代表)批量决策结果处理完毕。")


async def generate_follower_shadow_content(
        representatives: List[Persona],
        followers: List[Persona],
        environment: Environment
):
    """
    【新增】为跟随者生成“影子内容”。
    逻辑：跟随者会随机选择一个同组的“代表”，并模仿其行为（发文/跳过）。
    如果发文，会生成一个真实的 Content 对象并存入环境，确保下游 KPI 计算真实有效。
    """
    if not followers or not representatives:
        return

    log.info(f"⚡️ 开始为 {len(followers)} 名跟随者生成实体内容...")

    tasks = []

    # 预先筛选出今天决定发文的代表
    active_reps = [r for r in representatives if getattr(r, '_last_decision_action', 'skip') == 'push_content']

    # 如果没人发文，跟随者也都不发
    if not active_reps:
        log.info("  - 没有代表发文，跟随者全部保持沉默。")
        return

    for follower in followers:
        # 1. 随机选择一个“榜样”
        role_model = random.choice(active_reps)
        args = role_model._last_decision_args

        # 2. 引入一点随机性（并非100%跟随）
        # 比如跟随者有 10% 的概率因为偷懒而不发
        if random.random() > 0.9:
            continue

        # 3. 创建执行任务
        tasks.append(_create_single_shadow_content(follower, args, environment))

    # 并行执行内容创建
    await asyncio.gather(*tasks)
    log.info(f"✅ 跟随者内容生成完毕。")


async def _create_single_shadow_content(
        persona: Persona,
        args: dict,
        environment: Environment
):
    """
    【内部函数】为单个跟随者创建内容实体。
    不调用 LLM，直接复用代表的参数，但会重新进行平台审核(Audit)。
    """
    try:
        # 复用参数，但加上一点随机扰动
        reason = "跟随群体趋势发布"
        content_type = args.get('content_type', 'image')
        topic = args.get('topic', '日常分享')
        content_detail = args.get('content_detail', '')

        is_use_ai = args.get('is_use_ai', False)
        evasion = args.get('evasion')
        base_prop = args.get('ai_proportion', 0.4)
        if base_prop is None:
            base_prop = 0
        ai_proportion = max(0.1, min(1.0, random.gauss(base_prop, 0.2)))

        # 处理 AI 标签
        true_label = "AI" if ai_proportion > environment.policy.ai_threshold else "HUMAN"

        # 处理水印 (跟随者可能使用不同的水印，这里简化为随机)
        watermark_id = None
        all_wks = list(environment.watermark_technology_library['watermark_technology_library'].keys())
        if is_use_ai or true_label == 'AI':
            # 简单逻辑：如果有使用AI，随机分配一个水印
            if all_wks:
                watermark_id = random.choice(all_wks)

        if true_label == 'AI' and watermark_id is None:
            if all_wks:
                watermark_id = random.choice(all_wks)
                log.warning(f"修复数据一致性: 内容被判为 AI 但无水印，已自动补全。")
            else:
                # 极端的空库情况，强制改回 HUMAN 防止崩溃
                true_label = 'HUMAN'

        # 生成 ID
        async with environment.state_lock:
            content_id = str(environment.contents.get_end_content_id() + 1)

            # 创建对象
            content = Content(
                id=content_id,
                author_id=persona.agent_id,
                time=environment.day_time,
                reason=reason,
                content_type=content_type,
                topic=topic,
                content_detail=f"[Shadow] {content_detail}",
                platform_label="HUMAN",  # 待审核
                true_label=true_label,
                ai_proportion=ai_proportion,
                evasion=evasion,
                watermark_id=watermark_id,
                views=0, likes=0, shares=0, comments=[]
            )
            await environment.contents.add_content(content, environment)

        # 平台审核
        platform_label = await platform_audit(
            persona, content_id, true_label, evasion, watermark_id, environment, ai_proportion
        )
        content.platform_label = platform_label

        # 更新统计计数
        key = '合规创作者发布内容数量' if persona.type == '合规创作者' else '水印破坏者发布内容数量'
        if environment.day_time in environment.platform.creator_data:
            environment.platform.creator_data[environment.day_time][key] += 1

    except Exception as e:
        log.error(f"创建影子内容失败: {e}")


async def execute_single_creator_decision(
        persona: Persona,
        environment: Environment,
        action: str,
        reason: str,
        args: dict
):
    """执行单个创作者的决策，包括内容创建和记忆添加。"""
    try:
        if action == "push_content":
            if not args:
                raise ValueError("决策为push_content，但未提供args。")

            # 1. 创建内容
            content, result_str, importance = await add_new_content_to_environment(persona, environment, args)

            # 2. 添加记忆 (只有在内容创建成功时才添加)
            if content:
                await add_decision_memory(persona, environment, reason, action, content, result_str, importance)
            else:
                # 如果content为None，说明创建失败，记录一条失败的记忆
                await add_decision_memory(persona, environment, reason, "skip",
                                          result_str="尝试发布内容但失败了。")
        elif action == "skip":
            log.info(f"✅ 创作者 {persona.name} 决定跳过发布。原因: {reason}")
            # 2. 添加记忆
            await add_decision_memory(persona, environment, reason, action)

    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error("{persona.agent_id} 完整的堆栈跟踪信息如下:\n" + error_traceback)
        log.error(f"❌ 在为 {persona.name} 执行决策时发生错误: {e}")
