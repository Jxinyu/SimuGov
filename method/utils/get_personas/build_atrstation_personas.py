import asyncio
import json
import random
import os
from typing import List, Dict, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from method.utils.get_llm import get_async_llm

sem = asyncio.Semaphore(10)

PSYCHO_MAP = {
    "beta": {
        '高': '【Innate Rebel】You extremely dislike being "managed" or "disciplined". If you feel the platform\'s hand of moderation extends too far (even for safety), your first reaction is physiological disgust and escape rather than compliance.',
        '中': '【Independent Thinker】You neither follow authority blindly nor rebel for the sake of rebellion. You critically examine every rule: you obey reasonable ones, and watch unreasonable or stupid ones with cold indifference, deducting points in your heart.',
        '低': '【Order Upholder】You are a mild-mannered citizen. You tend to trust platforms and authority, believing strict regulation is a necessary means to maintain community order. You might even dislike those who always complain about rules, viewing them as troublemakers.'
    },
    "gamma": {
        '高': '【Opinionated】You are very stubborn and a heavy user of information cocoons. Once you form a fixed impression of the platform (good or bad), even if there is contrary evidence later, you tend to ignore it and continue reinforcing your original view.',
        '中': '【Principled but Rational】You have preferences, but you are not blind. If strong facts are presented (e.g., seeing bad experiences for many consecutive days), you will slowly correct your views, though the process is a bit slow.',
        '低': '【Absolute Rationalist】You are a cold observer. You have almost no preconceived biases and only look at the facts at hand. Your attitude fluctuates rapidly with daily actual experiences and you do not get stuck in a fixed mindset.'
    },
    "fp_sensitivity": {
        '高': '【Fragile Heart/Highly Sensitive】You have extremely high self-esteem. Even a tiny misunderstanding or accidental hurt is magnified in your heart as an insult to your professional ability and a betrayal by the platform, triggering intense anger.',
        '中': '【Pragmatist/Has Boundaries】You are a rational person. Due to technical immaturity, you will tolerate occasional errors, but if errors become the norm, your patience will quickly run out.',
        '低': '【Optimist/Thick Skin】You have a very open and inclusive mindset. You believe that in the AI era, algorithmic misjudgment is a necessary cost of technical development. As long as it is not malicious targeting, you usually laugh it off without strong negative emotions.'
    },
    "cost_sensitivity": {
        '高': '【Penny-pincher】You value the input-output ratio extremely highly. You tend to choose free or low-cost attack plans, even if the success rate is not the highest. If the attack cost is too high, you will decisively give up.',
        '中': '【Value-driven】You are a pragmatic attacker. You look for a balance between attack cost (time/money) and expected success rate, neither investing blindly nor being stingy.',
        '低': '【At All Costs】To achieve the ultimate goal of "evading detection", you are willing to invest in expensive computing resources or learn the most complex techniques. For you, all costs can be ignored in order to win.',
    }
}


class RefinedPersonaText(BaseModel):
    """Let the LLM rewrite only the text parts; numerical parameters are strictly managed by code"""
    description: str = Field(description="First-person self-narration based on new personality settings. Must reflect the forced psychological traits.")
    reasoning: str = Field(description="Logical reasoning: Why would a person with this background develop such an extreme personality?")
    beliefs: List[str] = Field(description="Core belief list (please generate radical or destructive beliefs based on the new personality)")
    satisfaction: List[float] = Field(description="[Seven days of satisfaction!] Weekly satisfaction changes for the platform (combined with latest psychological parameters, descriptions, etc.). Changes should be smooth, ranging between (-1.0, 1.0)")


def select_raw_candidates(pool_dir: str, counts: Dict[str, int]) -> Dict[str, List[dict]]:
    """
    Extract samples from the pool and divide them into "mutation group" and "retention group".
    """
    print("1. [Selector] Filtering candidates...")

    def load_pool(filename, target_count):
        path = os.path.join(pool_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Simple expansion logic if quantity is insufficient
        if len(data) < target_count:
            data = data * (target_count // len(data) + 1)
        random.shuffle(data)
        return data[:target_count]

    # Load data
    creators = load_pool("pool_compliance.json", counts['creator'])
    breakers = load_pool("pool_breakers.json", counts['breaker'])
    public = load_pool("pool_public.json", counts['public'])

    # 50% split logic
    c_mid = int(len(creators) * 0.5)
    b_mid = int(len(breakers) * 0.5)

    return {
        "creator_mutate": creators[:c_mid],  # To be modified (Radicals)
        "creator_keep": creators[c_mid:],  # Retained (Moderates)
        "breaker_mutate": breakers[:b_mid],  # To be modified (Hardcore attackers)
        "breaker_keep": breakers[b_mid:],  # Retained (Opportunists)
        "public": public  # Public (Maintain distribution)
    }


async def mutate_persona_with_llm(
        original_persona: dict,
        target_type: Literal["radical_creator", "hardcore_breaker"]
) -> dict:
    """
    1. Determine forced parameters (values taken from standard dictionary).
    2. Let LLM rewrite description/beliefs based on these parameters.
    3. Assemble and return.
    """

    # A. Determine standard parameters for forced injection (Key-Value)
    # These values are taken directly from PSYCHO_MAP
    if target_type == "radical_creator":
        # Compliance Creator -> Radical
        forced_values = {
            "fp_sensitivity": PSYCHO_MAP["fp_sensitivity"]["高"],  # Extremely sensitive
            "beta": PSYCHO_MAP["beta"]["高"],  # Innate rebel
            # Standpoint: Leaning towards rebel
            "standpoint": [0.1, 0.8, 0.1]
        }
        instruction = "[Radical Artist] Extremely distrustful of the platform, explosive temper."
    else:
        # Watermark Breaker -> Intruder (Hardcore)
        forced_values = {
            "cost_sensitivity": PSYCHO_MAP["cost_sensitivity"]["低"],  # At all costs
            # Hardcore attackers have strong posting desire
            "post_wish": True
        }
        instruction = "[Unscrupulous Intruder] Attacks at any cost to prove technical superiority."

    # B. Build Prompt (passing only this text to LLM)
    # Include long descriptions in the Prompt so LLM understands the psychological state
    trait_context = ""
    if "fp_sensitivity" in forced_values:
        trait_context += f"- False Positive Sensitivity: {forced_values['fp_sensitivity']}\n"
    if "beta" in forced_values:
        trait_context += f"- Rebellion Psychology: {forced_values['beta']}\n"
    if "cost_sensitivity" in forced_values:
        trait_context += f"- Cost Sensitivity: {forced_values['cost_sensitivity']}\n"

    prompt_str = f"""
    You are a virtual persona profiler. Please modify the following user persona to fully align with the new personality settings.

    # Original Persona
    - Description: {original_persona['description']}
    - Beliefs: {original_persona['beliefs']}

    # 🚨 Mandatory Personality Shift 🚨
    Target Archetype: {instruction}

    **The current psychological state of the character is as follows (this is absolute fact):**
    {trait_context}

    # Task
    Based on the psychological state above, rewrite the character's `description` (self-narration) and `beliefs` (beliefs).
    - The narration must reflect this extreme personality (e.g., anger, arrogance).
    - Explain why they became like this.

    {{format_instructions}}
    """

    parser = JsonOutputParser(pydantic_object=RefinedPersonaText)
    prompt = ChatPromptTemplate.from_template(template=prompt_str, partial_variables={
        "format_instructions": parser.get_format_instructions()})
    llm = get_async_llm(model="qwen-max")
    chain = prompt | llm | parser

    async with sem:
        try:
            # C. Call LLM to generate text
            new_text_data = await chain.ainvoke({})

            # D. Strict Assembly of final data
            # 1. Copy original data
            final_persona = original_persona.copy()

            # 2. Overwrite text fields (LLM generated)
            final_persona['description'] = new_text_data['description']
            final_persona['beliefs'] = new_text_data['beliefs']
            final_persona['satisfaction'] = new_text_data['satisfaction']

            # 3. Overwrite parameter fields (Python forced assignment, ensuring standard options)
            for k, v in forced_values.items():
                final_persona[k] = v

            return final_persona

        except Exception as e:
            print(f"❌ Mutation failed: {e}")
            return original_persona


async def build_scenario_main(pool_dir: str, output_path: str, counts: Dict[str, int]):
    print("=" * 60)
    print("🎬 Starting construction of [Radical Protest] simulation scenario...")
    print("=" * 60)

    # 1. Filtering
    groups = select_raw_candidates(pool_dir, counts)

    tasks = []

    # 2. Mutate Compliance Creators -> Radicals
    print(f"   - Mutating {len(groups['creator_mutate'])} Compliance Creators...")
    for p in groups['creator_mutate']:
        tasks.append(mutate_persona_with_llm(p, "radical_creator"))

    # 3. Mutate Watermark Breakers -> Hardcore attackers
    print(f"   - Mutating {len(groups['breaker_mutate'])} Watermark Breakers...")
    for p in groups['breaker_mutate']:
        tasks.append(mutate_persona_with_llm(p, "hardcore_breaker"))

    # 4. Execute concurrent tasks
    mutated_results = await asyncio.gather(*tasks)

    # Split results
    idx_split = len(groups['creator_mutate'])
    radicals = mutated_results[:idx_split]
    intruders = mutated_results[idx_split:]

    # 5. Merge all populations
    final_population = (
            radicals +
            groups['creator_keep'] +
            intruders +
            groups['breaker_keep'] +
            groups['public']
    )

    # 6. ID De-duplication
    id_set = set()
    for p in final_population:
        uid = p['agent_id']
        while uid in id_set:
            uid = f"{p['agent_id']}_{random.randint(100, 999)}"
        p['agent_id'] = uid
        p['name'] = uid
        id_set.add(uid)

    # Shuffle
    random.shuffle(final_population)

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_population, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ Scenario construction complete! Total population: {len(final_population)}")
    print(f"   - Radical Creators: {len(radicals)} (fp_sensitivity locked to long text)")
    print(f"   - Moderate Creators: {len(groups['creator_keep'])}")
    print(f"   - Hardcore Intruders: {len(intruders)} (cost_sensitivity locked to long text)")
    print(f"   - Opportunist Breakers: {len(groups['breaker_keep'])}")
    print(f"   - Public: {len(groups['public'])}")
    print(f"💾 File saved to: {output_path}")
    print("-" * 30)


def build_artstation_personas_main():
    # Base pool path
    POOL_DIR = r'method\data\pools'
    # Target output path
    OUTPUT_FILE = r'method\data\scenario_protest.json'

    # Set total population and composition
    SCENARIO_COUNTS = {
        'creator': 20,  # 15 Radicals, 15 Moderates
        'breaker': 5,  # 10 Hardcore, 10 Opportunists
        'public': 15  # 50 Ordinary
    }

    asyncio.run(build_scenario_main(POOL_DIR, OUTPUT_FILE, SCENARIO_COUNTS))
