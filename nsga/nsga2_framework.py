import asyncio
import copy
import logging
import random

import numpy as np
import pygmo as pg

from method.environment import Policy
from nsga.lhs import latin_hypercube_sampling
from method.simulation_main import simple, complete
from config import settings

log = logging.getLogger(__name__)

POLICY_PARAMS = {
    'f_penalty': {'type': 'continuous', 'bounds': (0.00, 1.00)},
    'ai_threshold': {'type': 'continuous', 'bounds': (0.00, 1.00)},
    'e_edu': {'type': 'discrete', 'values': ['Low', 'Medium', 'High']}
}

# --- Numerical precision and granularity control ---
GRANULARITY = 0.01
KPI_DECIMAL_PLACES = 5
SIMULATION_CONCURRENCY_LIMIT = asyncio.Semaphore(10)  # Concurrency control


def round_to_granularity(value, granularity=GRANULARITY):
    """Helper function to round values to a specified granularity."""
    result = round(value / granularity) * granularity
    return round(result, 5)


def calculate_hypervolume(front, ref_point):
    """Calculate the hypervolume of a given front relative to a reference point (ref_point)."""
    kpi_vectors = [list(ind['kpi'].values()) for ind in front]
    if not kpi_vectors:
        return 0.0
    hv = pg.hypervolume(kpi_vectors)
    return hv.compute(ref_point)


def remove_duplicates(population):
    """
    Used to prevent the elite solution set from being occupied by numerically identical clones.
    """
    unique_pop = []
    # Use set to record existing KPI combinations (safety, creativity, satisfaction)
    seen_kpis = set()

    for ind in population:
        # Must ensure ind['kpi'] has been calculated
        if not ind.get('kpi'):
            continue

        # Convert KPI to tuple for hash deduplication
        kpi_tuple = (
            round(ind['kpi']['safety'], KPI_DECIMAL_PLACES),
            round(ind['kpi']['creativity'], KPI_DECIMAL_PLACES),
            round(ind['kpi']['satisfaction'], KPI_DECIMAL_PLACES)
        )

        if kpi_tuple not in seen_kpis:
            seen_kpis.add(kpi_tuple)
            unique_pop.append(ind)

    return unique_pop


def generate_unique_refill(target_count, current_population, evaluated_cache):
    """
    Adopts a "rejection sampling" mechanism to ensure new individuals are also unique in policy parameters (input side).
    """
    new_individuals = []

    # 1. Establish fingerprint database: record existing policy combinations in the current population
    existing_signatures = set()

    for ind in current_population:
        p = ind['policy']
        sig = (
            round_to_granularity(p['f_penalty']),
            round_to_granularity(p['ai_threshold']),
            str(p['e_edu'])
        )
        existing_signatures.add(sig)

    # 2. Loop generation until target quantity is reached
    max_attempts = target_count * 20  # Prevent infinite loop
    attempts = 0

    while len(new_individuals) < target_count and attempts < max_attempts:
        attempts += 1

        # Generate a batch each time
        batch_size = target_count - len(new_individuals) + 5
        candidates = latin_hypercube_sampling(batch_size)

        for policy_raw in candidates:
            # A. Normalize parameters (granularity alignment)
            policy_clean = {
                'f_penalty': round_to_granularity(policy_raw['f_penalty']),
                'ai_threshold': round_to_granularity(policy_raw['ai_threshold']),
                'e_edu': str(policy_raw['e_edu'])
            }

            # B. Generate fingerprint
            sig = (policy_clean['f_penalty'], policy_clean['ai_threshold'], policy_clean['e_edu'])

            # C. Deduplication check
            if sig not in existing_signatures:
                existing_signatures.add(sig)

                # Check if global cache already exists (if cached, use directly; if not, leave empty for evaluation)
                kpi_data = {}
                policy_key_tuple = tuple(sorted(policy_clean.items()))
                if policy_key_tuple in evaluated_cache:
                    kpi_data = copy.deepcopy(evaluated_cache[policy_key_tuple])

                new_individuals.append({'policy': policy_clean, 'kpi': kpi_data})

                if len(new_individuals) >= target_count:
                    break

    if len(new_individuals) < target_count:
        log.warning(
            f"⚠️ Warning: Parameter space crowded, attempted {max_attempts} times and only generated {len(new_individuals)}/{target_count} unique new solutions.")

    return new_individuals


def calculate_stable_score(kpi_list: list, penalty_weight: float = 1.0) -> float:
    """Calculate comprehensive score considering stability. Mean - Weight * StdDev"""
    if not kpi_list:
        return 0.0
    data = np.array(kpi_list)
    mean_val = np.mean(data)
    std_val = np.std(data)
    final_score = float(mean_val - (penalty_weight * std_val))
    return float(max(0.0, final_score))


def calculate_theta_jitter(theta_history: list) -> float:
    """Calculate the jitter degree of Theta (policy)."""
    if not theta_history or len(theta_history) < 2:
        return 0.0
    diffs = [abs(theta_history[i] - theta_history[i - 1]) for i in range(1, len(theta_history))]
    return float(np.mean(diffs))


async def evaluate_policy(policy: dict):
    """
    Input: A set of policy parameters
    Output: Evaluation results after volatility and policy jitter penalties (NSGA-II objective minimization, so negative values are taken)
    """
    policy_obj = Policy(policy['ai_threshold'], policy['f_penalty'], policy['e_edu'])

    try:
        if settings.platform.efficiency_validation:
            kpi_results = await complete(policy_obj)
        else:
            kpi_results = await simple(policy_obj)

        s_list = kpi_results.get('safety', [])
        c_list = kpi_results.get('creativity', [])
        sat_list = kpi_results.get('satisfaction', [])
        theta_list = kpi_results.get('theta', [])

        # Penalty weight configuration
        w_safety = 1.0
        w_creativity = 1.0
        w_satisfaction = 0.8

        safety_base = calculate_stable_score(s_list, w_safety)
        creativity_base = calculate_stable_score(c_list, w_creativity)
        satisfaction_base = calculate_stable_score(sat_list, w_satisfaction)

        # Policy jitter penalty
        jitter_penalty = calculate_theta_jitter(theta_list) * 2.0

        final_safety = safety_base - jitter_penalty
        final_creativity = creativity_base - jitter_penalty
        final_satisfaction = satisfaction_base - jitter_penalty

        # NSGA-II default minimization, take negative numbers
        return {
            'safety': round(-final_safety, KPI_DECIMAL_PLACES),
            'creativity': round(-final_creativity, KPI_DECIMAL_PLACES),
            'satisfaction': round(-final_satisfaction, KPI_DECIMAL_PLACES)
        }

    except Exception as e:
        log.warning(f"Error occurred when evaluating policy {policy}: {e}. Returning worst fitness.")
        return {'safety': 0.0, 'creativity': 0.0, 'satisfaction': 0.0}


async def evaluate_population_with_cache(population: list, evaluated_cache: dict):
    """Asynchronous evaluation function with caching mechanism."""
    tasks = []
    indices_to_run = []

    for i, ind in enumerate(population):
        # If KPI already exists (e.g., generated from cache), skip
        if ind.get('kpi'):
            continue

        policy_key = tuple(sorted(ind['policy'].items()))

        if policy_key in evaluated_cache:
            ind['kpi'] = copy.deepcopy(evaluated_cache[policy_key])
        else:
            indices_to_run.append(i)

            async def limited_evaluate(policy_data):
                async with SIMULATION_CONCURRENCY_LIMIT:
                    return await evaluate_policy(policy_data)

            tasks.append(limited_evaluate(ind['policy']))

    if tasks:
        log.info(f"    -> Simulations needed for this batch: {len(tasks)} (Cache hits: {len(population) - len(tasks)})")
        results = await asyncio.gather(*tasks)

        for idx_in_pop, kpi_result in zip(indices_to_run, results):
            population[idx_in_pop]['kpi'] = kpi_result
            policy_key = tuple(sorted(population[idx_in_pop]['policy'].items()))
            evaluated_cache[policy_key] = kpi_result


def non_dominated_sort(population):
    """Fast non-dominated sort"""
    for ind in population:
        ind['dominates'] = []
        ind['dominated_by'] = 0
        ind['rank'] = 0
        ind['crowding_distance'] = 0

    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p, q):
                p['dominates'].append(q)
            elif dominates(q, p):
                p['dominated_by'] += 1
        if p['dominated_by'] == 0:
            p['rank'] = 1
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates']:
                q['dominated_by'] -= 1
                if q['dominated_by'] == 0:
                    q['rank'] = i + 2
                    next_front.append(q)
        if not next_front: break
        fronts.append(next_front)
        i += 1

    # Crowding distance calculation
    for front in fronts:
        if not front: continue
        kpi_keys = list(front[0]['kpi'].keys())
        for obj in kpi_keys:
            front.sort(key=lambda x: x['kpi'][obj])
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')
            if len(front) > 2:
                min_obj = front[0]['kpi'][obj]
                max_obj = front[-1]['kpi'][obj]
                if max_obj == min_obj: continue
                for j in range(1, len(front) - 1):
                    front[j]['crowding_distance'] += (front[j + 1]['kpi'][obj] - front[j - 1]['kpi'][obj]) / (
                            max_obj - min_obj)
    return population


def dominates(ind1, ind2):
    """Determine dominance relationship (smaller value is better)"""
    is_better = all(ind1['kpi'][obj] <= ind2['kpi'][obj] for obj in ind1['kpi'])
    is_strictly_better = any(ind1['kpi'][obj] < ind2['kpi'][obj] for obj in ind1['kpi'])
    return is_better and is_strictly_better


def crossover(parent1, parent2):
    child1_policy, child2_policy = {}, {}
    eta_c = 20
    for param, config in POLICY_PARAMS.items():
        p1_val, p2_val = parent1['policy'][param], parent2['policy'][param]
        if config['type'] == 'discrete':
            if random.random() < 0.5:
                child1_policy[param] = p1_val
                child2_policy[param] = p2_val
            else:
                child1_policy[param] = p2_val
                child2_policy[param] = p1_val
        else:
            rand = random.random()
            if rand <= 0.5:
                beta = (2 * rand) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
            c1 = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
            c2 = 0.5 * ((1 - beta) * p1_val + (1 + beta) * p2_val)
            bounds = config['bounds']
            child1_policy[param] = round_to_granularity(min(max(c1, bounds[0]), bounds[1]))
            child2_policy[param] = round_to_granularity(min(max(c2, bounds[0]), bounds[1]))
    return {'policy': child1_policy, 'kpi': {}}, {'policy': child2_policy, 'kpi': {}}


def mutate(individual):
    mutated_policy = individual['policy'].copy()
    eta_m = 20
    for param, config in POLICY_PARAMS.items():
        if random.random() < 1.0 / len(mutated_policy):
            if config['type'] == 'discrete':
                options = [v for v in config['values'] if v != mutated_policy[param]]
                if options:
                    mutated_policy[param] = random.choice(options)
            else:
                val = mutated_policy[param]
                low, high = config['bounds']
                rand = random.random()
                if rand < 0.5:
                    delta = (2 * rand) ** (1 / (eta_m + 1)) - 1
                    val += delta * (val - low)
                else:
                    delta = 1 - (2 * (1 - rand)) ** (1 / (eta_m + 1))
                    val += delta * (high - val)
                mutated_policy[param] = round_to_granularity(min(max(val, low), high))
    return {'policy': mutated_policy, 'kpi': {}}


def selection(population):
    p1 = random.choice(population)
    p2 = random.choice(population)
    if p1['rank'] < p2['rank']:
        return p1
    elif p2['rank'] < p1['rank']:
        return p2
    else:
        return p1 if p1['crowding_distance'] > p2['crowding_distance'] else p2


def select_final_elites(population, target_count=None):
    """Screen final elite solutions (Rank first, then crowding distance)"""
    pop_sorted = non_dominated_sort(population)
    if not target_count:
        return [ind for ind in pop_sorted if ind['rank'] == 1]

    final_elites = []
    fronts = {}
    for ind in pop_sorted:
        r = ind['rank']
        if r not in fronts: fronts[r] = []
        fronts[r].append(ind)

    rank_idx = 1
    while len(final_elites) < target_count and rank_idx in fronts:
        current_front = fronts[rank_idx]
        missing = target_count - len(final_elites)
        if len(current_front) <= missing:
            final_elites.extend(current_front)
        else:
            current_front.sort(key=lambda x: x['crowding_distance'], reverse=True)
            final_elites.extend(current_front[:missing])
        rank_idx += 1
    return final_elites


async def run_nsga2(population_size=20, generations=10,
                    convergence_patience=4, convergence_threshold=0.01, target_elite_count=None):
    """
    Main function to run NSGA-II algorithm (Revised version: data saving sequence + output logic)
    """
    evaluated_cache = {}

    log.info("Initializing population (LHS)...")
    initial_policies = latin_hypercube_sampling(n_samples=population_size)
    for p in initial_policies:
        for k, cfg in POLICY_PARAMS.items():
            if cfg['type'] == 'continuous':
                p[k] = round_to_granularity(p[k])

    population = [{'policy': p, 'kpi': {}} for p in initial_policies]
    all_generations_data = []

    log.info("Evaluating initial population...")
    await evaluate_population_with_cache(population, evaluated_cache)
    population = non_dominated_sort(population)
    all_generations_data.append(copy.deepcopy(population))

    hv_history = []
    generations_without_improvement = 0
    ref_point = [0.0, 0.0, 0.0]

    for gen in range(generations):
        log.info(f"\n=== Generation {gen + 1}/{generations} evolution ===")

        # 1. Reproduction
        offspring = []
        while len(offspring) < population_size:
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            offspring.append(mutate(c1))
            if len(offspring) < population_size:
                offspring.append(mutate(c2))

        # 2. Evaluate offspring
        await evaluate_population_with_cache(offspring, evaluated_cache)

        # 3. Merge
        combined_population = population + offspring

        # 4. Deduplication (based on KPI, prevent population involution)
        unique_population = remove_duplicates(combined_population)

        # 5. Refill (based on Policy, introduce new exploration)
        target_pool_size = max(population_size, int(population_size * 1.5))
        if len(unique_population) < target_pool_size:
            fill_count = target_pool_size - len(unique_population)
            # log.info(f"🧬 Injecting new blood: supplementing {fill_count} unique random individuals...")
            new_inds = generate_unique_refill(fill_count, unique_population, evaluated_cache)
            inds_to_eval = [ind for ind in new_inds if not ind.get('kpi')]
            if inds_to_eval:
                await evaluate_population_with_cache(inds_to_eval, evaluated_cache)
            unique_population.extend(new_inds)

        combined_population = unique_population

        # 6. Sorting and selection
        combined_population = non_dominated_sort(combined_population)
        new_population = []
        fronts_dict = {}
        for ind in combined_population:
            r = ind['rank']
            if r not in fronts_dict: fronts_dict[r] = []
            fronts_dict[r].append(ind)

        for r in sorted(fronts_dict.keys()):
            front = fronts_dict[r]
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                needed = population_size - len(new_population)
                new_population.extend(front[:needed])
                break

        population = new_population

        # 7. Convergence monitoring and data recording
        best_front = [ind for ind in population if ind['rank'] == 1]
        current_hv = calculate_hypervolume(best_front, ref_point)
        hv_history.append(current_hv)

        log.info(f"Current HV: {current_hv:.6f}, Rank 1 quantity: {len(best_front)}")

        population_to_save = copy.deepcopy(population)
        all_generations_data.append(population_to_save)

        # Convergence determination moved after saving
        if len(hv_history) > 1:
            prev = hv_history[-2]
            imp = (current_hv - prev) / abs(prev) if abs(prev) > 1e-9 else current_hv
            if imp < convergence_threshold:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0

            if generations_without_improvement >= convergence_patience:
                log.info(f"🚀 Algorithm has converged (no improvement for {convergence_patience} consecutive generations), stopping evolution.")
                break

    # Final output
    final_set = select_final_elites(population, target_count=target_elite_count)
    log.info(f"Final output elite policy count: {len(final_set)}")

    return final_set, all_generations_data
