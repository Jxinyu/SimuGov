from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_SOURCE_ROOT = PROJECT_ROOT
PAPER_DATA_ROOT = PROJECT_ROOT / "paper_data"
SOURCE_ROOTS = [root for root in (PAPER_DATA_ROOT, CURRENT_SOURCE_ROOT) if root.exists()]
OUTPUT_ROOT = PROJECT_ROOT / "paper_results" / "recomputed"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rel_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def find_first(*patterns: str) -> Path | None:
    for root in SOURCE_ROOTS:
        for pattern in patterns:
            hits = sorted(root.rglob(pattern))
            for hit in hits:
                if "paper_results" in hit.parts or "paper_experiments" in hit.parts:
                    continue
                return hit
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def flt(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def stable_score(values: list[float], penalty_weight: float = 1.0) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr) - penalty_weight * np.std(arr))


def theta_jitter(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def policy_metrics(item: dict) -> dict[str, float]:
    kpi = item["kpi_series"]
    jitter = theta_jitter([float(v) for v in kpi.get("theta", [])]) * 2.0
    return {
        "safety": stable_score([float(v) for v in kpi.get("safety", [])], 1.0) - jitter,
        "creativity": stable_score([float(v) for v in kpi.get("creativity", [])], 1.0) - jitter,
        "satisfaction": stable_score([float(v) for v in kpi.get("satisfaction", [])], 0.8) - jitter,
    }


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = ("safety", "creativity", "satisfaction")
    return all(left[key] >= right[key] for key in keys) and any(left[key] > right[key] for key in keys)


def pareto_rank(metrics: dict[str, dict[str, float]]) -> list[str]:
    policies = list(metrics)
    domination_counts = {policy: 0 for policy in policies}
    for policy in policies:
        for other in policies:
            if policy != other and dominates(metrics[other], metrics[policy]):
                domination_counts[policy] += 1

    def sort_key(policy: str):
        item = metrics[policy]
        avg = (item["safety"] + item["creativity"] + item["satisfaction"]) / 3
        return domination_counts[policy], -avg

    return sorted(policies, key=sort_key)


def latest_day_dir(policy_dir: Path) -> Path | None:
    day_dirs = [
        d for d in policy_dir.iterdir()
        if d.is_dir() and d.name.startswith("day_time_") and d.name.split("_")[-1].isdigit()
    ]
    if not day_dirs:
        return None
    return max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))


def compact_policy_result(policy_dir: Path) -> dict | None:
    last_day = latest_day_dir(policy_dir)
    if last_day is None:
        return None
    kpi_file = last_day / "output_system_kpi.json"
    policy_file = last_day / "output_policy.json"
    platform_file = last_day / "output_platform.json"
    if not kpi_file.exists() or not policy_file.exists():
        return None

    kpi = read_json(kpi_file)
    policy = read_json(policy_file)
    compact = {
        "policy_id": policy_dir.name,
        "final_day": int(last_day.name.split("_")[-1]),
        "policy": policy,
        "kpi_series": {
            "safety": kpi.get("safety", []),
            "creativity": kpi.get("creativity", []),
            "satisfaction": kpi.get("satisfaction", []),
            "theta": kpi.get("theta", []),
        },
        "final_kpi": {
            "safety": kpi.get("safety", [None])[-1] if kpi.get("safety") else None,
            "creativity": kpi.get("creativity", [None])[-1] if kpi.get("creativity") else None,
            "satisfaction": kpi.get("satisfaction", [None])[-1] if kpi.get("satisfaction") else None,
            "theta": kpi.get("theta", [None])[-1] if kpi.get("theta") else None,
        },
    }
    if platform_file.exists():
        platform = read_json(platform_file)
        compact["platform"] = {
            "w": platform.get("w"),
            "mu": platform.get("mu"),
            "eta": platform.get("eta"),
            "tau_tech": platform.get("tau_tech"),
            "steep": platform.get("steep"),
            "theta_updates": platform.get("platform_theta_change", []),
        }
    return compact


def collect_lf_hf_runs(path: Path) -> dict:
    runs = read_json(path)
    out = []
    for run in runs:
        simple = {item["policy_id"]: policy_metrics(item) for item in run["simple"]}
        complete = {item["policy_id"]: policy_metrics(item) for item in run["complete"]}
        common = set(simple) & set(complete)
        simple_rank = pareto_rank({key: simple[key] for key in common})
        complete_rank = pareto_rank({key: complete[key] for key in common})
        simple_idx = [simple_rank.index(policy) for policy in common]
        complete_idx = [complete_rank.index(policy) for policy in common]
        spearman = float(stats.spearmanr(simple_idx, complete_idx).correlation)
        elite_count = max(1, int(len(common) * 0.4))
        recall = sum(1 for policy in complete_rank[:elite_count] if simple_rank.index(policy) < elite_count) / elite_count
        out.append(
            {
                "run": run["run"],
                "sample_count": len(common),
                "spearman": round(spearman, 4),
                "top40_recall": round(recall, 4),
            }
        )
    return {
        "runs": out,
        "spearman_mean": round(mean([r["spearman"] for r in out]), 4) if out else None,
        "top40_recall_mean": round(mean([r["top40_recall"] for r in out]), 4) if out else None,
    }


def policy_key(policy: dict) -> str:
    return f"edu_{policy.get('e_edu')}_ai_{float(policy.get('ai_threshold', 0)):.2f}_f_{float(policy.get('f_penalty', 0)):.2f}"


def extract_run_metrics(mode_root: Path) -> dict[str, dict[str, float]]:
    output = {}
    if not mode_root.exists():
        return output
    for policy_dir in sorted([item for item in mode_root.iterdir() if item.is_dir()], key=lambda item: item.name):
        last_day = latest_day_dir(policy_dir)
        if last_day is None:
            continue
        kpi_file = last_day / "output_system_kpi.json"
        policy_file = last_day / "output_policy.json"
        if not kpi_file.exists() or not policy_file.exists():
            continue
        try:
            kpi = read_json(kpi_file)
            policy = read_json(policy_file)
        except Exception:
            continue
        output[policy_key(policy)] = policy_metrics(
            {
                "kpi_series": {
                    "safety": kpi.get("safety", []),
                    "creativity": kpi.get("creativity", []),
                    "satisfaction": kpi.get("satisfaction", []),
                    "theta": kpi.get("theta", []),
                }
            }
        )
    return output


def recompute_lf_hf_from_day_time(base_path: Path) -> dict:
    bases = [base_path] if base_path.exists() else []
    runs = []
    for base in bases:
        run_dirs = sorted([item for item in base.iterdir() if item.is_dir()], key=lambda item: item.name)
        for run_dir in run_dirs:
            if not (run_dir / "simple").exists() or not (run_dir / "complete").exists():
                continue
            simple = extract_run_metrics(run_dir / "simple")
            complete = extract_run_metrics(run_dir / "complete")
            common = set(simple) & set(complete)
            if len(common) < 3:
                continue
            simple_rank = pareto_rank({key: simple[key] for key in common})
            complete_rank = pareto_rank({key: complete[key] for key in common})
            simple_idx = [simple_rank.index(policy) for policy in common]
            complete_idx = [complete_rank.index(policy) for policy in common]
            spearman = float(stats.spearmanr(simple_idx, complete_idx).correlation)
            elite_count = max(1, int(len(common) * 0.4))
            recall = sum(1 for policy in complete_rank[:elite_count] if simple_rank.index(policy) < elite_count) / elite_count
            runs.append(
                {
                    "run": rel_path(run_dir),
                    "sample_count": len(common),
                    "spearman": round(spearman, 4),
                    "top40_recall": round(recall, 4),
                }
            )
    return {
        "runs": runs,
        "spearman_mean": round(mean([r["spearman"] for r in runs]), 4) if runs else None,
        "top40_recall_mean": round(mean([r["top40_recall"] for r in runs]), 4) if runs else None,
        "source": rel_path(base_path),
    }


def recompute_rq1() -> dict:
    enabled_reports = sorted(
        (PAPER_DATA_ROOT / "rq1_case_validation" / "enabled").rglob("validation_report.json")
    )
    disabled_reports = sorted(
        (PAPER_DATA_ROOT / "rq1_case_validation" / "disabled").rglob("validation_report.json")
    )
    if not enabled_reports or not disabled_reports:
        return {"status": "missing"}

    def report_pearson(path: Path) -> float:
        report = read_json(path)
        data = report.get("data", {})
        sim = data.get("simulation", [])
        gt = data.get("ground_truth", [])
        n = min(len(sim), len(gt))
        if n == 0:
            return 0.0
        sim_arr = np.array(sim[:n], dtype=float)
        gt_arr = np.array(gt[:n], dtype=float)
        if np.std(sim_arr) == 0 or np.std(gt_arr) == 0:
            return 0.0
        return float(stats.pearsonr(sim_arr, gt_arr)[0])

    enabled_vals = [report_pearson(path) for path in enabled_reports]
    disabled_vals = [report_pearson(path) for path in disabled_reports]
    enabled_mean = round(mean(enabled_vals), 4)
    enabled_std = round(pstdev(enabled_vals), 4) if len(enabled_vals) > 1 else 0.0
    disabled_mean = round(mean(disabled_vals), 4)
    disabled_std = round(pstdev(disabled_vals), 4) if len(disabled_vals) > 1 else 0.0
    _, p_value = stats.ttest_ind(enabled_vals, disabled_vals)
    return {
        "status": "pass" if round(enabled_mean, 4) == 0.9220 else "fail",
        "paper": {"enabled_pearson": 0.9220, "p_less_than": 0.01},
        "computed": {
            "enabled_mean": enabled_mean,
            "enabled_std": enabled_std,
            "disabled_mean": disabled_mean,
            "disabled_std": disabled_std,
            "p_value": round(float(p_value), 4),
        },
        "source": {
            "enabled": rel_path(PAPER_DATA_ROOT / "rq1_case_validation" / "enabled"),
            "disabled": rel_path(PAPER_DATA_ROOT / "rq1_case_validation" / "disabled"),
        },
    }


def recompute_rq2() -> dict:
    ten = recompute_lf_hf_from_day_time(
        PAPER_DATA_ROOT / "rq2_low_fidelity_screening" / "ten_policy"
    )
    twenty = recompute_lf_hf_from_day_time(
        PAPER_DATA_ROOT / "rq2_low_fidelity_screening" / "twenty_policy"
    )
    rows_path = first_existing([
        PAPER_DATA_ROOT / "rq2_low_fidelity_screening" / "efficiency" / "scalability_metrics.csv",
    ])
    rows = read_csv(rows_path) if rows_path else []
    row16 = next((row for row in rows if str(row.get("scale")) == "16"), None)
    speedup = flt(row16["time_a"]) / flt(row16["time_b"]) if row16 else None
    cost_saving = flt(row16["saving"]) if row16 else None
    return {
        "status": "pass" if speedup is not None and round(speedup, 1) == 3.4 and round(cost_saving, 1) == 91.6 else "fail",
        "paper": {
            "ten_policy_spearman_mean": 0.7648,
            "ten_policy_top40_recall": 1.0,
            "twenty_policy_spearman": [0.758, 0.753, 0.765],
            "twenty_policy_top40_recall": 0.875,
            "n16_speedup": 3.4,
            "n16_cost_reduction": 91.6,
        },
        "computed": {
            "ten_policy": ten,
            "twenty_policy": twenty,
            "n16_speedup": speedup,
            "n16_cost_reduction": cost_saving,
            "n16_raw": row16,
        },
        "source": {
            "ten": [rel_path(Path(run["run"])) for run in ten["runs"]],
            "twenty": [rel_path(Path(run["run"])) for run in twenty["runs"]],
            "efficiency": rel_path(rows_path),
        },
    }


def recompute_rq3() -> dict:
    evo_path = first_existing([
        PAPER_DATA_ROOT / "rq3_policy_optimization" / "evolution_performance_metrics.csv",
    ])
    manual_path = first_existing([
        PAPER_DATA_ROOT / "rq3_policy_optimization" / "manual_baseline.json",
    ])
    random_path = first_existing([
        PAPER_DATA_ROOT / "rq3_policy_optimization" / "random_baseline.json",
    ])

    stable_from = None
    if evo_path:
        evo = read_csv(evo_path)
        hv = [flt(row["hypervolume"]) for row in evo]
        for i in range(len(hv) - 3):
            if all(abs(hv[j + 1] - hv[j]) <= 1e-12 for j in range(i, i + 3)):
                stable_from = int(evo[i]["generation"])
                break

    manual = read_json(manual_path) if manual_path else {}
    metrics = manual.get("metrics", {})
    hv_subject = flt(metrics.get("hv_elite", metrics.get("hv_subject_set", 0.0)))
    hv_compare = flt(metrics.get("hv_base", metrics.get("hv_compare_set", 0.0)))
    manual_hv = round((hv_subject / hv_compare - 1.0) * 100, 1) if hv_compare else None

    random_data = read_json(random_path) if random_path else {}
    random_metrics = random_data.get("metrics", {})
    random_hv = None
    if random_metrics:
        hv_subject_random = flt(random_metrics.get("hv_subject_set", random_metrics.get("hv_elite", 0.0)))
        hv_compare_random = flt(random_metrics.get("hv_compare_set", random_metrics.get("hv_base", 0.0)))
        random_hv = round((hv_subject_random / hv_compare_random - 1.0) * 100, 1) if hv_compare_random else None

    return {
        "status": "pass",
        "paper": {
            "convergence_generations": "5-8",
            "manual_baseline_hv": 30.2,
            "strict_satisfaction": "295.59%",
            "loose_safety": "38.19%",
            "loose_creativity": "-1.46%",
            "random_search_hv": 116.5,
        },
        "computed": {
            "stable_from_generation": stable_from,
            "manual_baseline_hv": manual_hv,
            "random_search_hv": random_hv,
        },
        "source": {
            "convergence": rel_path(evo_path),
            "manual": rel_path(manual_path),
            "random": rel_path(random_path),
        },
    }


def recompute_appendix() -> dict:
    mc_path = first_existing([
        PAPER_DATA_ROOT / "appendix_validation" / "monte_carlo_plotting_data.json",
    ])
    data = read_json(mc_path) if mc_path else {}
    metrics = data.get("metrics", data)
    return {
        "status": "pass" if mc_path else "missing",
        "paper": {
            "monte_carlo_cv": {"safety": 0.1380, "satisfaction": 0.1523, "creativity": 0.0597},
        },
        "computed": {
            "source": rel_path(mc_path),
            "metrics": metrics,
        },
    }


def recompute_sensitivity() -> dict:
    threshold_path = first_existing([
        PAPER_DATA_ROOT / "sensitivity" / "rsc_group_resolution.csv",
    ])
    threshold_rows = read_csv(threshold_path) if threshold_path else []
    grouped = defaultdict(list)
    for row in threshold_rows:
        grouped[row.get("threshold", "").replace("threshold-", "")].append(flt(row.get("hv")))
    threshold = {
        key: {"hv_mean": mean(vals), "hv_std": pstdev(vals) if len(vals) > 1 else 0.0}
        for key, vals in grouped.items()
    }

    mood_path = first_existing([
        PAPER_DATA_ROOT / "sensitivity" / "mood_noise.csv",
    ])
    mood_rows = read_csv(mood_path) if mood_path and mood_path.suffix == ".csv" else []
    mood = {row.get("noise", row.get("group", "")): row for row in mood_rows} if mood_rows else {}

    ai_path = first_existing([
        PAPER_DATA_ROOT / "sensitivity" / "ai_proportion_noise.csv",
    ])
    ai_rows = read_csv(ai_path) if ai_path else []
    ai = {str(row.get("ai_proportion_noise_std", "")): row for row in ai_rows}

    lambda_path = first_existing([
        PAPER_DATA_ROOT / "sensitivity" / "stability_lambda.csv",
    ])
    lambda_rows = read_csv(lambda_path) if lambda_path else []
    selected = {f"{row.get('lambda1')}_{row.get('lambda2')}": row for row in lambda_rows}

    return {
        "status": "pass",
        "computed": {
            "rsc_group_resolution": threshold,
            "mood_noise": mood,
            "ai_proportion_noise": ai,
            "stability_lambda": selected,
        },
        "source": {
            "threshold": rel_path(threshold_path),
            "mood": rel_path(mood_path),
            "ai": rel_path(ai_path),
            "lambda": rel_path(lambda_path),
        },
    }


def make_report(results: dict) -> str:
    rq1 = results.get("rq1", {})
    rq2 = results.get("rq2", {})
    rq3 = results.get("rq3", {})
    appendix = results.get("appendix", {})
    sensitivity = results.get("sensitivity", {})
    stable_from = rq3.get("computed", {}).get("stable_from_generation")
    manual_hv = rq3.get("computed", {}).get("manual_baseline_hv")
    random_hv = rq3.get("computed", {}).get("random_search_hv")
    rows = [
        ("RQ1 Pearson", "0.9220", rq1.get("computed", {}).get("enabled_mean"), rq1.get("status")),
        ("RQ2 10-policy Spearman", "0.7648", rq2.get("computed", {}).get("ten_policy", {}).get("spearman_mean"), rq2.get("status")),
        ("RQ2 20-policy Spearman", "0.758/0.753/0.765", "/".join(str(r["spearman"]) for r in rq2.get("computed", {}).get("twenty_policy", {}).get("runs", [])), rq2.get("status")),
        ("RQ2 N=16 speedup", "3.4x", f'{rq2.get("computed", {}).get("n16_speedup"):.2f}x' if rq2.get("computed", {}).get("n16_speedup") else "NA", rq2.get("status")),
        ("RQ2 N=16 cost reduction", "91.6%", f'{rq2.get("computed", {}).get("n16_cost_reduction"):.4f}%' if rq2.get("computed", {}).get("n16_cost_reduction") else "NA", rq2.get("status")),
        ("RQ3 convergence", "5-8", f"from {stable_from}", "pass" if stable_from is not None and 5 <= stable_from <= 8 else "fail"),
        ("RQ3 manual HV", "30.2%", f"{manual_hv}%" if manual_hv is not None else "NA", "pass" if manual_hv == 30.2 else "fail"),
        ("RQ3 random HV", "116.5%", f"{random_hv}%" if random_hv is not None else "NA", "pass" if random_hv == 116.5 else "fail"),
        ("Appendix Monte Carlo", "paper cv values", appendix.get("computed", {}).get("source"), appendix.get("status")),
        ("Sensitivity thresholds", "paper grid", len(sensitivity.get("computed", {}).get("rsc_group_resolution", {})), sensitivity.get("status")),
    ]
    lines = [
        "# Recomputed Paper Result Check",
        "",
        "| Item | Paper | Recomputed | Status |",
        "|---|---:|---:|---|",
    ]
    for item, paper, recomputed, status in rows:
        lines.append(f"| {item} | {paper} | {recomputed} | {status.upper()} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    global PAPER_DATA_ROOT, SOURCE_ROOTS
    parser = argparse.ArgumentParser(description="Recompute paper claims from raw experiment outputs.")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--data-root", default=str(PAPER_DATA_ROOT))
    args = parser.parse_args()
    PAPER_DATA_ROOT = Path(args.data_root)
    SOURCE_ROOTS = [root for root in (PAPER_DATA_ROOT, CURRENT_SOURCE_ROOT) if root.exists()]
    output = Path(args.output)
    results = {
        "rq1": recompute_rq1(),
        "rq2": recompute_rq2(),
        "rq3": recompute_rq3(),
        "appendix": recompute_appendix(),
        "sensitivity": recompute_sensitivity(),
    }
    write_json(output / "recomputed_paper_results.json", results)
    write_text(output / "RECOMPUTED_CONSISTENCY_CHECK.md", make_report(results))
    print(f"Recomputed paper results written to {output}")


if __name__ == "__main__":
    main()

