from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "raw_experiments"


NAMES = {
    "social": "\u4eff\u771f\u793e\u4f1a\u8bc4\u4f30",
    "case": "\u6848\u4f8b\u9a8c\u8bc1",
    "verified": "\u9a8c\u8bc1\u901a\u8fc7",
    "enabled": "\u5f00\u542f\u5fc3\u7406\u53c2\u6570",
    "disabled": "\u5173\u95ed\u5fc3\u7406\u53c2\u6570",
    "internal": "\u5185\u90e8\u4e00\u81f4\u6027\u9a8c\u8bc1",
    "low_fidelity": "\u4f4e\u7c92\u5ea6\u6a21\u578b\u7b5b\u9009\u6709\u6548\u6027\u9a8c\u8bc1",
    "multi": "\u591a\u7c92\u5ea6\u65b9\u6cd5\u8bc4\u4f30",
    "efficiency": "\u6548\u7387\u5b9e\u9a8c",
    "closed_loop": "\u95ed\u73af\u6709\u6548\u6027\u5b9e\u9a8c",
    "convergence": "\u6536\u655b\u6027\u9a8c\u8bc1\u901a\u8fc7",
    "baseline": "\u57fa\u51c6\u5bf9\u6bd4\u9a8c\u8bc1\u901a\u8fc7",
    "random_baseline": "\u968f\u673a\u57fa\u51c6\u6bd4\u8f83",
    "sensitivity": "\u654f\u611f\u6027\u5206\u6790",
}

REQUIRED_DAY_FILES = [
    "output_agent_think_memories.json",
    "output_contents.json",
    "output_memories.json",
    "output_personas.json",
    "output_platform.json",
    "output_policy.json",
    "output_system_kpi.json",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def day_number(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return -1


def copy_latest_day_tree(src_policy_dir: Path, dst_policy_dir: Path) -> None:
    day_dirs = [
        d for d in src_policy_dir.iterdir()
        if d.is_dir() and d.name.startswith("day_time_") and day_number(d) >= 0
    ]
    if not day_dirs:
        raise FileNotFoundError(f"No day_time_* in {src_policy_dir}")
    latest_day = max(day_dirs, key=day_number)
    copy_tree(latest_day, dst_policy_dir / latest_day.name)


def sorted_dirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def numeric_dirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def migrate_rq1(src_root: Path, out_root: Path) -> None:
    root = src_root / "experiment" / NAMES["social"] / NAMES["case"] / NAMES["verified"]
    for src_group, dst_group in ((NAMES["enabled"], "enabled"), (NAMES["disabled"], "disabled")):
        for run_dir in numeric_dirs(root / src_group):
            dst_run = out_root / "rq1_case_validation" / dst_group / f"run_{run_dir.name}"
            copy_latest_day_tree(run_dir / "case_validation", dst_run / "case_validation")
            reports = sorted(run_dir.glob("validation_report_*.json"))
            if reports:
                copy_file(reports[-1], dst_run / "validation_report.json")


def migrate_lf_group(src_group_root: Path, out_group_root: Path) -> None:
    run_index = 1
    for run_dir in sorted_dirs(src_group_root):
        if not (run_dir / "simple").exists() or not (run_dir / "complete").exists():
            continue
        dst_run = out_group_root / f"run_{run_index}"
        run_index += 1
        for mode in ("simple", "complete"):
            for policy_index, policy_dir in enumerate(sorted_dirs(run_dir / mode), start=1):
                copy_latest_day_tree(policy_dir, dst_run / mode / f"policy_{policy_index:02d}")


def migrate_rq2(src_root: Path, out_root: Path) -> None:
    root = src_root / "experiment" / NAMES["low_fidelity"] / NAMES["verified"]
    migrate_lf_group(root / "result", out_root / "rq2_low_fidelity_screening" / "ten_policy")
    migrate_lf_group(root / "20-lf", out_root / "rq2_low_fidelity_screening" / "twenty_policy")
    copy_file(
        src_root / "experiment" / NAMES["multi"] / NAMES["efficiency"] / NAMES["verified"] / "result" / "paper" / "2-3-2" / "scalability_metrics.csv",
        out_root / "rq2_low_fidelity_screening" / "efficiency" / "scalability_metrics.csv",
    )


def migrate_rq3(src_root: Path, out_root: Path) -> None:
    root = src_root / "experiment" / NAMES["multi"] / NAMES["closed_loop"]
    copy_file(root / NAMES["verified"] / NAMES["convergence"] / "exported_data" / "evolution_performance_metrics.csv", out_root / "rq3_policy_optimization" / "evolution_performance_metrics.csv")
    copy_file(root / NAMES["verified"] / NAMES["baseline"] / "1" / "20260118_193457" / "benchmarking_data.json", out_root / "rq3_policy_optimization" / "manual_baseline.json")
    copy_file(root / "output" / NAMES["random_baseline"] / "02_elite_vs_best_random_group_data.json", out_root / "rq3_policy_optimization" / "random_baseline.json")


def migrate_appendix(src_root: Path, out_root: Path) -> None:
    copy_file(
        src_root / "experiment" / NAMES["social"] / NAMES["internal"] / "output" / "1770026858" / "robustness_analysis" / "monte_carlo_plotting_data.json",
        out_root / "appendix_validation" / "monte_carlo_plotting_data.json",
    )


def migrate_sensitivity(src_root: Path, out_root: Path) -> None:
    root = src_root / "experiment" / NAMES["sensitivity"]
    copy_file(root / "threshold" / "output" / "hv_by_threshold_runs.csv", out_root / "sensitivity" / "rsc_group_resolution.csv")
    mood = root / "mood_noise" / "output" / "mood_noise_summary.csv"
    if mood.exists():
        copy_file(mood, out_root / "sensitivity" / "mood_noise.csv")
    copy_file(root / "ai_proportion_noise" / "output" / "ai_proportion_noise_summary.csv", out_root / "sensitivity" / "ai_proportion_noise.csv")
    copy_file(root / "stability_regularization" / "output" / "stability_lambda_comparison.csv", out_root / "sensitivity" / "stability_lambda.csv")


def validate_raw_tree(out_root: Path) -> dict:
    day_dirs = list(out_root.rglob("day_time_*"))
    complete = 0
    incomplete = []
    groups = {}
    for day_dir in day_dirs:
        parent = day_dir.parent.as_posix()
        groups.setdefault(parent, 0)
        groups[parent] += 1
        missing = [name for name in REQUIRED_DAY_FILES if not (day_dir / name).exists()]
        if missing:
            incomplete.append({"path": day_dir.as_posix(), "missing": missing})
        else:
            complete += 1
    multi_day_groups = {path: count for path, count in groups.items() if count > 1}
    return {
        "day_time_dirs": len(day_dirs),
        "policy_or_run_groups": len(groups),
        "groups_with_multiple_day_dirs": len(multi_day_groups),
        "complete_day_time_dirs": complete,
        "incomplete": incomplete[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate final-day raw day_time experiment outputs into English paths.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    src_root = Path(args.source_root)
    out_root = Path(args.output)
    if out_root.exists():
        shutil.rmtree(out_root)

    migrate_rq1(src_root, out_root)
    migrate_rq2(src_root, out_root)
    migrate_rq3(src_root, out_root)
    migrate_appendix(src_root, out_root)
    migrate_sensitivity(src_root, out_root)
    manifest = validate_raw_tree(out_root)
    manifest["source_root"] = "external_source"
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
