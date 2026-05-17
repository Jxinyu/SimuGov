from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "raw_experiments"
OUTPUT_ROOT = PROJECT_ROOT / "paper_data"


EDU_MAP = {"low": "low", "medium": "medium", "high": "high"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def latest_day_dir(policy_dir: Path) -> Path | None:
    day_dirs = [d for d in policy_dir.iterdir() if d.is_dir() and d.name.startswith("day_time_") and d.name.split("_")[-1].isdigit()]
    if not day_dirs:
        return None
    return max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))


def prepare_rq1(raw_root: Path, out_root: Path) -> None:
    for group in ("enabled", "disabled"):
        for run_dir in sorted((raw_root / "rq1_case_validation" / group).glob("run_*")):
            report = run_dir / "validation_report.json"
            if not report.exists():
                continue
            data = read_json(report).get("data", {})
            write_json(
                out_root / "rq1_case_validation" / group / run_dir.name / "validation_report.json",
                {
                    "data": {
                        "ground_truth": data.get("ground_truth", []),
                        "simulation": data.get("simulation", []),
                        "satisfaction": data.get("satisfaction", []),
                    }
                },
            )


def prepare_policy(policy_dir: Path, out_dir: Path) -> None:
    last_day = latest_day_dir(policy_dir)
    if last_day is None:
        return
    kpi = read_json(last_day / "output_system_kpi.json")
    policy = read_json(last_day / "output_policy.json")
    write_json(
        out_dir / last_day.name / "output_system_kpi.json",
        {
            "safety": kpi.get("safety", []),
            "creativity": kpi.get("creativity", []),
            "satisfaction": kpi.get("satisfaction", []),
            "theta": kpi.get("theta", []),
        },
    )
    write_json(
        out_dir / last_day.name / "output_policy.json",
        {
            "e_edu": policy.get("e_edu"),
            "ai_threshold": policy.get("ai_threshold"),
            "f_penalty": policy.get("f_penalty"),
        },
    )


def prepare_lf_hf(raw_root: Path, out_root: Path, group: str) -> None:
    src = raw_root / "rq2_low_fidelity_screening" / group
    dst = out_root / "rq2_low_fidelity_screening" / group
    for run_dir in sorted(src.glob("run_*")):
        for mode in ("simple", "complete"):
            for policy_dir in sorted((run_dir / mode).glob("policy_*")):
                prepare_policy(policy_dir, dst / run_dir.name / mode / policy_dir.name)


def prepare_rq2(raw_root: Path, out_root: Path) -> None:
    prepare_lf_hf(raw_root, out_root, "ten_policy")
    prepare_lf_hf(raw_root, out_root, "twenty_policy")
    copy_file(
        raw_root / "rq2_low_fidelity_screening" / "efficiency" / "scalability_metrics.csv",
        out_root / "rq2_low_fidelity_screening" / "efficiency" / "scalability_metrics.csv",
    )


def prepare_rq3(raw_root: Path, out_root: Path) -> None:
    src = raw_root / "rq3_policy_optimization"
    dst = out_root / "rq3_policy_optimization"
    copy_file(src / "evolution_performance_metrics.csv", dst / "evolution_performance_metrics.csv")
    for src_name, dst_name in (("manual_baseline.json", "manual_baseline.json"), ("random_baseline.json", "random_baseline.json")):
        data = read_json(src / src_name)
        write_json(dst / dst_name, {"metrics": data.get("metrics", {})})


def prepare_appendix(raw_root: Path, out_root: Path) -> None:
    src = raw_root / "appendix_validation" / "monte_carlo_plotting_data.json"
    data = read_json(src)
    write_json(out_root / "appendix_validation" / "monte_carlo_plotting_data.json", {"metrics": data.get("metrics", data)})


def prepare_sensitivity(raw_root: Path, out_root: Path) -> None:
    src = raw_root / "sensitivity"
    dst = out_root / "sensitivity"
    for name in ("rsc_group_resolution.csv", "mood_noise.csv", "ai_proportion_noise.csv", "stability_lambda.csv"):
        path = src / name
        if path.exists():
            copy_file(path, dst / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare minimal paper_data from raw day_time experiment outputs.")
    parser.add_argument("--raw-root", default=str(RAW_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.output)
    if out_root.exists():
        shutil.rmtree(out_root)

    prepare_rq1(raw_root, out_root)
    prepare_rq2(raw_root, out_root)
    prepare_rq3(raw_root, out_root)
    prepare_appendix(raw_root, out_root)
    prepare_sensitivity(raw_root, out_root)
    write_json(out_root / "manifest.json", {"source_root": "raw_experiments", "output_root": "paper_data"})
    print(f"Prepared paper data at {out_root}")


if __name__ == "__main__":
    main()
