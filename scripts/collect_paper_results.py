from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DATA_ROOT = PROJECT_ROOT / "paper_data"
OUTPUT_ROOT = PROJECT_ROOT / "paper_results" / "collected"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def rel_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def latest_day_dir(policy_dir: Path) -> Path | None:
    day_dirs = [d for d in policy_dir.iterdir() if d.is_dir() and d.name.startswith("day_time_") and d.name.split("_")[-1].isdigit()]
    if not day_dirs:
        return None
    return max(day_dirs, key=lambda d: int(d.name.split("_")[-1]))


def final_day_snapshot(policy_dir: Path) -> dict | None:
    last_day = latest_day_dir(policy_dir)
    if last_day is None:
        return None
    kpi_file = last_day / "output_system_kpi.json"
    policy_file = last_day / "output_policy.json"
    if not kpi_file.exists():
        return None
    kpi = read_json(kpi_file)
    policy = read_json(policy_file) if policy_file.exists() else {}
    return {
        "policy_dir": rel_path(policy_dir),
        "final_day": int(last_day.name.split("_")[-1]),
        "policy": policy,
        "final_kpi": {
            "safety": (kpi.get("safety") or [None])[-1],
            "creativity": (kpi.get("creativity") or [None])[-1],
            "satisfaction": (kpi.get("satisfaction") or [None])[-1],
            "theta": (kpi.get("theta") or [None])[-1],
        },
    }


def collect_snapshots(root: Path) -> list[dict]:
    snapshots = []
    for path in root.rglob("day_time_*"):
        if path.is_dir():
            item = final_day_snapshot(path.parent)
            if item is not None:
                snapshots.append(item)
    seen = set()
    out = []
    for item in snapshots:
        key = item["policy_dir"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def collect_rq1(output_root: Path) -> dict:
    src_root = PAPER_DATA_ROOT / "rq1_case_validation"
    enabled = list((src_root / "enabled").rglob("validation_report.json"))
    disabled = list((src_root / "disabled").rglob("validation_report.json"))
    if not enabled or not disabled:
        return {"status": "missing"}
    copy_file(enabled[0], output_root / "rq1" / "enabled_validation_report.json")
    copy_file(disabled[0], output_root / "rq1" / "disabled_validation_report.json")
    write_json(output_root / "rq1" / "final_day_snapshots.json", collect_snapshots(src_root))
    return {"status": "pass", "source": rel_path(src_root), "artifact": rel_path(output_root / "rq1")}


def collect_rq2(output_root: Path) -> dict:
    src_root = PAPER_DATA_ROOT / "rq2_low_fidelity_screening"
    ten = src_root / "ten_policy"
    twenty = src_root / "twenty_policy"
    eff = src_root / "efficiency" / "scalability_metrics.csv"
    if not ten.exists() or not twenty.exists() or not eff.exists():
        return {"status": "missing"}
    write_json(output_root / "rq2" / "final_day_snapshots.json", collect_snapshots(src_root))
    copy_file(eff, output_root / "rq2" / "scalability_metrics.csv")
    return {
        "status": "pass",
        "source": {
            "ten": rel_path(ten),
            "twenty": rel_path(twenty),
            "efficiency": rel_path(eff),
        },
        "artifact": rel_path(output_root / "rq2"),
    }


def collect_rq3(output_root: Path) -> dict:
    src_root = PAPER_DATA_ROOT / "rq3_policy_optimization"
    convergence = src_root / "evolution_performance_metrics.csv"
    manual = src_root / "manual_baseline.json"
    random = src_root / "random_baseline.json"
    if not convergence.exists() or not manual.exists() or not random.exists():
        return {"status": "missing"}
    copy_file(convergence, output_root / "rq3" / "evolution_performance_metrics.csv")
    copy_file(manual, output_root / "rq3" / "manual_baseline.json")
    copy_file(random, output_root / "rq3" / "random_baseline.json")
    write_json(output_root / "rq3" / "final_day_snapshots.json", collect_snapshots(src_root))
    return {"status": "pass", "source": rel_path(src_root), "artifact": rel_path(output_root / "rq3")}


def collect_appendix(output_root: Path) -> dict:
    src_root = PAPER_DATA_ROOT / "appendix_validation"
    mc = src_root / "monte_carlo_plotting_data.json"
    if not mc.exists():
        return {"status": "missing"}
    copy_file(mc, output_root / "appendix" / "monte_carlo_plotting_data.json")
    write_json(output_root / "appendix" / "final_day_snapshots.json", collect_snapshots(src_root))
    return {"status": "pass", "source": rel_path(src_root), "artifact": rel_path(output_root / "appendix")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paper results from paper_data.")
    parser.add_argument("experiment", nargs="?", default="all", choices=["all", "rq1", "rq2", "rq3", "appendix"])
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": rel_path(PAPER_DATA_ROOT),
        "output_root": rel_path(output),
        "note": "English-only relative paths.",
    }

    if args.experiment in {"all", "rq1"}:
        manifest["rq1"] = collect_rq1(output)
    if args.experiment in {"all", "rq2"}:
        manifest["rq2"] = collect_rq2(output)
    if args.experiment in {"all", "rq3"}:
        manifest["rq3"] = collect_rq3(output)
    if args.experiment in {"all", "appendix"}:
        manifest["appendix"] = collect_appendix(output)

    write_json(output / "manifest.json", manifest)
    print(f"Collected into {rel_path(output)}")


if __name__ == "__main__":
    main()
