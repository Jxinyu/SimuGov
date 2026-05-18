from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = PROJECT_ROOT / "_artifact_check"
CHECK_DATA_ROOT = CHECK_ROOT / "paper_data"
CHECK_RESULT_ROOT = CHECK_ROOT / "paper_results"


def run_step(args: list[str]) -> None:
    print(f"$ {' '.join(args)}")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step([
        python,
        "scripts/prepare_paper_data.py",
        "--raw-root",
        "raw_experiments",
        "--output",
        CHECK_DATA_ROOT.as_posix(),
    ])
    run_step([
        python,
        "scripts/recompute_paper_results.py",
        "--data-root",
        CHECK_DATA_ROOT.as_posix(),
        "--output",
        CHECK_RESULT_ROOT.as_posix(),
    ])
    report = CHECK_RESULT_ROOT / "RECOMPUTED_CONSISTENCY_CHECK.md"
    print(f"\nFast check complete. Main report: {report.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
