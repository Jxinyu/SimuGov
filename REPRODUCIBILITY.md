# Reproducibility Guide

This document explains how to verify the main results reported in the KDD 2026 AI4Sciences paper:

> SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design

The bundled reproduction workflow recomputes the reported metrics from archived final-day simulation snapshots. It does not require LLM API keys. API keys are needed only for running new high-fidelity or low-fidelity LLM-driven simulations.

## Environment

Recommended environment:

- Python: 3.10 or newer
- OS: Linux, macOS, or Windows
- Required packages for recomputation: see `requirements-repro.txt`
- Full simulation packages: see `requirements.txt`
- LLM/API access: not required for the paper-result recomputation path

Install dependencies:

```bash
pip install -r requirements-repro.txt
```

## Artifact Contents

- `raw_experiments/`: archived final-day simulation snapshots used for paper-result recomputation.
- `paper_data/`: compact reproduction package generated from `raw_experiments/`.
- `paper_results/`: generated result reports and consistency checks.
- `scripts/prepare_paper_data.py`: rebuilds the compact reproduction package.
- `scripts/recompute_paper_results.py`: recomputes reported metrics from `paper_data/`.
- `scripts/run_fast_check.py`: one-command artifact verification path for reviewers.

## Fast Verification

For a quick artifact check, run:

```bash
python scripts/run_fast_check.py
```

This command rebuilds `paper_data/` from `raw_experiments/`, recomputes all reported metrics, and writes:

```text
_artifact_check/paper_results/RECOMPUTED_CONSISTENCY_CHECK.md
_artifact_check/paper_results/recomputed_paper_results.json
```

The fast-check command intentionally writes to `_artifact_check/` so it does not modify the tracked `paper_data/` package.

Expected outcome: the consistency report should mark the recomputed paper-result checks as `PASS`.

## Full Paper-Result Reproduction

The paper-result reproduction chain is:

```text
raw_experiments -> paper_data -> paper_results
```

Step 1: rebuild the compact paper-data package.

```bash
python scripts/prepare_paper_data.py --raw-root raw_experiments --output paper_data
```

Step 2: recompute the reported results.

```bash
python scripts/recompute_paper_results.py --data-root paper_data --output paper_results/recomputed
```

Step 3: inspect the main report.

```text
paper_results/recomputed/RECOMPUTED_CONSISTENCY_CHECK.md
```

## Claims Covered by the Recompute Pipeline

The recomputation pipeline checks:

- RQ1 case validation Pearson correlation.
- RQ2 low-fidelity/high-fidelity screening consistency.
- RQ2 efficiency and cost reduction.
- RQ3 convergence behavior.
- RQ3 manual and random baseline comparisons.
- Appendix Monte Carlo validation.
- Sensitivity analysis.

## Running New Simulations

New SimuGov simulations call external LLM services and are therefore not part of the zero-key fast verification path.

To run new simulations, copy the environment template and provide API credentials:

```bash
copy .env.example .env
```

Then set:

```text
LLM__KEY1=your_api_key_here
```

Example commands:

```bash
python main_experiment.py single-low
python main_experiment.py single-high
python main_experiment.py optimize --population-size 10 --generations 1
python main_experiment.py optimize --population-size 10 --generations 1 --high-eval
```

New simulation outputs are written to `result_data/`, which is intentionally ignored by git.

## Notes and Limitations

- The high-fidelity simulations can be expensive because they use LLM calls over agent populations and policy settings.
- The reproducibility package therefore archives final-day snapshots and provides deterministic metric recomputation for paper-result verification.
- Re-running new LLM simulations may produce small variations due to model-side nondeterminism, API changes, and stochastic simulation components.
- The provided fast check is intended for artifact reviewers who need to verify the computational pipeline without incurring LLM costs.
