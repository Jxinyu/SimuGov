# SimuGov

**A simulation-optimization framework for generative AI governance strategy design.**

SimuGov provides an agent-based social simulation environment and a multi-fidelity optimization pipeline for evaluating AI governance policies. This repository is the official artifact package for the KDD 2026 AI4Sciences paper:

> SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design

The repository includes:

- Source code for running new SimuGov simulations and policy optimization.
- Final-day raw simulation snapshots used to reproduce the reported paper results.
- Scripts that rebuild compact paper data and recompute all reported metrics.

The paper reproduction pipeline does not require LLM API keys. API keys are needed only for running new LLM-driven simulations.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/prepare_paper_data.py --raw-root raw_experiments --output paper_data
python scripts/recompute_paper_results.py --data-root paper_data --output paper_results/recomputed
```

The main verification report is written to:

```text
paper_results/recomputed/RECOMPUTED_CONSISTENCY_CHECK.md
```

## Repository Layout

- `main_experiment.py`: command-line entry point for running new SimuGov simulations.
- `config/`: configuration models and YAML settings.
- `method/`, `nsga/`, `utils/`: simulation, agent, optimization, and analysis implementation.
- `raw_experiments/`: raw final-day `day_time_*` simulation snapshots used by the paper.
- `paper_data/`: compact reproduction package generated from `raw_experiments/`.
- `paper_results/`: derived reproduction outputs and consistency reports.
- `scripts/`: data migration, preparation, collection, and recomputation utilities.
- `data/`: lightweight auxiliary data used by the codebase.

## Install

```bash
pip install -r requirements.txt
```

API keys are not required to recompute the reported paper results from the bundled snapshots. API keys are required only when running new LLM-driven simulations.

## Reproduce Paper Results

The reproducibility chain is:

```text
raw_experiments -> paper_data -> paper_results
```

Prepare the compact paper data package:

```bash
python scripts/prepare_paper_data.py --raw-root raw_experiments --output paper_data
```

Recompute all reported paper results:

```bash
python scripts/recompute_paper_results.py --data-root paper_data --output paper_results/recomputed
```

Optional: collect intermediate artifacts for inspection:

```bash
python scripts/collect_paper_results.py all --output paper_results/collected
```

The main verification report is:

```text
paper_results/recomputed/RECOMPUTED_CONSISTENCY_CHECK.md
```

## Raw Data Migration

The bundled `raw_experiments/` directory already contains the raw final-day snapshots needed for reproduction. If migrating again from an original full experiment archive, run:

```bash
python scripts/migrate_raw_experiments.py --source-root path/to/original/project --output raw_experiments
```

The migration script keeps only the final `day_time_*` snapshot for each run or policy. Earlier days are intentionally omitted because the reported metrics are computed from the final-day state.

## Run New Simulations

New simulations call LLM services. Configure at least one key before running these commands:

```bash
copy .env.example .env
```

Then set the key value in `.env` using the nested settings format expected by `config/config.py`, for example:

```text
LLM__KEY1=your_api_key_here
```

Examples:

```bash
python main_experiment.py single-low
python main_experiment.py single-high
python main_experiment.py optimize --population-size 10 --generations 1
python main_experiment.py optimize --population-size 10 --generations 1 --high-eval
```

Simulation outputs are written under `result_data/`.

## Verified Claims

The recomputation pipeline checks:

- RQ1 case validation Pearson correlation
- RQ2 low-fidelity screening consistency
- RQ2 efficiency and cost reduction
- RQ3 convergence
- RQ3 manual and random baseline comparisons
- Appendix Monte Carlo validation
- Sensitivity analysis

`paper_results/` is derived output only and can be regenerated from `raw_experiments/`.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
