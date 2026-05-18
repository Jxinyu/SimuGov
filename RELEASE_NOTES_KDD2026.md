# SimuGov KDD 2026 Camera-Ready Artifact

This release contains the reproducibility artifact accompanying the KDD 2026 AI4Sciences Track paper:

> SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design

## Contents

- Source code for SimuGov simulations and governance strategy optimization.
- Archived final-day simulation snapshots used to verify reported paper results.
- Scripts for rebuilding compact paper data and recomputing reported metrics.
- Artifact-review documentation for KDD 2026 reproducibility checks.
- Citation metadata and release checklist for Zenodo archival.

## Quick Verification

Install the minimal artifact-review dependencies:

```bash
pip install -r requirements-repro.txt
```

Run the fast check:

```bash
python scripts/run_fast_check.py
```

Expected report:

```text
_artifact_check/paper_results/RECOMPUTED_CONSISTENCY_CHECK.md
```

The current artifact check passes all listed recomputation checks, including:

- RQ1 Pearson validation.
- RQ2 low-fidelity/high-fidelity screening consistency.
- RQ2 efficiency and cost reduction.
- RQ3 convergence.
- RQ3 manual baseline comparison.
- RQ3 random-search baseline comparison.
- Appendix Monte Carlo validation.
- Sensitivity analysis checks.

## Notes

- The fast verification path does not require LLM API keys.
- Running new SimuGov simulations requires external LLM API credentials and may incur nontrivial cost.
- The Zenodo DOI should be added to `CITATION.cff`, `artifact.md`, `README.md`, and the camera-ready paper after this GitHub release is archived.
