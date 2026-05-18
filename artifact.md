# Artifact Review Guide

## Paper

SimuGov: A Simulation Optimization Framework for Generative AI Governance Strategy Design

Accepted by the KDD 2026 AI4Sciences Track.

## Artifact Summary

This artifact contains the source code, archived final-day simulation snapshots, and recomputation scripts used to verify the main reported results of SimuGov.

The no-key reproduction path recomputes paper metrics from bundled snapshots. Running new LLM-driven simulations is supported separately and requires API credentials.

## Repository

GitHub repository:

```text
https://github.com/Jxinyu/SimuGov
```

Artifact DOI:

```text
To be filled after archiving the GitHub release on Zenodo.
```

## Quick Check

Install dependencies:

```bash
pip install -r requirements-repro.txt
```

Run the reviewer-friendly verification script:

```bash
python scripts/run_fast_check.py
```

Expected outputs:

```text
_artifact_check/paper_results/RECOMPUTED_CONSISTENCY_CHECK.md
_artifact_check/paper_results/recomputed_paper_results.json
```

The consistency report should show `PASS` for the recomputed checks.

## Full Reproduction

See `REPRODUCIBILITY.md` for the full reproduction procedure.

## Artifact Scope

This artifact supports:

- The AI watermark governance simulation setting.
- PAEP-based psychological and adversarial-environment agent modeling.
- RSC-based low-fidelity screening.
- Multi-objective governance strategy optimization.
- Recomputed results for RQ1, RQ2, RQ3, appendix validation, and sensitivity analyses.

## Resource Availability Statement Template

After Zenodo generates the release DOI, use the following camera-ready text and replace the placeholder DOI:

```latex
\newcommand\kddavailabilityurl{https://doi.org/10.5281/zenodo.xxxxxxx}
\ifdefempty{\kddavailabilityurl}{}{
\begingroup\small\noindent\raggedright\textbf{Resource Availability:}\\
The source code and reproducibility artifact for this paper have been made publicly available at
\url{\kddavailabilityurl}. The corresponding GitHub repository is available at
\url{https://github.com/Jxinyu/SimuGov}.
\endgroup
}
```

## Reviewer Notes

- The fast check does not require API keys.
- Full new simulations require external LLM access and may incur nontrivial cost.
- `raw_experiments/` contains final-day snapshots rather than full multi-day logs, because the reported metrics are computed from final-day states.
