# KDD 2026 Artifact Release Checklist

Use this checklist before creating the GitHub release and Zenodo DOI.

## Repository Readiness

- [ ] Repository is public.
- [ ] `README.md` explains the project, installation, quick start, and reproduction path.
- [ ] `LICENSE` is present and matches the intended open-source license.
- [ ] `CITATION.cff` is present.
- [ ] `REPRODUCIBILITY.md` is present.
- [ ] `artifact.md` is present.
- [ ] `requirements.txt` is present for the full simulation environment.
- [ ] `requirements-repro.txt` is present for artifact-review recomputation.
- [ ] `.env` and real API keys are not tracked.
- [ ] IDE metadata such as `.idea/` is not tracked.
- [ ] Runtime outputs such as `result_data/` are not tracked.

## Verification

- [ ] Install dependencies in a clean environment.
- [ ] Run:

```bash
python scripts/run_fast_check.py
```

- [ ] Confirm `_artifact_check/paper_results/RECOMPUTED_CONSISTENCY_CHECK.md` is generated.
- [ ] Confirm the report statuses are `PASS`.

## GitHub Release

- [ ] Create a release tag, recommended:

```text
v1.0.0-kdd2026
```

- [ ] Release title:

```text
SimuGov KDD 2026 Camera-Ready Artifact
```

- [ ] Release notes mention that the artifact accompanies the KDD 2026 AI4Sciences paper.

## Zenodo DOI

- [ ] Connect the GitHub repository to Zenodo.
- [ ] Enable archiving for `Jxinyu/SimuGov`.
- [ ] Publish or re-publish the GitHub release.
- [ ] Copy the Zenodo release DOI.
- [ ] Add the DOI to `README.md`, `CITATION.cff`, `artifact.md`, and the camera-ready paper.

## Camera-Ready Paper

- [ ] Add the KDD `Resource Availability` statement after `\maketitle`.
- [ ] Include both the Zenodo DOI and GitHub repository link.
- [ ] Confirm the DOI resolves correctly.
