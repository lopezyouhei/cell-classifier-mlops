# cell-classifier-mlops
Reproducible training-to-serving pipeline for imbalanced biomedical image classification.

A frozen ViT backbone with a trained classification head on BloodMNIST (8 classes of blood-cell microscopy, imbalanced). The model tracked and versioned using MLflow, served using FastAPI, and shipped by CI/CD with an automated rollback path.

Live at https://cell-classifier-mlops.fly.dev — [`/health`](https://cell-classifier-mlops.fly.dev/health) · [`/version`](https://cell-classifier-mlops.fly.dev/version) · [`/docs`](https://cell-classifier-mlops.fly.dev/docs)

> **Status: M0** The service is containarized and deployed, currently running in stub mode (no models in registry yet). 
> It reports the model version and git SHA for now. The training pipeline and CI/CD are planned for the next steps.

## Roadmap

| Stage | Scope | Status |
|---|---|---|
| **M0** | Containerized FastAPI service on Fly, health checks, stub model loading | done |
| **M1** | DVC pipeline: prepare → embed → train → evaluate, tracked in MLflow | next |
| **M2** | Registry-driven serving — model resolved by alias (not baked into the image) | planned |
| **M3** | CI: tests, macro-F1 quality gate, ... | needs planning |
| **M4** | CD: build, deply, auto rollback, ... | needs planning |
| **M5** | Structured logging, `/metrics`, drift check, operations docs | needs planning |

# Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
# GPU box (training)
uv sync --extra cu126 --group train --group dev

# CPU-only
uv sync --extra cpu --group train --group dev
```

Run the service locally in stub mode:

```bash
STUB_MODE=1 uv run uvicorn src.serve.app:app --reload
```

Checks:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

## Architecture

```
prepare -> embed (cached) -> train head -> evaluate + quality gate
                                |
                            MLflow registry
                                |
                 FastAPI service (resolved by alias)
```

## Decisions highlights

**Why the backbone is frozen** The trained artifact stays a few hundred KB (head) instead of several MB (ViT), inference can then run on CPU, and the expensive computation happens once on every experiment. This also isolates what is actually being trained (head).

**Why embeddings are cached** `embed` runs once per dataset and backbone pair. Head training (small MLP) then takes just a few seconds, which will allow CI to train + evaluate a model on every pull request.

**Why the model loads from a registry, and is not baked in the image** This decouples the model lifecycle from the application lifecycle. When a better model is available or a deployed model is not performing well anymore (data drift), a model can be promoted or rolled back by only changing the *alias*. This involves a restart of the service, and avoids having to rebuild and deploy. Additionally, `/version` reports exactly what is being served.

**Why one lockfile handles both CUDA and CPU.** `torch` is configured with 2 extras (`cu126`, `cpu`) in the `pyproject.toml` file, so `uv.lock` carries both versions and the install target picks one. Training on a GPU and serving on CPU come from the same locked dependency set. Pros: no separate requirements files.

**Lean docker image** The first build (M0) found 1.4 GB of package-manager cache baked into the layer, and 0.8GB of dependencies added in through `mlflow` — which ships the tracking *server*. For this project we only need to resolve a model URI, so the leaner `mlflow-skinny` package is more appropriate.

## Stack

uv · DVC · MLflow · PyTorch · timm · FastAPI · Docker · GitHub Actions · Fly.io

## Dataset

[MedMNIST v2](https://medmnist.com/) — BloodMNIST, 224×224 variant.
Yang et al., *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D
Biomedical Image Classification*, Scientific Data, 2023.