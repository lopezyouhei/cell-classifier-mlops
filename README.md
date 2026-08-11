# cell-classifier-mlops
Reproducible training-to-serving pipeline for imbalanced biomedical image classification.

A frozen ViT backbone with a trained classification head on BloodMNIST (8 classes of blood-cell microscopy, imbalanced). The model tracked and versioned in an Azure Machine Learning workspace through the MLflow API, served using FastAPI container on Azure Container Apps, and resolved from the model registry at startup. This structure ensures that promoting or rolling back a model doesn't require an image rebuild. 

Test macro-F1 is 0.973 (val: 0.977), and the CI quality gate is set at 0.96. The quality gate is set high enough to catch genuine regressions and loose enough to compensate for numeric differences between machines.

Live at https://ca-cellclassifier.happybeach-6e00f3cd.germanywestcentral.azurecontainerapps.io — [`/health`](https://ca-cellclassifier.happybeach-6e00f3cd.germanywestcentral.azurecontainerapps.io/health) · [`/version`](https://ca-cellclassifier.happybeach-6e00f3cd.germanywestcentral.azurecontainerapps.io/version) · [`/ predict`](https://ca-cellclassifier.happybeach-6e00f3cd.germanywestcentral.azurecontainerapps.io/predict) · [`/docs`](https://ca-cellclassifier.happybeach-6e00f3cd.germanywestcentral.azurecontainerapps.io/docs)

The service scales to zero when idle, so the first request after a quiet period will take 30-60 seconds to cold-start.

> **Status: M2 complete** The service resolves a model version from the Azure ML registry over the network, authenticated by managed identity, and rollback has been performed end to end.

## Roadmap

| Stage | Scope | Status |
|---|---|---|
| **M0** | Containerized FastAPI service on Fly, health checks, stub model loading | done |
| **M1** | DVC pipeline: prepare → embed → train → evaluate, tracked in MLflow | done |
| **M2** | Registry-driven serving, managed identity, proven rollback | done |
| **M3** | CI: lint, type check, tests, macro-F1 quality gate, smoke tests | next |
| **M4** | CD: build, deploy, post-deploy smoke tests, auto rollback | planned |
| **M5** | Structured logging, `/metrics`, drift check, operations docs | planned |


## Results
Test macro-F1 is 0.973 and validation is 0.977 from a linear head on frozen ViT-S embeddings. The CI quality gate is set at 0.96. 

The data has an imbalance ratio of 2.7 which is not very severe, but for this reason macro-F1 is reported instead of accuracy. Accuracy overemphasizes the majority classes and therefore focusing on macro-F1 and per-class recall are required.


## Architecture

```
prepare -> embed (cached) -> train head -> evaluate + quality gate
                                |
                AZ ML Workspace (tracking + model registry)
                                |
    Container App resolves MODEL_VERSION at startup, via managed identity
```

| Item | Service|
| ---  | ---    |
| Experiment tracking and model registry | Azure ML |
| Dataset and embedding cache | DVC on the workspace's blob storage |
| Container image | Azure Container Apps, scale-to-zero |
| Authentication | System-assigned managed identity, no stored credentials |


Training registers a new version on every run. Deciding which version is served is a separate, deliberate step. This makes rollbacks an alias change rather than a redeployment. 

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
# GPU box (training)
uv sync --extra cu126 --group train --group dev

# CPU-only
uv sync --extra cpu --group train --group dev
```

Run the service locally without a registry:

```bash
STUB_MODE=1 uv run uvicorn src.serve.app:app --reload
```

Run the pipeline (requires an Azure ML workspace and az login, not explained here):

```bash
export MLFLOW_TRACKING_URI=$(az ml workspace show -n <workspace> -g <group> --query mlflow_tracking_uri -o tsv)
uv run dvc repro
```


Checks:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

## Decisions highlights

**Why the backbone is frozen** The trained artifact stays a few hundred KB (head) instead of several MB (ViT), inference can then run on CPU, and the expensive computation happens once on every experiment. This also isolates what is actually being trained (head).

**Why embeddings are cached** `embed` runs once per dataset and backbone pair. Head training (small MLP) then takes just a few seconds, which will allow CI to train + evaluate a model on every pull request.

**Why the model loads from a registry, and is not baked in the image** This decouples the model lifecycle from the application lifecycle. When a better model is available or a deployed model is not performing well anymore (data drift), a model can be promoted or rolled back. This involves a restart of the service, and avoids having to rebuild and deploy. Additionally, `/version` reports exactly what is being served.

**Why promotion is version pinned rather than an MLflow alias** Initially I wanted to use MLflow alias, but Azure ML implements a subset of the MLflow registry API and returns a 404 error for the alias endpoint. This is the reason the service pins an explicit `MODEL_VERSION`instead.

**Managed identity** The container is setup with a sytem-assigned managed identity (`AzureML Data Scientist` on the workspace to resolve the model and `Storage Blob Data Reader`on the storage account to get artifact bytes). No keys to rotate.

**Why one lockfile handles both CUDA and CPU.** `torch` is configured with 2 extras (`cu126`, `cpu`) in the `pyproject.toml` file, so `uv.lock` carries both versions and the install target picks one. Training on a GPU and serving on CPU come from the same locked dependency set. Pros: no separate requirements files.

**Lean docker image** The first build (M0) found 1.4 GB of package-manager cache baked into the layer, and 0.8GB of dependencies added in through `mlflow` — which ships the tracking *server*. For this project we only need to resolve a model URI, so the leaner `mlflow-skinny` package is more appropriate.

## Stack

uv · DVC · MLflow · PyTorch · timm · FastAPI · Docker · GitHub Actions · Fly.io

## Dataset

[MedMNIST v2](https://medmnist.com/) — BloodMNIST, 224×224 variant.
Yang et al., *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D
Biomedical Image Classification*, Scientific Data, 2023.