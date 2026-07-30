# model is resolved from MLflow model registry at startup using an alias and not saved into an image

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedModel:
    model: Any | None
    version: str
    stub: bool  # for serving while no model is available


def _model_uri(cfg: dict) -> str:
    """Explicit version supercedes alias, so CD can pin a rollback target.

    Args:
        cfg (dict): MLflow config.

    Returns:
        str: model version / alias
    """

    name = cfg["mlflow"]["registered_model_name"]
    version = os.getenv("MODEL_VERSION")
    if version:
        return f"models:/{name}/{version}"
    alias = os.getenv("MODEL_ALIAS", cfg["mlflow"]["champion_alias"])

    return f"models:/{name}@{alias}"


def load_model(cfg: dict) -> LoadedModel:
    if os.getenv("STUB_MODE", "0") == "1":
        return LoadedModel(model=None, version="stub", stub=True)

    import mlflow

    uri = _model_uri(cfg)
    model = mlflow.pytorch.load_model(uri)
    model.eval()

    return LoadedModel(model=model, version=uri.split("/")[-1], stub=False)


def load_backbone(cfg: dict):
    """Frozen ViT backbone is stored in the image as it doesn't change between releases.

    Args:
        cfg (dict): MLflow config.
    """

    import timm

    model = timm.create_model(cfg["backbone"]["name"], pretrained=True, num_classes=0)
    model.eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    return model, transform
