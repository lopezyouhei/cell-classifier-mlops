import json
import logging
import os
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.config import load_config
from src.serve.loader import load_backbone, load_model

logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s","level":"%(levelname)s", "msg":%(message)s}',
)

log = logging.getLogger("cell-classifier")

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/train.yaml")
cfg = load_config(CONFIG_PATH)

STATE: dict = {"loaded": None, "backbone": None, "transform": None, "labels": None}
METRICS: dict = {
    "requests": 0,
    "errors": 0,
    "latency_ms_total": 0.0,
    "predictions": Counter(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    STATE["loaded"] = load_model(cfg)
    if not STATE["loaded"].stub:
        STATE["backbone"], STATE["transform"] = load_backbone(cfg)
    try:
        with open("artifacts/splits.json") as f:
            meta = json.load(f)
        STATE["labels"] = [meta["labels"][str(i)] for i in range(meta["n_classes"])]
    except FileNotFoundError:
        STATE["labels"] = None
    log.info(json.dumps({"event": "startup", "model_version": STATE["loaded"].version}))

    yield

    log.info(json.dumps({"event": "shutdown"}))


app = FastAPI(title="cell-classifier-mlops", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict:
    "Model version and image's git SHA."
    loaded = STATE["loaded"]
    return {
        "model_version": loaded.version if loaded else "unloaded",
        "stub_mode": bool(loaded and loaded.stub),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "registered_model": cfg["mlflow"]["registered_model_name"],
    }
