import io
import json
import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from PIL import Image

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


def _load_labels() -> list[str] | None:
    try:
        with open("configs/labels.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        return None
    return [meta["labels"][str(i)] for i in range(meta["n_classes"])]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    STATE["loaded"] = load_model(cfg)
    if not STATE["loaded"].stub:
        STATE["backbone"], STATE["transform"] = load_backbone(cfg)
    STATE["labels"] = _load_labels()
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


def _infer(image: Image.Image) -> dict:
    """
    Blocking work (forward pass through backbone and head). Runs in worker thread.
    """
    loaded = STATE["loaded"]
    with torch.inference_mode():
        x = STATE["transform"](image).unsqueeze(0)
        emb = STATE["backbone"](x)
        probs = torch.softmax(loaded.model(emb), dim=1)[0]
    idx = int(probs.argmax())
    labels = STATE["labels"]
    return {
        "predicted_class": idx,
        "label": labels[idx] if labels else str(idx),
        "probabilities": [round(float(p), 4) for p in probs],
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    METRICS["requests"] += 1
    loaded = STATE["loaded"]

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        METRICS["errors"] += 1
        raise HTTPException(status_code=400, detail="invalid image") from exc

    if loaded is None:
        METRICS["errors"] += 1
        raise HTTPException(status_code=503, detail="model not loaded")

    if loaded.stub:
        result = {"predicted_class": 0, "label": "stub", "probabilities": [1.0]}
    else:
        result = await run_in_threadpool(_infer, image)

    elapsed = (time.perf_counter() - start) * 1000
    METRICS["latency_ms_total"] += elapsed
    METRICS["predictions"][result["label"]] += 1

    log.info(
        json.dumps(
            {
                "event": "predict",
                "request_id": request_id,
                "model_version": loaded.version,
                "label": result["label"],
                "latency_ms": round(elapsed, 2),
            }
        )
    )

    return {**result, "request_id": request_id, "model_version": loaded.version}
