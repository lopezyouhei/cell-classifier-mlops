import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serve.app import app

os.environ["STUB_MODE"] = "1"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_version_reports_stub(client):
    body = client.get("/version").json()
    assert body["stub_mode"] is True
    assert "git_sha" in body


def test_predict_contract(client):
    buf = io.BytesIO()
    Image.new("RGB", size=(224, 224), color=(128, 0, 0)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("cell.png", buf, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert {"predicted_class", "label", "request_id", "model_version"} <= body.keys()


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", io.BytesIO(b"nope"), "text/plain")})
    assert r.status_code == 400
