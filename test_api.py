import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from diagml import api  # noqa: E402
from diagml.data import load_data  # noqa: E402


@pytest.fixture()
def client(trained_dir, monkeypatch):
    monkeypatch.setattr(api, "ARTIFACTS_DIR", str(trained_dir))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(client):
    body = client.get("/model").json()
    assert len(body["feature_names"]) == 30


def test_predict(client):
    X, _ = load_data()
    r = client.post("/predict", json={"features": X.iloc[0].to_dict()})
    assert r.status_code == 200
    assert r.json()["class_name"] == "malignant"


def test_predict_validation_error(client):
    r = client.post("/predict", json={"features": {"mean radius": 1.0}})
    assert r.status_code == 422


def test_batch(client):
    X, _ = load_data()
    recs = [X.iloc[i].to_dict() for i in range(3)]
    r = client.post("/predict/batch", json={"records": recs})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3
