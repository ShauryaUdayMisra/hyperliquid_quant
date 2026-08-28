"""The dashboard's front door: what a platform probe and a stranger each see.

A health check cannot present credentials. Pointing one at an authenticated
route made Railway read the 401 as a dead container and kill a deployment
that was running correctly, so the distinction below is load-bearing.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret")
    monkeypatch.setenv("DASHBOARD_USER", "admin")
    import dashboard.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app), app_module


def test_health_answers_without_credentials(client):
    http, _ = client
    response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reveals_nothing_about_the_account(client):
    http, _ = client
    body = http.get("/health").text.lower()
    for secret in ("equity", "cash", "position", "100000"):
        assert secret not in body


def test_the_dashboard_itself_still_demands_a_password(client):
    http, _ = client
    assert http.get("/").status_code == 401
    assert http.get("/api/state").status_code == 401


def test_the_right_password_gets_in(client):
    http, _ = client
    assert http.get("/", auth=("admin", "s3cret")).status_code == 200
