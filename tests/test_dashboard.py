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


def test_the_pages_javascript_parses() -> None:
    """A syntax error in the page script blanks the entire dashboard.

    Every value on the page is rendered by one script, so a single bad
    token -- a redeclared variable, a stray brace -- leaves the user staring
    at "loading..." with all six panels empty and every API returning 200.
    That happened. Nothing else in the suite would have caught it.
    """
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot parse the page script")

    import dashboard.app as app_module

    scripts = re.findall(r"<script[^>]*>(.*?)</script>", app_module.INDEX_HTML, re.S)
    assert scripts, "the page has no script to check"

    result = subprocess.run(
        [node, "--check", "-"], input="\n".join(scripts),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"page script does not parse:\n{result.stderr}"


def test_the_forced_entry_is_reported_at_the_decision_that_acts_on_it() -> None:
    """The timer expiring and the trader acting are different moments.

    The trader only wakes on bar boundaries, so a timer that expires at
    12:21 is acted on at the 13:00 decision. Reporting the expiry told the
    user 12:21 and nothing happened then.
    """
    import dashboard.app as app_module

    hour = 3_600_000
    deadline = 12 * hour + 21 * 60_000        # 12:21 on an arbitrary day
    acted_on = app_module._next_decision_ms(deadline - 1)

    assert acted_on > deadline
    assert acted_on == 13 * hour + 15_000     # the 13:00 decision, plus its wait

    # A deadline that lands exactly on a boundary is served by that boundary.
    on_the_hour = 13 * hour
    assert app_module._next_decision_ms(on_the_hour - 1) == on_the_hour + 15_000
