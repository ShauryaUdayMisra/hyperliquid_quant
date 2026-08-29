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


def test_the_daily_pnl_boundary_matches_the_timezone_it_is_labelled_with() -> None:
    """A UTC midnight under an IST clock is a quietly wrong number.

    The card says "since 00:00 IST". If the server cut the day at UTC
    midnight the boundary would actually be 05:30 IST, and nothing on the
    page would say so.
    """
    from datetime import datetime

    import dashboard.app as app_module

    now_ms = 1_787_900_000_000
    midnight = app_module._local_midnight_ms(now_ms)
    local = datetime.fromtimestamp(midnight / 1000, tz=app_module._DISPLAY_ZONE)

    assert (local.hour, local.minute, local.second) == (0, 0, 0)
    assert midnight <= now_ms
    assert now_ms - midnight < 24 * 3_600_000


def test_an_unusable_timezone_falls_back_instead_of_breaking_the_page(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret")
    monkeypatch.setenv("DISPLAY_TIMEZONE", "Mars/Olympus_Mons")
    import dashboard.app as app_module

    importlib.reload(app_module)
    assert app_module.DISPLAY_TZ == "UTC"

    monkeypatch.delenv("DISPLAY_TIMEZONE")
    importlib.reload(app_module)


def test_the_page_reports_the_settings_it_is_running(monkeypatch) -> None:
    """Configuration the reader cannot see is configuration they cannot trust."""
    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret")
    import importlib

    import dashboard.app as app_module

    importlib.reload(app_module)
    http = TestClient(app_module.app)
    body = http.get("/api/state", auth=("admin", "s3cret")).json()

    for key in ("timezone", "markets", "max_position_usd", "signal_threshold"):
        assert key in body, f"/api/state does not expose {key}"
    for key in ("min_hold_hours", "max_hold_hours", "max_idle_hours"):
        assert key in body["activity"], f"activity does not expose {key}"


def test_the_learning_panel_says_which_model_is_deciding(client):
    """A model that refits itself on a schedule makes a change in behaviour
    indistinguishable from a change in the market -- unless the page says
    which model is running and when it next changes."""
    http, _ = client
    body = http.get("/api/learning", auth=("admin", "s3cret")).json()

    assert set(body) == {"model", "retrain", "scorecard"}
    assert "enabled" in body["retrain"]
    assert "describe" in body["retrain"]
    assert "resolved" in body["scorecard"]


def test_the_learning_endpoint_is_behind_the_password(client):
    http, _ = client
    assert http.get("/api/learning").status_code == 401


def test_a_scorecard_with_no_resolved_calls_still_renders(client):
    """NaN is not JSON, and one unserialisable value blanks every panel."""
    http, _ = client
    response = http.get("/api/learning", auth=("admin", "s3cret"))
    assert response.status_code == 200
    assert "NaN" not in response.text
