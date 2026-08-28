"""The Phase 2 gate: the accounting proof must pass in CI, not just by hand."""

from __future__ import annotations

import pytest

from backtest.proof import SCENARIOS, run_proof


@pytest.mark.parametrize("factory", SCENARIOS, ids=lambda f: f.__name__)
def test_each_accounting_scenario_reconciles(factory) -> None:
    scenario = factory()
    failures = [
        f"{c.name}: hand={c.expected!r} engine={c.actual!r} diff={c.expected - c.actual!r}"
        for c in scenario.checks
        if not c.passed
    ]
    assert not failures, f"{scenario.name}\n" + "\n".join(failures)


def test_the_proof_as_a_whole_passes() -> None:
    _, ok = run_proof()
    assert ok
