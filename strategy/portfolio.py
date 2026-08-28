"""Portfolio-level allocation across markets.

The risk engine judges one trade at a time. Something still has to decide
*which* trades to bring to it when more markets look attractive than the
position limit allows. That is this module.

Rule: rank candidates by model confidence and take the strongest first.
Existing positions keep their slot -- churning out of a held position to
open a marginally better one pays two sets of costs for a small edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.predict import Signal


@dataclass
class Candidate:
    signal: Signal
    currently_held: bool
    atr_fraction: float | None

    @property
    def wants_exposure(self) -> bool:
        return self.signal.direction in {"long", "short"}


def rank_candidates(candidates: list[Candidate], max_positions: int) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into those to act on and those to skip.

    Returns ``(selected, skipped)``. Positions already open are always
    selected so they can be managed (held, resized or closed); only *new*
    entries compete for the remaining slots.
    """
    held = [c for c in candidates if c.currently_held]
    new_entries = [c for c in candidates if not c.currently_held and c.wants_exposure]

    # Held positions that the model no longer likes will be closed by the
    # strategy, freeing their slot on a later bar rather than this one.
    slots = max(0, max_positions - len(held))
    new_entries.sort(key=lambda c: c.signal.confidence, reverse=True)

    selected = held + new_entries[:slots]
    skipped = new_entries[slots:]
    return selected, skipped
