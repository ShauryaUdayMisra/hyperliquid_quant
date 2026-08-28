"""Which markets to trade.

``MARKETS`` used to be a literal list. Three hand-picked coins is not a
universe, so it now also accepts:

    BTC,ETH,SOL   an explicit list, as before
    top:25        the 25 most liquid perps by 24h notional volume
    all           every perp Hyperliquid currently lists and has not delisted

The volume-ranked forms are resolved against the exchange at startup, not at
import, because resolving them costs a network call and settings must be
importable offline. A resolution failure falls back to whatever explicit
markets are configured rather than leaving the trader with nothing to do.

Ranking by traded volume is not a view on which coins are worth trading. It
is a liquidity filter: the fill simulator charges square-root market impact,
so a thin market punishes size honestly and a $58k order in a coin that
trades $200k a day would be modelled as moving the price against itself
enormously. Liquid markets are simply the ones where a paper fill means
something.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

#: Never trade these even when they rank, whatever the spec says.
#: Compared case-insensitively; the exchange's own casing is preserved
#: everywhere else because names like "kPEPE" are not upper-case.
EXCLUDED = frozenset({"USDC", "USDT"})


def parse_spec(raw: str) -> tuple[str, int | None]:
    """Split a MARKETS value into ``(kind, count)``.

    ``kind`` is one of ``"list"``, ``"top"`` or ``"all"``.
    """
    text = (raw or "").strip()
    lowered = text.lower()
    if lowered == "all":
        return "all", None
    if lowered.startswith("top:"):
        tail = lowered.split(":", 1)[1].strip()
        if not tail.isdigit() or int(tail) < 1:
            raise ValueError(
                f"MARKETS='{raw}' is not a usable count; write it like 'top:25'"
            )
        return "top", int(tail)
    return "list", None


def explicit_markets(raw: str) -> tuple[str, ...]:
    """The literal coins in a spec, ignoring any ranked form."""
    return tuple(
        m.strip().upper()
        for m in (raw or "").split(",")
        if m.strip() and ":" not in m and m.strip().lower() != "all"
    )


def rank_by_volume(meta: dict[str, Any], contexts: Sequence[dict[str, Any]]) -> list[str]:
    """Live perps, most traded first.

    Delisted assets are dropped: they still appear in the universe but their
    candles stop, which would quietly poison the feature matrix with a coin
    that has no recent bars.
    """
    universe = meta.get("universe") or []
    if len(universe) != len(contexts):
        raise ValueError(
            f"universe has {len(universe)} assets but {len(contexts)} contexts; "
            "refusing to pair them up by guesswork"
        )

    ranked: list[tuple[float, str]] = []
    for asset, context in zip(universe, contexts):
        # The exchange's own casing is authoritative. Upper-casing turned
        # "kPEPE" into "KPEPE", which every later API call then failed to
        # find -- a market that could never be refreshed and never had bars.
        name = str(asset.get("name", "")).strip()
        if not name or name.upper() in EXCLUDED or asset.get("isDelisted"):
            continue
        try:
            volume = float(context.get("dayNtlVlm") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume <= 0.0:
            continue
        ranked.append((volume, name))

    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in ranked]


def resolve(raw: str, client: Any, *, fallback: Iterable[str] = ()) -> tuple[str, ...]:
    """Turn a MARKETS value into the coins to trade.

    ``client`` needs only ``meta_and_asset_ctxs()``. Any failure to reach the
    exchange degrades to ``fallback`` with a warning, because a trader with a
    smaller universe is useful and a trader with no universe is not.
    """
    kind, count = parse_spec(raw)
    if kind == "list":
        return explicit_markets(raw)

    try:
        meta, contexts = client.meta_and_asset_ctxs()
        ranked = rank_by_volume(meta, contexts)
    except Exception as exc:  # noqa: BLE001 - degrade, never abort startup
        chosen = tuple(dict.fromkeys(fallback))
        log.warning(
            "could not resolve MARKETS='%s' against the exchange (%s); "
            "falling back to %s", raw, exc, list(chosen) or "nothing",
        )
        return chosen

    if kind == "top":
        ranked = ranked[:count]
    log.info("MARKETS='%s' resolved to %d market(s): %s%s",
             raw, len(ranked), ", ".join(ranked[:10]),
             " ..." if len(ranked) > 10 else "")
    return tuple(ranked)
