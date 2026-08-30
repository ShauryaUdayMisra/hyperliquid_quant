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

#: Memecoins. Excluded by default; set ``EXCLUDE_MEMECOINS=false`` to trade
#: them anyway.
#:
#: There is no field on the exchange that says "this is a memecoin", so this
#: is a hand-maintained judgement call and will go stale as new ones list.
#: That is the honest trade-off: a wrong name here silently removes a market
#: from the universe, so the resolver logs every name it drops rather than
#: quietly shrinking the book.
#:
#: The line drawn is "a token whose value proposition is the joke itself".
#: Gaming, NFT and metaverse tokens (SAND, GALA, AXS, APE, YGG, ZORA) are
#: not on it even though they are volatile, and neither are legacy or failed
#: chains (kLUNC). Volatility is not the criterion -- position sizing
#: already handles that -- and neither is liquidity: dropping these six from
#: the live top:25 lowered its median depth by 37%.
MEMECOINS = frozenset({
    "DOGE", "SHIB", "PEPE", "BONK", "FLOKI", "WIF", "BOME", "POPCAT",
    "PNUT", "MOODENG", "GOAT", "BRETT", "TURBO", "NEIRO", "MEME", "MEW",
    "MOG", "TRUMP", "MELANIA", "FARTCOIN", "PENGU", "SPX", "PURR",
    "CASHCAT", "PUMP", "HMSTR", "NOT", "ANIME", "GRIFFAIN", "AIXBT",
    "PEOPLE", "BABYDOGE", "DEGEN", "TOSHI", "CHILLGUY", "SLERF", "MYRO",
    "GIGA", "SNEK", "FWOG", "ZEREBRO", "WOJAK", "LADYS", "SUNDOG",
})


def is_memecoin(name: str) -> bool:
    """Whether a market name refers to a memecoin.

    Handles Hyperliquid's thousand-unit prefix: ``kPEPE`` is 1,000 PEPE and
    must be caught by an entry for PEPE. The prefix is only stripped when it
    is a lowercase k in front of an otherwise upper-case name, which is the
    exchange's own convention -- so ``KAITO`` and ``KAS`` are not mistaken
    for prefixed forms of ``AITO`` and ``AS``.
    """
    text = (name or "").strip()
    if not text:
        return False
    if text.upper() in MEMECOINS:
        return True
    if len(text) > 1 and text[0] == "k" and text[1:].isupper():
        return text[1:].upper() in MEMECOINS
    return False


def excluded_names(
    *, exclude_memecoins: bool = True, extra: Iterable[str] = ()
) -> frozenset[str]:
    """Everything gated out, upper-cased for case-insensitive comparison."""
    names = set(EXCLUDED)
    names.update(m.strip().upper() for m in extra if m and m.strip())
    if exclude_memecoins:
        names.update(MEMECOINS)
    return frozenset(names)


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


def rank_by_volume(
    meta: dict[str, Any],
    contexts: Sequence[dict[str, Any]],
    *,
    exclude_memecoins: bool | None = None,
    extra_excluded: Iterable[str] = (),
) -> list[str]:
    """Live perps, most traded first, minus anything gated out.

    Delisted assets are dropped: they still appear in the universe but their
    candles stop, which would quietly poison the feature matrix with a coin
    that has no recent bars.

    Memecoins are dropped by default. Note what this does and does not buy:
    measured on the live top:25, excluding them made the book *shallower*,
    not deeper -- median hourly volume fell 37% and deployable capital 3%,
    because the names promoted in their place rank lower by volume. So this
    is not a liquidity filter; position sizing already handles liquidity
    honestly. It is a judgement that reported volume on a memecoin is worth
    less than the same number on LTC -- more of it wash traded, more of the
    price history gaps and squeezes -- and none of that is visible to an
    impact model that sees only notional.
    """
    if exclude_memecoins is None:
        from config.settings import SETTINGS

        exclude_memecoins = SETTINGS.data.exclude_memecoins
        extra_excluded = extra_excluded or SETTINGS.data.excluded_markets
    gated = excluded_names(exclude_memecoins=exclude_memecoins, extra=extra_excluded)
    universe = meta.get("universe") or []
    if len(universe) != len(contexts):
        raise ValueError(
            f"universe has {len(universe)} assets but {len(contexts)} contexts; "
            "refusing to pair them up by guesswork"
        )

    ranked: list[tuple[float, str]] = []
    dropped: list[str] = []
    for asset, context in zip(universe, contexts):
        # The exchange's own casing is authoritative. Upper-casing turned
        # "kPEPE" into "KPEPE", which every later API call then failed to
        # find -- a market that could never be refreshed and never had bars.
        name = str(asset.get("name", "")).strip()
        if not name or asset.get("isDelisted"):
            continue
        if name.upper() in gated or (exclude_memecoins and is_memecoin(name)):
            dropped.append(name)
            continue
        try:
            volume = float(context.get("dayNtlVlm") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume <= 0.0:
            continue
        ranked.append((volume, name))

    # Said out loud. A name wrongly on the deny-list removes a market
    # silently otherwise, and a shrinking universe looks identical to a
    # quiet market.
    if dropped:
        log.info("gated out %d market(s): %s", len(dropped), ", ".join(sorted(dropped)))

    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in ranked]


def resolve(raw: str, client: Any, *, fallback: Iterable[str] = ()) -> tuple[str, ...]:
    """Turn a MARKETS value into the coins to trade.

    ``client`` needs only ``meta_and_asset_ctxs()``. Any failure to reach the
    exchange degrades to ``fallback`` with a warning, because a trader with a
    smaller universe is useful and a trader with no universe is not.
    """
    from config.settings import SETTINGS

    kind, count = parse_spec(raw)
    if kind == "list":
        # Filtered as well. "Do not trade memecoins" is a property of the
        # system, not of one way of spelling the universe -- naming DOGE
        # explicitly should not be a way around it. Loudly, because asking
        # for a market and not getting it needs an explanation.
        named = explicit_markets(raw)
        gated = excluded_names(
            exclude_memecoins=SETTINGS.data.exclude_memecoins,
            extra=SETTINGS.data.excluded_markets,
        )
        kept = tuple(
            m for m in named
            if m.upper() not in gated
            and not (SETTINGS.data.exclude_memecoins and is_memecoin(m))
        )
        if len(kept) != len(named):
            log.warning(
                "MARKETS names %s but %s %s gated out; trading %s",
                list(named), [m for m in named if m not in kept],
                "is" if len(named) - len(kept) == 1 else "are", list(kept),
            )
        return kept

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
