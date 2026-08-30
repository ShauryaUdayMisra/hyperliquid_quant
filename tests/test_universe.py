"""Choosing which markets to trade.

MARKETS grew from a hand-written list of three coins into something that can
ask the exchange for its most liquid perps, or for all of them.
"""

from __future__ import annotations

import pytest

from data.universe import EXCLUDED, explicit_markets, parse_spec, rank_by_volume, resolve


def meta(*names):
    return {"universe": [{"name": n, "maxLeverage": 10} for n in names]}


def ctxs(*volumes):
    return [{"dayNtlVlm": str(v)} for v in volumes]


def test_a_plain_list_is_left_alone() -> None:
    assert parse_spec("BTC,ETH,SOL") == ("list", None)
    assert explicit_markets("BTC, eth ,Sol") == ("BTC", "ETH", "SOL")


def test_top_n_and_all_are_recognised() -> None:
    assert parse_spec("top:25") == ("top", 25)
    assert parse_spec("TOP:5") == ("top", 5)
    assert parse_spec("all") == ("all", None)


def test_a_nonsense_count_is_refused_rather_than_guessed() -> None:
    for bad in ("top:", "top:0", "top:-3", "top:many"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_markets_rank_by_traded_volume() -> None:
    ranked = rank_by_volume(meta("A", "B", "C"), ctxs(1_000, 9_000, 5_000))
    assert ranked == ["B", "C", "A"]


def test_delisted_and_untraded_markets_are_dropped() -> None:
    """A delisted perp still appears in the universe but its candles stop."""
    m = meta("LIVE", "DEAD", "QUIET")
    m["universe"][1]["isDelisted"] = True
    assert rank_by_volume(m, ctxs(100, 900, 0)) == ["LIVE"]


def test_the_exchanges_own_casing_is_preserved() -> None:
    """Hyperliquid lists "kPEPE", not "KPEPE".

    Upper-casing the name produced a market that every later API call failed
    to find: it could never be refreshed, never had bars, and took the whole
    trading cycle down with it.

    Those two are memecoins and are gated out by default now, so the filter
    is turned off here -- this is a test about casing, and letting a second
    rule decide the outcome would stop it testing the first.
    """
    ranked = rank_by_volume(
        meta("kPEPE", "BTC", "kBONK"), ctxs(9, 8, 7), exclude_memecoins=False
    )
    assert ranked == ["kPEPE", "BTC", "kBONK"]


def test_stablecoins_are_never_traded() -> None:
    names = tuple(EXCLUDED) + ("BTC",)
    ranked = rank_by_volume(meta(*names), ctxs(*([9_000] * len(EXCLUDED)), 1))
    assert ranked == ["BTC"]

    # ...whatever case they are listed in.
    assert rank_by_volume(meta("usdc", "BTC"), ctxs(9_000, 1)) == ["BTC"]


def test_mismatched_universe_and_contexts_raise_rather_than_pair_up_blindly() -> None:
    with pytest.raises(ValueError):
        rank_by_volume(meta("A", "B"), ctxs(1))


class _Client:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def meta_and_asset_ctxs(self):
        if self.error:
            raise self.error
        return self.payload


def test_top_n_returns_the_n_most_traded() -> None:
    client = _Client([meta("A", "B", "C", "D"), ctxs(1, 4, 3, 2)])
    assert resolve("top:2", client) == ("B", "C")


def test_all_returns_every_live_market() -> None:
    client = _Client([meta("A", "B", "C"), ctxs(1, 4, 3)])
    assert resolve("all", client) == ("B", "C", "A")


def test_an_unreachable_exchange_degrades_instead_of_aborting() -> None:
    """A smaller universe is useful; no universe is not."""
    client = _Client(error=OSError("network down"))
    assert resolve("top:25", client, fallback=["BTC", "ETH"]) == ("BTC", "ETH")


def test_a_literal_spec_needs_no_network_at_all() -> None:
    client = _Client(error=AssertionError("must not be called"))
    assert resolve("BTC,ETH", client) == ("BTC", "ETH")


# ==========================================================================
# Memecoins
# ==========================================================================

def test_the_thousand_unit_prefix_does_not_hide_a_memecoin() -> None:
    """Hyperliquid lists 1,000 PEPE as "kPEPE". An entry for PEPE has to
    catch it, or the deny-list silently misses the form actually traded."""
    from data.universe import is_memecoin

    assert is_memecoin("kPEPE")
    assert is_memecoin("kBONK")
    assert is_memecoin("kSHIB")


def test_a_name_that_merely_starts_with_k_is_not_a_prefixed_form() -> None:
    """The prefix is a lowercase k in front of an upper-case name. Stripping
    it from KAITO would leave AITO and invite a false match."""
    from data.universe import is_memecoin

    assert not is_memecoin("KAITO")
    assert not is_memecoin("KAS")
    assert not is_memecoin("kLUNC")


def test_volatility_is_not_the_criterion() -> None:
    """Gaming and NFT tokens are volatile but are not jokes; position sizing
    already handles volatility, so it is not what this list is for."""
    from data.universe import is_memecoin

    for coin in ("SAND", "GALA", "AXS", "APE", "YGG", "ZORA", "BTC", "SOL"):
        assert not is_memecoin(coin), coin


def test_memecoins_are_dropped_from_a_ranked_universe() -> None:
    from data.universe import rank_by_volume

    meta = {"universe": [{"name": n} for n in ("BTC", "DOGE", "ETH", "kPEPE", "SOL")]}
    contexts = [{"dayNtlVlm": v} for v in (900, 800, 700, 600, 500)]

    assert rank_by_volume(meta, contexts, exclude_memecoins=True) == ["BTC", "ETH", "SOL"]
    assert "DOGE" in rank_by_volume(meta, contexts, exclude_memecoins=False)


def test_naming_a_memecoin_explicitly_is_not_a_way_around_the_rule(monkeypatch) -> None:
    """"Do not trade memecoins" is a property of the system, not of one way
    of spelling the universe."""
    import importlib

    monkeypatch.setenv("EXCLUDE_MEMECOINS", "true")
    import config.settings as settings_module
    importlib.reload(settings_module)
    import data.universe as universe_module
    importlib.reload(universe_module)

    assert universe_module.resolve("BTC,DOGE,ETH", client=None) == ("BTC", "ETH")


def test_the_exclusion_can_be_turned_off_deliberately(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("EXCLUDE_MEMECOINS", "false")
    import config.settings as settings_module
    importlib.reload(settings_module)
    import data.universe as universe_module
    importlib.reload(universe_module)

    assert "DOGE" in universe_module.resolve("BTC,DOGE", client=None)

    monkeypatch.delenv("EXCLUDE_MEMECOINS")
    importlib.reload(settings_module)
    importlib.reload(universe_module)
