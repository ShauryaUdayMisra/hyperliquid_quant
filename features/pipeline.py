"""Assembles every feature block into one point-in-time matrix per coin.

The pipeline is deliberately boring: it calls each block, concatenates the
results, and attaches the regime label. The interesting part is
:func:`assert_point_in_time`, which empirically proves the whole matrix is
causal instead of taking the individual modules' word for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import INTERVAL_MS
from models.labels import LABEL_COLUMNS
from features import cross_asset, funding as funding_features, momentum, orderbook, regime, volatility, volume
from features.base import ensure_sorted

log = logging.getLogger(__name__)

#: Longest lookback any block uses. Rows before this have NaN-heavy features
#: and must not be fed to a model as if they were complete.
MAX_LOOKBACK_BARS = 720


@dataclass
class FeatureConfig:
    interval: str = "1h"
    benchmark: str = "BTC"
    include_orderbook: bool = True
    include_cross_asset: bool = True

    @property
    def interval_ms(self) -> int:
        return INTERVAL_MS[self.interval]

    @property
    def bars_per_year(self) -> float:
        return 365.25 * 24 * 3600 * 1000 / self.interval_ms


def build_for_coin(
    coin: str,
    bars: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    book_snapshots: pd.DataFrame | None = None,
    bars_by_coin: dict[str, pd.DataFrame] | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Full feature matrix for one coin, indexed like ``bars``."""
    config = config or FeatureConfig()
    bars = ensure_sorted(bars)

    # Funding must be merged before the funding block runs, and the merge
    # is backward-looking so no unpublished rate can leak in.
    if funding is not None and not funding.empty:
        bars = funding_features.attach_funding_to_bars(bars, funding)

    blocks = [
        momentum.compute(bars),
        volatility.compute(bars, bars_per_year=config.bars_per_year),
        volume.compute(bars),
        funding_features.compute(bars),
    ]

    if config.include_orderbook:
        blocks.append(orderbook.compute(bars, book_snapshots, config.interval_ms))

    if config.include_cross_asset and bars_by_coin and len(bars_by_coin) > 1:
        blocks.append(cross_asset.compute(coin, bars_by_coin, benchmark=config.benchmark))

    matrix = pd.concat([b for b in blocks if not b.empty], axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]

    labels = regime.classify(bars)
    matrix = pd.concat([matrix, labels], axis=1)

    matrix.insert(0, "ts_ms", bars["ts_ms"].to_numpy())
    matrix.insert(1, "coin", coin)
    matrix.insert(2, "close", bars["close"].to_numpy())

    # Infinities come from a zero denominator that slipped past a guard.
    # They break LightGBM silently, so convert them here and let the NaN
    # accounting below surface the problem.
    numeric = matrix.select_dtypes(include=[np.number]).columns
    matrix[numeric] = matrix[numeric].replace([np.inf, -np.inf], np.nan)
    return matrix


def align_bars(
    bars_by_coin: dict[str, pd.DataFrame], *, min_history_share: float = 0.5
) -> dict[str, pd.DataFrame]:
    """Put every market on one timestamp grid before features are built.

    The cross-asset block refuses to run on frames of different lengths, and
    it is right to: correlating one coin's Monday with another's Tuesday is
    worse than having no cross-asset feature at all. But in real storage the
    markets are never equal -- one was topped up a minute later than the
    next, one was listed last week -- so something has to reconcile them,
    and an exception is not it. Unaligned markets raised
    ``coins are not aligned`` from deep inside the feature build, which took
    down the whole call: the "one bad market must never stop the rest" rule,
    in the one place a scheduled retrain would meet it every time.

    Two steps, in this order:

    1. **Drop the stragglers.** A market with a fraction of everyone else's
       history is usually newly listed. Intersecting with it would throw
       away months of every other market's bars to accommodate it, so it is
       dropped instead and the rest keep their depth.
    2. **Inner-join the survivors.** Only the timestamps they all share.
       Never forward-fill: that invents a price, and back-filling a coin
       that started later gives it a history it did not have.

    Callers must align *bars*, not the matrices built from them: the live
    loop and the backtest both index a coin's feature frame by the same
    position they index its bars, so a matrix that is shorter than its bars
    would score the wrong row.
    """
    usable = {c: f for c, f in bars_by_coin.items() if not f.empty}
    if len(usable) < 2:
        return usable

    lengths = {c: len(f) for c, f in usable.items()}
    longest = max(lengths.values())
    floor = longest * min_history_share
    short = sorted(c for c, n in lengths.items() if n < floor)
    if short:
        log.warning(
            "%d market(s) have less than %.0f%% of the deepest history and are "
            "left out of this feature build rather than truncating everyone to "
            "them: %s",
            len(short), min_history_share * 100,
            ", ".join(f"{c} ({lengths[c]:,} bars)" for c in short),
        )
        usable = {c: f for c, f in usable.items() if c not in set(short)}
        if len(usable) < 2:
            return usable

    common: set[int] | None = None
    for frame in usable.values():
        stamps = set(frame["ts_ms"].astype("int64"))
        common = stamps if common is None else (common & stamps)
    if not common:
        raise ValueError(
            f"{len(usable)} market(s) share no common timestamp; they cannot be "
            "put on one grid"
        )

    keep = np.array(sorted(common), dtype=np.int64)
    dropped = max(lengths[c] for c in usable) - len(keep)
    if dropped:
        log.info(
            "aligned %d market(s) onto %d shared bar(s); %d unshared bar(s) set "
            "aside", len(usable), len(keep), dropped,
        )
    return {
        coin: (
            frame if len(frame) == len(keep) and frame["ts_ms"].is_monotonic_increasing
            else frame.loc[frame["ts_ms"].isin(keep)]
                      .sort_values("ts_ms")
                      .reset_index(drop=True)
        )
        for coin, frame in usable.items()
    }


def build_universe(
    bars_by_coin: dict[str, pd.DataFrame],
    *,
    funding_by_coin: dict[str, pd.DataFrame] | None = None,
    book_by_coin: dict[str, pd.DataFrame] | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or FeatureConfig()
    funding_by_coin = funding_by_coin or {}
    book_by_coin = book_by_coin or {}
    matrices = {
        coin: build_for_coin(
            coin,
            bars,
            funding=funding_by_coin.get(coin),
            book_snapshots=book_by_coin.get(coin),
            bars_by_coin=bars_by_coin,
            config=config,
        )
        for coin, bars in bars_by_coin.items()
    }
    return _align_columns(matrices)


def _align_columns(matrices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Give every coin the same columns, filling absences with NaN.

    The benchmark has no correlation-with-itself features, so its matrix is
    naturally narrower than the others. Stacking for training papers over
    this (``concat`` fills the gaps with NaN), but inference reads one coin's
    matrix at a time and would find columns simply missing -- so the
    benchmark could never produce a signal at all.

    Aligning here makes inference see exactly what training saw: the column
    present, the value NaN, which gradient-boosted trees handle natively.
    """
    if len(matrices) < 2:
        return matrices
    union: list[str] = []
    for frame in matrices.values():
        union.extend(c for c in frame.columns if c not in union)
    return {
        coin: frame.reindex(columns=union) if list(frame.columns) != union else frame
        for coin, frame in matrices.items()
    }


#: Never model inputs. Metadata plus anything derived from the future.
EXCLUDED_COLUMNS = frozenset(
    {"ts_ms", "close", "coin", "regime"} | set(LABEL_COLUMNS)
)


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model-eligible columns: numeric, not metadata, and not label-derived.

    Omitting the label columns here once cost a perfect 1.000 AUC: the
    trainer happily used ``label`` as a feature and predicted itself.
    """
    return [
        c for c in matrix.columns
        if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(matrix[c])
    ]


def coverage(matrix: pd.DataFrame) -> pd.DataFrame:
    """Non-null share per feature. Anything near zero is a broken input."""
    columns = feature_columns(matrix)
    usable = matrix.iloc[MAX_LOOKBACK_BARS:] if len(matrix) > MAX_LOOKBACK_BARS else matrix
    return (
        pd.DataFrame(
            {
                "feature": columns,
                "non_null": [usable[c].notna().mean() for c in columns],
            }
        )
        .sort_values("non_null")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# The point-in-time proof
# --------------------------------------------------------------------------

def assert_point_in_time(
    bars_by_coin: dict[str, pd.DataFrame],
    *,
    coin: str | None = None,
    checkpoints: int = 6,
    config: FeatureConfig | None = None,
    funding_by_coin: dict[str, pd.DataFrame] | None = None,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Prove no feature at bar *t* depends on any bar after *t*.

    Method: compute the matrix on the full history, then recompute it on
    history truncated at *t* and compare row *t*. If a feature peeked
    forward, hiding the future changes its value and the comparison fails.

    Returns a per-checkpoint report; an empty ``mismatches`` column
    everywhere means the pipeline is causal.
    """
    config = config or FeatureConfig()
    coin = coin or next(iter(bars_by_coin))
    funding_by_coin = funding_by_coin or {}

    full = build_universe(bars_by_coin, funding_by_coin=funding_by_coin, config=config)[coin]
    columns = feature_columns(full)
    n = len(full)

    # Skip the warmup region, where almost everything is legitimately NaN.
    start = min(MAX_LOOKBACK_BARS, n // 2)
    positions = np.linspace(start, n - 1, num=checkpoints, dtype=int)

    rows = []
    for t in sorted(set(int(p) for p in positions)):
        truncated_bars = {c: f.iloc[: t + 1].copy() for c, f in bars_by_coin.items()}
        truncated_funding = {
            c: f[f["ts_ms"] <= int(bars_by_coin[coin]["ts_ms"].iloc[t])].copy()
            for c, f in funding_by_coin.items()
        }
        partial = build_universe(
            truncated_bars, funding_by_coin=truncated_funding, config=config
        )[coin]

        mismatches = []
        for column in columns:
            a = full[column].iloc[t]
            b = partial[column].iloc[t]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                mismatches.append(column)
                continue
            if not np.isclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True):
                mismatches.append(column)

        rows.append(
            {
                "bar": t,
                "ts_ms": int(full["ts_ms"].iloc[t]),
                "features_checked": len(columns),
                "mismatches": mismatches,
                "ok": not mismatches,
            }
        )

    report = pd.DataFrame(rows)
    if not report["ok"].all():
        leaking = sorted({c for row in report["mismatches"] for c in row})
        log.error("LOOK-AHEAD DETECTED in: %s", leaking)
    return report
