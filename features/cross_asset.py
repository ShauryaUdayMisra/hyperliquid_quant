"""Cross-asset features: how one market sits relative to the others.

Crypto perps are highly correlated, so a coin's own momentum is partly just
beta to BTC. Separating the two matters: "SOL is up 3%" means something
different when BTC is up 3% than when BTC is flat.

Every statistic is a trailing window over aligned bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import ensure_sorted, log_returns, min_periods, prefix, rolling_z


def compute(
    coin: str,
    bars_by_coin: dict[str, pd.DataFrame],
    *,
    benchmark: str = "BTC",
    windows: tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    """Cross-asset block for ``coin`` against the rest of the universe.

    ``bars_by_coin`` must already be aligned on a common timestamp grid --
    the backtest engine's inner join guarantees this. Misaligned frames
    would silently correlate a coin's Monday with another's Tuesday.
    """
    bars = ensure_sorted(bars_by_coin[coin])
    out = pd.DataFrame(index=bars.index)
    own_returns = log_returns(bars["close"])

    lengths = {c: len(f) for c, f in bars_by_coin.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"coins are not aligned; lengths differ: {lengths}")

    # -- versus the benchmark --
    if benchmark in bars_by_coin and benchmark != coin:
        bench = ensure_sorted(bars_by_coin[benchmark])
        bench_returns = log_returns(bench["close"]).reset_index(drop=True)
        own = own_returns.reset_index(drop=True)

        for window in windows:
            mp = min_periods(window)
            out[f"corr_{benchmark.lower()}_{window}"] = own.rolling(window, min_periods=mp).corr(
                bench_returns
            ).to_numpy()

            covariance = own.rolling(window, min_periods=mp).cov(bench_returns)
            variance = bench_returns.rolling(window, min_periods=mp).var(ddof=1)
            beta = (covariance / variance.replace(0.0, np.nan)).to_numpy()
            out[f"beta_{benchmark.lower()}_{window}"] = beta

            # Residual return: the part of the move the benchmark does not
            # explain. This is where coin-specific information lives.
            out[f"residual_ret_{window}"] = (
                own.rolling(window, min_periods=mp).sum()
                - beta * bench_returns.rolling(window, min_periods=mp).sum()
            ).to_numpy()

        for window in windows:
            own_move = bars["close"] / bars["close"].shift(window) - 1.0
            bench_move = (
                bench["close"].reset_index(drop=True)
                / bench["close"].reset_index(drop=True).shift(window)
                - 1.0
            )
            out[f"rel_strength_{window}"] = (own_move.reset_index(drop=True) - bench_move).to_numpy()

    # -- versus the whole universe --
    returns_matrix = pd.DataFrame(
        {c: log_returns(ensure_sorted(f["close"].to_frame().assign(ts_ms=f["ts_ms"]))["close"]).to_numpy()
         for c, f in bars_by_coin.items()}
    )
    universe_mean = returns_matrix.mean(axis=1)

    for window in windows:
        mp = min_periods(window)
        # Breadth: what fraction of the universe is rising with us.
        agreeing = (np.sign(returns_matrix).T == np.sign(returns_matrix[coin])).T.mean(axis=1)
        out[f"breadth_{window}"] = agreeing.rolling(window, min_periods=mp).mean().to_numpy()
        # Dispersion: when correlations break down, coin-specific signals
        # are worth more and index-like exposure is worth less.
        out[f"dispersion_{window}"] = (
            returns_matrix.std(axis=1, ddof=1).rolling(window, min_periods=mp).mean().to_numpy()
        )
        out[f"excess_vs_universe_{window}"] = (
            (returns_matrix[coin] - universe_mean).rolling(window, min_periods=mp).sum().to_numpy()
        )

    out["excess_vs_universe_24_z"] = rolling_z(out.get("excess_vs_universe_24", pd.Series(index=out.index, dtype=float)), 168)
    return prefix(out, "cross")
