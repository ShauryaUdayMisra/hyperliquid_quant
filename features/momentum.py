"""Momentum and trend features. All trailing, all causal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import ensure_sorted, log_returns, min_periods, prefix, rolling_z

#: Lookbacks in bars. On 1h bars these are 1h/4h/12h/24h/3d/7d.
DEFAULT_LOOKBACKS = (1, 4, 12, 24, 72, 168)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, computed with an exponential average of past bars only."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)

    # A window with no down bars divides by zero. That is a genuine extreme
    # -- maximally overbought -- so it maps to 100 rather than to NaN, which
    # would silently discard the strongest reading the indicator can give.
    warmed = avg_gain.notna() & avg_loss.notna()
    result = result.mask(warmed & (avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask(warmed & (avg_loss == 0) & (avg_gain == 0), 50.0)
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD normalised by price, so it is comparable across coins."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = (ema_fast - ema_slow) / close
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def efficiency_ratio(close: pd.Series, window: int = 24) -> pd.Series:
    """Kaufman efficiency: net move divided by total path length, in [0, 1].

    Near 1 the market is trending cleanly; near 0 it is chopping. This is
    the backbone of the regime classifier.
    """
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=min_periods(window)).sum()
    return net / path.replace(0.0, np.nan)


def distance_from_extreme(close: pd.Series, window: int = 168) -> pd.DataFrame:
    """Where price sits inside its trailing range.

    The window includes the current bar, which is legitimate: the current
    close is known at decision time. It must NOT include any later bar.
    """
    high = close.rolling(window, min_periods=min_periods(window)).max()
    low = close.rolling(window, min_periods=min_periods(window)).min()
    span = (high - low).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            f"pct_of_range_{window}": (close - low) / span,
            f"drawdown_from_high_{window}": close / high - 1.0,
            f"gain_from_low_{window}": close / low - 1.0,
        }
    )


def compute(bars: pd.DataFrame, lookbacks=DEFAULT_LOOKBACKS) -> pd.DataFrame:
    """Full momentum block for one coin."""
    bars = ensure_sorted(bars)
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)

    for n in lookbacks:
        # Simple return over n bars: uses close at t and t-n, both known.
        out[f"ret_{n}"] = close / close.shift(n) - 1.0
        # Standardised so a 1% BTC move and a 1% SOL move are comparable.
        out[f"ret_{n}_z"] = rolling_z(out[f"ret_{n}"], max(24, n * 4))

    log_ret = log_returns(close)
    for n in (12, 48):
        # Mean log return per bar: a smoother trend estimate than raw return.
        out[f"trend_{n}"] = log_ret.rolling(n, min_periods=min_periods(n)).mean()

    for fast, slow in ((12, 48), (24, 168)):
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        out[f"ema_cross_{fast}_{slow}"] = ema_fast / ema_slow - 1.0

    out["rsi_14"] = rsi(close, 14) / 100.0
    out["rsi_48"] = rsi(close, 48) / 100.0
    out = out.join(macd(close))
    out["efficiency_24"] = efficiency_ratio(close, 24)
    out["efficiency_72"] = efficiency_ratio(close, 72)
    out = out.join(distance_from_extreme(close, 168))

    return prefix(out, "mom")
