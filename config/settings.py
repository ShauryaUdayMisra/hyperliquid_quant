"""Central configuration for the Hyperliquid paper-trading research system.

Everything that a human might want to tune lives here or in ``.env``.
Nothing in this file may contain a secret; secrets are read from the
environment only.

SAFETY: this project is read-only with respect to Hyperliquid. It never
signs a transaction and never holds a key that can move funds. The
``assert_no_trading_credentials`` guard below is executed at import time
so that an accidentally-populated key aborts the process immediately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------
# Safety guard
# --------------------------------------------------------------------------

#: Environment variables that would imply the ability to sign orders or move
#: funds. If any of these is populated the process refuses to start.
FORBIDDEN_ENV_VARS = (
    "HL_PRIVATE_KEY",
    "HYPERLIQUID_PRIVATE_KEY",
    "PRIVATE_KEY",
    "SECRET_KEY",
    "WALLET_PRIVATE_KEY",
    "HL_API_SECRET",
)


class TradingCredentialsPresent(RuntimeError):
    """Raised when a credential capable of moving real funds is detected."""


def assert_no_trading_credentials(env: dict[str, str] | None = None) -> None:
    """Abort if the environment carries anything that could sign a trade."""
    env = os.environ if env is None else env
    found = [k for k in FORBIDDEN_ENV_VARS if (env.get(k) or "").strip()]
    if found:
        raise TradingCredentialsPresent(
            "This system is paper-trading only and must never hold signing keys. "
            f"Remove these environment variables before starting: {', '.join(found)}"
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _env_path_or_none(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() or None if value else None


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    storage: Path = PROJECT_ROOT / "storage"
    parquet: Path = PROJECT_ROOT / "storage" / "parquet"
    duckdb_file: Path = PROJECT_ROOT / "storage" / "db" / "market.duckdb"
    logs: Path = PROJECT_ROOT / "logs"
    models: Path = PROJECT_ROOT / "models"

    def ensure(self) -> "Paths":
        for p in (self.storage, self.parquet, self.duckdb_file.parent, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------
# Hyperliquid API
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HyperliquidConfig:
    """Public, read-only market-data endpoints."""

    info_url: str = _env("HL_INFO_URL", "https://api.hyperliquid.xyz/info")
    ws_url: str = _env("HL_WS_URL", "wss://api.hyperliquid.xyz/ws")

    #: Optional PEM bundle used to verify TLS. Needed on networks where a
    #: corporate firewall (e.g. FortiGate) performs SSL inspection. Leave
    #: unset to use the system trust store, which is the correct default.
    ca_bundle: str | None = _env_path_or_none("HL_CA_BUNDLE")

    request_timeout_s: float = _env_float("HL_REQUEST_TIMEOUT_S", 20.0)
    max_retries: int = _env_int("HL_MAX_RETRIES", 5)
    backoff_base_s: float = _env_float("HL_BACKOFF_BASE_S", 0.5)
    backoff_max_s: float = _env_float("HL_BACKOFF_MAX_S", 30.0)

    #: Hyperliquid budgets the /info endpoint by request weight, 1200 per
    #: minute per IP. We stay well under it.
    rate_limit_per_minute: int = _env_int("HL_RATE_LIMIT_PER_MINUTE", 600)

    #: Hard server-side cap on rows returned by a single candleSnapshot call.
    candle_page_limit: int = 5000

    #: Seconds between WebSocket keepalive pings (server drops idle at 60s).
    ws_ping_interval_s: float = _env_float("HL_WS_PING_INTERVAL_S", 45.0)

    def verify(self) -> str | bool:
        """Value to hand to httpx/ssl as the ``verify`` argument."""
        return self.ca_bundle if self.ca_bundle else True


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    #: Raw MARKETS value. May be a literal list, "top:N" or "all"; the
    #: ranked forms need the exchange to resolve, which is why the spec is
    #: kept verbatim and resolution happens in data.universe at startup.
    markets_spec: str = _env("MARKETS", "BTC,ETH,SOL")

    #: The literal coins named in the spec. For a ranked spec this is empty
    #: until data.universe.resolve() has run.
    markets: tuple[str, ...] = tuple(
        m.strip().upper()
        for m in _env("MARKETS", "BTC,ETH,SOL").split(",")
        if m.strip() and ":" not in m and m.strip().lower() != "all"
    )

    #: Bars of history the live loop reads per market. Enough to cover
    #: features.pipeline.MAX_LOOKBACK_BARS with room to spare. Uncapped, a
    #: large universe would load every stored bar for every coin on every
    #: cycle, which is where a 3-coin system stops scaling.
    #: Memecoins are not traded. There is no exchange field that identifies
    #: one, so data.universe.MEMECOINS is a hand-maintained list and the
    #: resolver logs every name it drops.
    exclude_memecoins: bool = _env_bool("EXCLUDE_MEMECOINS", True)

    #: Further names to gate out, on top of the built-in lists.
    excluded_markets: tuple[str, ...] = tuple(
        m.strip().upper()
        for m in _env("EXCLUDED_MARKETS", "").split(",")
        if m.strip()
    )

    live_lookback_bars: int = _env_int("LIVE_LOOKBACK_BARS", 1_000)

    #: Candle intervals to backfill and keep in sync.
    candle_intervals: tuple[str, ...] = tuple(
        i.strip() for i in _env("CANDLE_INTERVALS", "1m,5m,1h").split(",") if i.strip()
    )

    #: How far back the initial historical backfill reaches.
    backfill_days: int = _env_int("BACKFILL_DAYS", 180)

    #: Seconds between live order-book snapshots written by the collector.
    orderbook_snapshot_interval_s: float = _env_float("ORDERBOOK_SNAPSHOT_INTERVAL_S", 10.0)

    #: Order-book depth (levels per side) retained in a snapshot.
    orderbook_depth: int = _env_int("ORDERBOOK_DEPTH", 20)

    #: Seconds between asset-context (funding / OI / mark) polls.
    asset_ctx_interval_s: float = _env_float("ASSET_CTX_INTERVAL_S", 60.0)

    #: Flush the in-memory buffers to Parquet at least this often.
    flush_interval_s: float = _env_float("FLUSH_INTERVAL_S", 60.0)

    #: A timestamp more than this far in the future is data corruption.
    #: Small positive tolerance absorbs exchange/host clock skew.
    future_timestamp_tolerance_s: float = _env_float("FUTURE_TS_TOLERANCE_S", 5.0)


#: Candle interval -> milliseconds. Used for gap detection.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

#: Hyperliquid funds perpetuals every hour.
FUNDING_INTERVAL_MS = 3_600_000


# --------------------------------------------------------------------------
# Risk limits
#
# Two named profiles, selected with RISK_PROFILE in .env, so the same code
# can be run twice and the results compared. The risk engine is identical in
# both cases -- it always sizes positions, tracks margin and enforces
# liquidation. Only the numbers it enforces change.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskLimits:
    """Hard limits the risk engine enforces. The engine may veto any signal."""

    name: str = "conservative"
    starting_capital: float = 100_000.0
    max_position_usd: float = 10_000.0
    max_leverage: float = 2.0
    #: Fraction of equity risked per trade (distance to stop x size).
    risk_per_trade: float = 0.0025
    #: Trading halts for the rest of the UTC day past this loss fraction.
    max_daily_loss: float = 0.02
    #: Trading halts entirely past this peak-to-trough equity drawdown.
    max_portfolio_dd: float = 0.10
    max_open_positions: int = 3

    @property
    def daily_loss_cap_active(self) -> bool:
        return self.max_daily_loss < 1.0

    @property
    def drawdown_cap_active(self) -> bool:
        return self.max_portfolio_dd < 1.0

    def describe(self) -> str:
        return (
            f"risk profile '{self.name}': capital=${self.starting_capital:,.0f} "
            f"max_pos=${self.max_position_usd:,.0f} lev={self.max_leverage}x "
            f"risk/trade={self.risk_per_trade:.2%} "
            f"daily_loss={'off' if not self.daily_loss_cap_active else f'{self.max_daily_loss:.1%}'} "
            f"max_dd={'off' if not self.drawdown_cap_active else f'{self.max_portfolio_dd:.1%}'} "
            f"max_positions={self.max_open_positions}"
        )


#: The original brief's limits. Survivable, slow, hard to blow up.
CONSERVATIVE_RISK = RiskLimits(
    name="conservative",
    max_position_usd=10_000.0,
    max_leverage=2.0,
    risk_per_trade=0.0025,
    max_daily_loss=0.02,
    max_portfolio_dd=0.10,
    max_open_positions=3,
)

#: Full-deployment, high-leverage profile with the circuit breakers removed.
#: Position sizing, margin tracking and liquidation still apply -- with no
#: daily-loss or drawdown halt, liquidation becomes the ONLY backstop.
AGGRESSIVE_RISK = RiskLimits(
    name="aggressive",
    max_position_usd=100_000.0,
    max_leverage=10.0,
    risk_per_trade=1.0,
    max_daily_loss=1.0,
    max_portfolio_dd=1.0,
    max_open_positions=3,
)

RISK_PROFILES: dict[str, RiskLimits] = {
    "conservative": CONSERVATIVE_RISK,
    "aggressive": AGGRESSIVE_RISK,
}


def resolve_risk_profile(name: str | None = None) -> RiskLimits:
    """Look up a profile, then apply any per-field .env overrides."""
    key = (name or _env("RISK_PROFILE", "aggressive")).strip().lower()
    if key not in RISK_PROFILES:
        raise ValueError(
            f"unknown RISK_PROFILE '{key}'; choose one of {sorted(RISK_PROFILES)}"
        )
    base = RISK_PROFILES[key]
    return RiskLimits(
        name=base.name,
        starting_capital=_env_float("STARTING_CAPITAL", base.starting_capital),
        max_position_usd=_env_float("MAX_POSITION_USD", base.max_position_usd),
        max_leverage=_env_float("MAX_LEVERAGE", base.max_leverage),
        risk_per_trade=_env_float("RISK_PER_TRADE", base.risk_per_trade),
        max_daily_loss=_env_float("MAX_DAILY_LOSS", base.max_daily_loss),
        max_portfolio_dd=_env_float("MAX_PORTFOLIO_DD", base.max_portfolio_dd),
        max_open_positions=_env_int("MAX_OPEN_POSITIONS", base.max_open_positions),
    )


# --------------------------------------------------------------------------
# Execution realism (Phase 2 paper exchange)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionConfig:
    """Costs and frictions the paper exchange charges.

    Defaults are Hyperliquid's base-tier perp fees. They are deliberately
    pessimistic where the true value is unknowable in a backtest.
    """

    taker_fee: float = _env_float("TAKER_FEE", 0.00045)   # 4.5 bps
    maker_fee: float = _env_float("MAKER_FEE", 0.00015)   # 1.5 bps

    #: Half-spread applied to every market order, as a fraction of price,
    #: used when no live order book is available for the bar.
    default_half_spread: float = _env_float("DEFAULT_HALF_SPREAD", 0.00005)

    #: Square-root market-impact coefficient: impact = k * sqrt(notional / adv).
    impact_coefficient: float = _env_float("IMPACT_COEFFICIENT", 0.10)

    #: Round-trip decision-to-fill delay. Signals act on the NEXT bar anyway;
    #: this adds sub-bar slippage on top.
    latency_ms: int = _env_int("LATENCY_MS", 250)

    #: A single order may not consume more than this share of a bar's volume.
    #: Anything beyond it becomes a partial fill.
    max_bar_volume_share: float = _env_float("MAX_BAR_VOLUME_SHARE", 0.10)

    #: The most market impact a single order is allowed to expect to pay, in
    #: basis points. This is a *sizing* budget, not a fill constraint: it
    #: decides how large a position may be in a given market, where
    #: max_bar_volume_share only decides how much of it can fill at once.
    #:
    #: It exists because a fixed dollar cap is a different trade in every
    #: market. $10,000 is 0.01% of BTC's hourly volume and 5.6% of WLD's,
    #: which under square-root impact is 9bp of cost against 237bp -- the
    #: same order that is nearly free on BTC costs eight times the entire
    #: move the model is trying to predict. Live, this was the whole loss:
    #: 1.67% average slippage per fill against a 0.30% target.
    #: 10bp each way puts a round trip at 30bp against a 1.00% label -- the
    #: cost takes about a third of the move being predicted. Tighter than
    #: that and only BTC can hold a meaningful position; looser and the
    #: costs eat the thesis.
    max_impact_bps: float = _env_float("MAX_IMPACT_BPS", 10.0)

    #: Maintenance-margin fraction fallback when an asset's max leverage is
    #: unknown. Hyperliquid uses half the initial margin at max leverage,
    #: i.e. 1 / (2 * max_leverage).
    default_max_asset_leverage: float = _env_float("DEFAULT_MAX_ASSET_LEVERAGE", 20.0)

    #: Extra cost charged when a position is force-closed by liquidation.
    liquidation_penalty: float = _env_float("LIQUIDATION_PENALTY", 0.01)

    def max_notional_for_impact(self, bar_notional: float) -> float:
        """Largest order whose expected impact stays inside the budget.

        Inverts the simulator's ``impact = k * sqrt(participation)``, so the
        rule that sizes a position and the model that charges it for the
        fill can never disagree.
        """
        if bar_notional <= 0 or self.impact_coefficient <= 0:
            return 0.0
        if self.max_impact_bps <= 0:
            return float("inf")
        participation = (self.max_impact_bps / 10_000.0 / self.impact_coefficient) ** 2
        return float(bar_notional * min(1.0, participation))

    def round_trip_cost(self) -> float:
        """What getting into a position and back out is expected to cost.

        Two taker fees, two half-spreads and two lots of impact at the
        sizing budget. This is the number a label's threshold has to clear
        before the question is worth asking at all: predicting a move
        smaller than this is predicting something unprofitable even when
        the prediction is right.
        """
        impact = self.max_impact_bps / 10_000.0
        return 2.0 * (self.taker_fee + self.default_half_spread + impact)

    def maintenance_margin_fraction(self, max_asset_leverage: float | None = None) -> float:
        leverage = max_asset_leverage or self.default_max_asset_leverage
        return 1.0 / (2.0 * leverage)


# --------------------------------------------------------------------------
# Reporting (Phase 7 — configured here, not used yet)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportConfig:
    smtp_host: str = _env("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = _env_int("SMTP_PORT", 587)
    smtp_user: str = _env("SMTP_USER", "")
    smtp_app_password: str = os.getenv("SMTP_APP_PASSWORD", "")
    sender: str = _env("REPORT_SENDER", "")
    #: No default. An address baked into source gets scraped from any
    #: public repository, and a report silently going to the wrong
    #: inbox is worse than one that refuses to send.
    recipient: str = _env("REPORT_RECIPIENT", "")
    #: Local hours at which the 6-hourly report fires.
    schedule_hours: tuple[int, ...] = tuple(
        int(h) for h in _env("REPORT_HOURS", "0,6,12,18").split(",") if h.strip()
    )
    enabled: bool = _env_bool("REPORT_ENABLED", False)


# --------------------------------------------------------------------------
# Root settings object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyConfig:
    """Activity rules that sit outside the model's opinion."""

    #: Hard cap on how long any position may be held, in hours. The label
    #: the model was trained on asks about a short horizon, so a position
    #: carried well past it is being held on an expired forecast. Closing
    #: and re-entering is allowed, so a persistent signal survives.
    max_hold_hours: float = _env_float("MAX_HOLD_HOURS", 24.0)

    #: A position is not closed on a fading probability before this age.
    #: Risk exits -- the holding cap and liquidation -- ignore it.
    min_hold_hours: float = _env_float("MIN_HOLD_HOURS", 4.0)

    #: After this many hours holding nothing, open the strongest candidate
    #: regardless of the entry threshold. Buys activity, not accuracy --
    #: these are by construction the trades the model was not confident
    #: enough to ask for. Measured in wall-clock time rather than bars
    #: counted in memory, so a restart cannot reset the clock. 0 disables.
    max_idle_hours: float = _env_float("MAX_IDLE_HOURS", 6.0)

    @property
    def max_hold_ms(self) -> int | None:
        return int(self.max_hold_hours * 3_600_000) if self.max_hold_hours > 0 else None

    @property
    def min_hold_ms(self) -> int | None:
        return int(self.min_hold_hours * 3_600_000) if self.min_hold_hours > 0 else None

    @property
    def max_idle_ms(self) -> int | None:
        return int(self.max_idle_hours * 3_600_000) if self.max_idle_hours > 0 else None

    def describe(self) -> str:
        hold = (f"hold {self.min_hold_hours:g}-{self.max_hold_hours:g}h"
                if self.max_hold_ms else f"minimum hold {self.min_hold_hours:g}h")
        idle = (f"force an entry after {self.max_idle_hours:g}h flat"
                if self.max_idle_ms else "never force an entry")
        return f"{hold}; {idle}"


@dataclass(frozen=True)
class LabelSettings:
    """The question the models are asked.

    Configurable because it is the main lever on whether the system can pay
    for itself. The default horizon used to be 4 bars at 0.30%, which asked
    the model to predict a move smaller than a round trip costs to trade in
    most markets -- so being right earned less than being in. A longer
    horizon lets the threshold clear costs without demanding the model
    become more accurate.
    """

    horizon_bars: int = _env_int("LABEL_HORIZON_BARS", 24)
    threshold: float = _env_float("LABEL_THRESHOLD", 0.010)

    def describe(self) -> str:
        return (f"{self.horizon_bars}-bar horizon, "
                f"{self.threshold:.2%} move")


@dataclass(frozen=True)
class LearningConfig:
    """How the deployed model is kept current with what has happened since.

    Without this the model is frozen at whatever the first boot trained: it
    never sees a single bar of the market it is trading, so it can be wrong
    about the same setup every day and never learn anything from it.
    Retraining is scheduled on wall-clock time rather than on cycles, for
    the same reason the idle clock is -- a counter in memory resets on every
    redeploy.
    """

    #: Retrain the live model on a schedule. Off means the model is frozen
    #: at whatever the first boot produced, forever.
    enabled: bool = _env_bool("RETRAIN_ENABLED", True)

    #: Hours between retrains, measured from the model's own
    #: ``trained_at_ms``. A restart therefore does not reset the clock, and
    #: a model that is already stale retrains on the first cycle after boot
    #: rather than a full period later.
    every_hours: float = _env_float("RETRAIN_EVERY_HOURS", 24.0)

    #: Do not retrain until at least this many new bars have been stored
    #: since the model was fitted. Refitting on the same data spends minutes
    #: of CPU to produce a reseeded copy of the same model. Counted as
    #: distinct bar timestamps, so the number means "hours of new market"
    #: whether three markets are being traded or two hundred, and one
    #: lagging market cannot hold the count down.
    min_new_bars: int = _env_int("RETRAIN_MIN_NEW_BARS", 12)

    #: A candidate is rejected if its walk-forward AUC is this far below the
    #: incumbent's. Deliberately loose: at an AUC near 0.50 the difference
    #: between two models is mostly noise, and fresher data breaks the tie.
    min_auc_margin: float = _env_float("RETRAIN_MIN_AUC_MARGIN", 0.02)

    @property
    def every_ms(self) -> int | None:
        if not self.enabled or self.every_hours <= 0:
            return None
        return int(self.every_hours * 3_600_000)

    def describe(self) -> str:
        if self.every_ms is None:
            return "retraining OFF - the model stays frozen at its first fit"
        return (
            f"retrain every {self.every_hours:g}h on all stored history "
            f"(needs {self.min_new_bars}+ new bars; a candidate more than "
            f"{self.min_auc_margin:.2f} AUC below the incumbent is rejected)"
        )


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskLimits = field(default_factory=resolve_risk_profile)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    label: LabelSettings = field(default_factory=LabelSettings)
    learning: LearningConfig = field(default_factory=LearningConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    #: Global kill-switch documenting intent. Nothing in this repo may place
    #: a real order regardless of its value.
    paper_trading_only: bool = True

    log_level: str = _env("LOG_LEVEL", "INFO")


def get_settings() -> Settings:
    assert_no_trading_credentials()
    settings = Settings()
    settings.paths.ensure()
    return settings


SETTINGS = get_settings()
