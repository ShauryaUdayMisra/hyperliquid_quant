"""Phase 7 dashboard: a live read-only view of the paper account.

Reads the same Parquet performance record the 6-hour email uses, so the
dashboard and the report can never disagree with each other. It holds no
state, has no write endpoints, and cannot place an order.

Bind it to localhost. There is no authentication, so reach it over an SSH
tunnel rather than exposing the port:

    ssh -L 8000:localhost:8000 user@your-server
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config.settings import INTERVAL_MS, SETTINGS
from data.database import MarketDatabase, ParquetStore
from execution.paper_exchange import PaperExchange
from execution.simulator import FillSimulator
from execution.state_store import StateStore
from reporting.report_builder import PlanLine, market_reasoning

log = logging.getLogger(__name__)

app = FastAPI(title="Hyperliquid Paper Trading", docs_url=None, redoc_url=None)

# --------------------------------------------------------------------------
# Access control
#
# Locally this is unnecessary; on a public host it is the difference between
# a private dashboard and one anyone with the URL can read. Auth switches on
# automatically as soon as DASHBOARD_PASSWORD is set, so a deploy that
# forgets it fails loudly at startup rather than silently going public.
# --------------------------------------------------------------------------

DASHBOARD_USER = os.getenv("DASHBOARD_USER", "admin")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
PUBLIC_DEPLOY = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PUBLIC_DEPLOY"))

if PUBLIC_DEPLOY and not DASHBOARD_PASSWORD:
    raise RuntimeError(
        "This looks like a public deployment but DASHBOARD_PASSWORD is not set. "
        "Refusing to serve the account dashboard without a password. Set "
        "DASHBOARD_PASSWORD in the host's environment variables."
    )

_security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials | None = Depends(_security)) -> None:
    """No-op when no password is configured; enforced the moment one is."""
    if not DASHBOARD_PASSWORD:
        return
    ok = credentials is not None and secrets.compare_digest(
        credentials.username, DASHBOARD_USER
    ) and secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorised",
            headers={"WWW-Authenticate": "Basic"},
        )


#: Timezone every time on the page is shown in, and the one whose midnight
#: starts "today" for the daily P&L. The trader itself is UTC throughout --
#: this is presentation only, applied at both ends so the label and the
#: number cannot disagree.
DISPLAY_TZ = os.environ.get("DISPLAY_TIMEZONE", "Asia/Kolkata").strip() or "UTC"

try:
    _DISPLAY_ZONE = ZoneInfo(DISPLAY_TZ)
except Exception:  # noqa: BLE001 - a bad tz name must not stop the dashboard
    log.warning("unknown DISPLAY_TIMEZONE %r; falling back to UTC", DISPLAY_TZ)
    DISPLAY_TZ = "UTC"
    _DISPLAY_ZONE = ZoneInfo("UTC")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _local_midnight_ms(now_ms: int) -> int:
    """Start of the current day in the display timezone.

    Using UTC midnight while showing IST clocks would put the daily P&L
    boundary at 5:30am for the reader, which is a quietly wrong number
    rather than a visibly wrong one.
    """
    local = datetime.fromtimestamp(now_ms / 1000, tz=_DISPLAY_ZONE)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


def _trade_interval() -> str:
    """The bar the trader decides on. start.sh passes it as an argument, and
    the dashboard cannot see argv -- so read the same variable it does."""
    name = os.environ.get("TRADE_INTERVAL", "1h")
    return name if name in INTERVAL_MS else "1h"


def _interval_ms() -> int:
    return INTERVAL_MS[_trade_interval()]


def _next_decision_ms(after_ms: int) -> int:
    """The next bar boundary the trader will wake on.

    Mirrors PaperTrader._seconds_to_next_bar, including the 15s it waits
    for the exchange to publish the closing bar.
    """
    interval = _interval_ms()
    return ((after_ms // interval) + 1) * interval + 15_000


HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _store() -> ParquetStore:
    return ParquetStore(SETTINGS.paths)


def _configured_markets() -> list[str]:
    """The markets the trader is actually running, as it recorded them.

    Re-resolving "top:25" here would ask the exchange a second time and
    could disagree with the trader if volumes shifted, so the decisions
    table -- written by the trader itself -- is the source of truth, with
    the literal setting as a fallback before the first cycle lands.
    """
    try:
        frame = _query("SELECT DISTINCT coin FROM decisions")
        coins = sorted(str(c) for c in frame["coin"]) if not frame.empty else []
    except Exception:  # noqa: BLE001 - the dashboard must render regardless
        coins = []
    return coins or list(SETTINGS.data.markets)


def _signal_threshold() -> float:
    """What start.sh passes to the trader; the dashboard cannot see argv."""
    try:
        return float(os.environ.get("SIGNAL_THRESHOLD", "0.55"))
    except ValueError:
        return 0.55


def _load_account() -> PaperExchange:
    exchange, _ = _load_account_and_extra()
    return exchange


def _load_account_and_extra() -> tuple[PaperExchange, dict]:
    exchange = PaperExchange(
        SETTINGS.risk.starting_capital,
        config=SETTINGS.execution,
        simulator=FillSimulator(SETTINGS.execution),
    )
    extra = StateStore(SETTINGS.paths.storage / "paper_account.json").load_into(exchange)
    return exchange, extra or {}


def _query(sql: str, params=None) -> pd.DataFrame:
    try:
        with MarketDatabase(_store()) as db:
            return db.query(sql, params or [])
    except Exception as exc:  # noqa: BLE001 - an empty panel beats a 500
        log.warning("dashboard query failed: %s", exc)
        return pd.DataFrame()


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


# --------------------------------------------------------------------------
# P&L over three horizons
# --------------------------------------------------------------------------

def _pnl_since(curve: pd.DataFrame, since_ms: int, equity: float) -> float | None:
    """Change in equity since ``since_ms``.

    Returns None when no reading exists from before the cutoff -- the honest
    answer for "P&L this hour" when the trader started ten minutes ago, and
    better than reporting the whole run's P&L as if it happened in an hour.
    """
    if curve.empty:
        return None
    earlier = curve.loc[curve["ts_ms"] <= since_ms]
    if earlier.empty:
        return None
    return equity - float(earlier["equity"].iloc[-1])


@app.get("/api/pnl", dependencies=[Depends(require_auth)])
def api_pnl() -> JSONResponse:
    exchange = _load_account()
    marks = {p.coin: p.entry_price for p in exchange.open_positions()}
    equity = exchange.equity(marks)

    curve = (
        _query("SELECT ts_ms, equity FROM equity ORDER BY ts_ms")
        if _store().has_data("equity")
        else pd.DataFrame()
    )
    now_ms = int(time.time() * 1000)

    return JSONResponse({
        "equity": equity,
        "starting_capital": exchange.starting_capital,
        "lifetime": equity - exchange.starting_capital,
        "lifetime_pct": (equity / exchange.starting_capital - 1.0)
        if exchange.starting_capital else 0.0,
        "day": _pnl_since(curve, _local_midnight_ms(now_ms), equity),
        "hour": _pnl_since(curve, now_ms - HOUR_MS, equity),
        "readings": int(len(curve)),
        "timezone": DISPLAY_TZ,
    })


@app.get("/api/equity", dependencies=[Depends(require_auth)])
def api_equity(limit: int = 2000) -> JSONResponse:
    if not _store().has_data("equity"):
        return JSONResponse({"points": []})
    frame = _query(
        f"SELECT ts_ms, equity FROM equity ORDER BY ts_ms DESC LIMIT {int(limit)}"
    ).sort_values("ts_ms")
    return JSONResponse({"points": _records(frame)})


@app.get("/api/state", dependencies=[Depends(require_auth)])
def api_state() -> JSONResponse:
    exchange, extra = _load_account_and_extra()
    marks = {p.coin: p.entry_price for p in exchange.open_positions()}

    # "Nothing is happening" is a question the dashboard should answer, so
    # the activity rules and the running idle clock are shown rather than
    # left to be inferred from an empty fills table.
    idle_since = extra.get("idle_since_ms")
    idle_ms = max(0, _now_ms() - int(idle_since)) if idle_since else None
    max_idle_ms = SETTINGS.strategy.max_idle_ms

    # The moment the timer expires is not the moment anything happens: the
    # trader only decides on bar boundaries. Report the decision that will
    # act on it, or the number is off by up to a full interval -- which is
    # exactly how a 13:00 entry got described as 12:21.
    deadline = int(idle_since) + max_idle_ms if idle_since and max_idle_ms else None
    return JSONResponse({
        "timezone": DISPLAY_TZ,
        "markets": list(_configured_markets()),
        "max_position_usd": SETTINGS.risk.max_position_usd,
        "risk_per_trade": SETTINGS.risk.risk_per_trade,
        "signal_threshold": _signal_threshold(),
        "activity": {
            "min_hold_hours": SETTINGS.strategy.min_hold_hours,
            "max_hold_hours": SETTINGS.strategy.max_hold_hours,
            "max_idle_hours": SETTINGS.strategy.max_idle_hours,
            "idle_since_ms": idle_since,
            "idle_hours": round(idle_ms / 3_600_000, 2) if idle_ms is not None else None,
            "interval_ms": _interval_ms(),
            "next_decision_ms": _next_decision_ms(_now_ms()),
            "forced_entry_due_ms": deadline,
            "forced_entry_at_ms": (
                _next_decision_ms(deadline - 1) if deadline else None
            ),
        },
        "profile": SETTINGS.risk.name,
        "max_leverage": SETTINGS.risk.max_leverage,
        "max_positions": SETTINGS.risk.max_open_positions,
        "starting_capital": exchange.starting_capital,
        "cash": exchange.cash,
        "equity": exchange.equity(marks),
        "unrealized": exchange.unrealized_pnl(marks),
        "positions": [
            {
                "coin": p.coin, "side": p.direction, "size": p.size,
                "entry_price": p.entry_price, "notional": p.notional(p.entry_price),
                "unrealized": p.unrealized_pnl(p.entry_price),
                "funding_paid": p.funding_paid,
            }
            for p in exchange.open_positions()
        ],
        "closed_trades": len(exchange.closed_trades),
        "fills": len(exchange.fills),
        "fees": exchange.total_fees,
        "funding": exchange.total_funding,
        "slippage": exchange.total_slippage,
        "liquidations": exchange.liquidation_count,
        "bankrupt": exchange.bankrupt,
    })


# --------------------------------------------------------------------------
# What it bought and sold
# --------------------------------------------------------------------------

@app.get("/api/activity", dependencies=[Depends(require_auth)])
def api_activity(limit: int = 60) -> JSONResponse:
    """Every buy and sell, newest first, plus completed round trips."""
    exchange = _load_account()

    fills = [
        {
            "ts_ms": f.ts_ms, "coin": f.coin, "side": f.side.value,
            "size": f.size, "price": f.price, "notional": f.size * f.price,
            "fee": f.fee, "slippage": f.slippage_cost,
            "realized_pnl": f.realized_pnl, "is_liquidation": f.is_liquidation,
            "reason": f.context.reason,
            "probability": f.context.model_probability,
            "regime": f.context.regime,
        }
        for f in sorted(exchange.fills, key=lambda f: f.ts_ms, reverse=True)[:limit]
    ]
    trades = [
        {
            "coin": t.coin, "direction": t.direction,
            "opened_ts_ms": t.opened_ts_ms, "closed_ts_ms": t.closed_ts_ms,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "net_pnl": t.net_pnl, "fees": t.fees, "funding": t.funding,
            "liquidated": t.liquidated,
        }
        for t in sorted(exchange.closed_trades, key=lambda t: t.closed_ts_ms, reverse=True)[:limit]
    ]
    return JSONResponse({"fills": fills, "trades": trades})


# --------------------------------------------------------------------------
# The reasoning that goes into the 6-hour email
# --------------------------------------------------------------------------

@app.get("/api/reasoning", dependencies=[Depends(require_auth)])
def api_reasoning() -> JSONResponse:
    """The latest decision per market, with the same prose the email carries.

    Generated by the identical template the report uses, so what you read
    here is what arrives in the inbox -- not a second, divergent narrative.
    """
    if not _store().has_data("decisions"):
        return JSONResponse({"markets": [], "window_hours": 6})

    frame = _query(
        "SELECT * FROM decisions WHERE ts_ms >= ? ORDER BY ts_ms",
        [int(time.time() * 1000) - 6 * HOUR_MS],
    )
    if frame.empty:
        frame = _query("SELECT * FROM decisions ORDER BY ts_ms DESC LIMIT 30")

    markets = []
    for coin, group in frame.groupby("coin"):
        row = group.sort_values("ts_ms").iloc[-1]
        drivers = []
        for part in str(row.get("top_features", "") or "").split(";"):
            part = part.strip()
            if part:
                # Stored as "name + (value)"; the sign carries the direction.
                drivers.append((part, 1.0 if " + " in part else -1.0))

        plan = PlanLine(
            coin=str(coin),
            probability=float(row.get("probability", float("nan"))),
            confidence=float(row.get("confidence", 0.0)),
            direction=str(row.get("direction", "flat")),
            regime=str(row.get("regime", "unknown")),
            target_notional=float(row.get("target_notional", 0.0) or 0.0),
            current_notional=float(row.get("current_notional", 0.0) or 0.0),
            action=str(row.get("action", "")),
            risk_summary=str(row.get("risk_summary", "")),
            drivers=drivers,
        )
        markets.append({
            "coin": plan.coin,
            "ts_ms": int(row["ts_ms"]),
            "probability": plan.probability,
            "confidence": plan.confidence,
            "direction": plan.direction,
            "regime": plan.regime,
            "target_notional": plan.target_notional,
            "action": plan.action,
            "risk_summary": plan.risk_summary,
            "drivers": [name for name, _ in drivers],
            "reasoning": market_reasoning(plan),
        })
    return JSONResponse({"markets": sorted(markets, key=lambda m: m["coin"]),
                         "window_hours": 6})


@app.get("/api/decisions", dependencies=[Depends(require_auth)])
def api_decisions(limit: int = 40) -> JSONResponse:
    if not _store().has_data("decisions"):
        return JSONResponse({"decisions": []})
    return JSONResponse({"decisions": _records(
        _query(f"SELECT * FROM decisions ORDER BY ts_ms DESC LIMIT {int(limit)}")
    )})


# --------------------------------------------------------------------------
# What it has learned
# --------------------------------------------------------------------------

#: The live scorecard reads every stored bar to resolve every recorded call,
#: which is far too much work to repeat for each viewer every 30 seconds --
#: and pointless, because it can only change when a bar closes.
_SCORECARD_CACHE: dict[str, object] = {"ts_ms": 0, "value": None}
_SCORECARD_TTL_MS = 5 * 60_000


def _model_path() -> Path:
    """Where the live model actually is.

    Railway attaches one volume per service, so the artefact lives beside
    the market data under the mount rather than in the image's models/
    directory. MODEL_PATH is what start.sh passes the trader; fall back to
    both conventional locations so a local run finds it too.
    """
    configured = os.environ.get("MODEL_PATH")
    candidates = [Path(configured)] if configured else []
    candidates += [SETTINGS.paths.storage / "model.pkl", SETTINGS.paths.models / "model.pkl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _cached_scorecard(model) -> dict:
    now = _now_ms()
    cached = _SCORECARD_CACHE["value"]
    if cached is not None and now - int(_SCORECARD_CACHE["ts_ms"]) < _SCORECARD_TTL_MS:
        return cached  # type: ignore[return-value]

    from models.scorecard import load_scorecard

    try:
        card = load_scorecard(
            coins=_configured_markets(),
            interval=_trade_interval(),
            interval_ms=_interval_ms(),
            label_config=model.label_config if model else None,
            entry_threshold=_signal_threshold(),
            store=_store(),
        ).to_dict()
    except Exception as exc:  # noqa: BLE001 - an empty panel beats a 500
        log.warning("live scorecard failed: %s", exc)
        card = {}
    _SCORECARD_CACHE.update(ts_ms=now, value=card)
    return card


@app.get("/api/learning", dependencies=[Depends(require_auth)])
def api_learning() -> JSONResponse:
    """The model behind the decisions, how it has actually done, and when it
    next refits.

    Configuration the reader cannot see is configuration they cannot trust,
    and that goes double for a model that changes itself on a schedule: a
    shift in behaviour is otherwise indistinguishable from a shift in the
    market.
    """
    model = down_model = None
    try:
        from models.train import TrainedModel, down_model_path

        path = _model_path()
        if path.exists():
            model = TrainedModel.load(path)
        down_path = down_model_path(path)
        if down_path.exists():
            down_model = TrainedModel.load(down_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the live model: %s", exc)

    _, extra = _load_account_and_extra()
    every_ms = SETTINGS.learning.every_ms
    last_ms = extra.get("last_retrain_ms") or (model.trained_at_ms if model else None)

    history = _records(
        _query("SELECT * FROM retrains ORDER BY ts_ms DESC LIMIT 12")
    ) if _store().has_data("retrains") else []

    return JSONResponse({
        "model": None if model is None else {
            "backend": model.backend_name,
            "question": model.label_config.name,
            "features": len(model.features),
            "trained_at_ms": model.trained_at_ms,
            "trained_through_ms": model.train_span[1],
            "val_auc": _finite(model.mean_val_auc),
            "log_loss_lift": _finite(model.mean_log_loss_lift),
        },
        # Whether the system can act on a falling market at all, or can only
        # sit one out. That is not something a reader should have to infer
        # from an absence of short fills.
        "shorting": {
            "enabled": down_model is not None,
            "question": None if down_model is None else down_model.label_config.name,
            "val_auc": None if down_model is None else _finite(down_model.mean_val_auc),
            "trained_through_ms": None if down_model is None else down_model.train_span[1],
        },
        "retrain": {
            "enabled": every_ms is not None,
            "every_hours": SETTINGS.learning.every_hours,
            "describe": SETTINGS.learning.describe(),
            "last_ms": last_ms,
            "next_ms": (last_ms + every_ms) if (every_ms and last_ms) else None,
            "history": history,
        },
        "scorecard": _cached_scorecard(model),
    })


def _finite(value: float) -> float | None:
    """NaN is a perfectly ordinary metric here and is not JSON."""
    return float(value) if value == value and abs(value) != float("inf") else None


# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Unauthenticated liveness probe.

    Deliberately outside require_auth: platform health checks cannot present
    credentials, and pointing one at "/" means it reads the 401 as a failure
    and kills a container that is working perfectly. Returns no account data,
    so there is nothing here worth protecting.
    """
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Paper Trading</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#fff;--fg:#101828;--muted:#667085;--line:#e6e8eb;--panel:#fafbfc;
   --up:#0b7a3b;--down:#a3162a;--warn:#b54708}
 @media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;
   --line:#21262d;--panel:#161b22;--up:#3fb950;--down:#f85149;--warn:#d29922}}
 *{box-sizing:border-box}
 body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
   font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
 .wrap{max-width:1060px;margin:0 auto}
 h1{font-size:20px;margin:0 0 2px} h2{font-size:15px;margin:28px 0 10px}
 .muted{color:var(--muted);font-size:13px}
 .pnl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
 .card{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--panel)}
 .card .label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
 .card .value{font-size:23px;font-weight:700;margin-top:5px;font-variant-numeric:tabular-nums}
 .card .sub{font-size:12px;color:var(--muted);margin-top:2px}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th{text-align:left;color:var(--muted);font-weight:600;padding:8px;
   border-bottom:1px solid var(--line);white-space:nowrap}
 td{padding:8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
 .up{color:var(--up)} .down{color:var(--down)} .warn{color:var(--warn)}
 .scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
 .reason{border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:10px}
 .reason h3{margin:0 0 4px;font-size:15px}
 .chips{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}
 .chip{font-size:11px;border:1px solid var(--line);border-radius:20px;padding:2px 9px;color:var(--muted)}
 .banner{border-left:4px solid var(--warn);background:var(--panel);padding:10px 12px;
   border-radius:0 8px 8px 0;margin:14px 0;font-size:13px}
 footer{margin-top:34px;padding-top:14px;border-top:1px solid var(--line);
   color:var(--muted);font-size:12px}
 .side-buy{color:var(--up);font-weight:600} .side-sell{color:var(--down);font-weight:600}
</style></head><body><div class="wrap">
<h1>Hyperliquid Paper Trading</h1>
<p class="muted" id="sub">loading…</p>
<p class="muted" id="rules"></p>
<div id="banner"></div>

<div class="pnl" id="pnl"></div>

<h2>Equity</h2>
<div class="scroll"><svg id="chart" width="1000" height="200" role="img" aria-label="Equity curve"></svg></div>

<h2>Open positions</h2>
<div class="scroll"><table id="pos"><thead><tr><th>Market</th><th>Side</th><th>Size</th>
<th>Entry</th><th>Notional</th><th>Unrealised</th></tr></thead><tbody></tbody></table></div>

<h2>What it bought and sold</h2>
<div class="scroll"><table id="fills"><thead><tr><th>Time</th><th>Market</th><th>Side</th>
<th>Size</th><th>Price</th><th>Notional</th><th>Fee</th><th>Realised</th><th>Why</th>
</tr></thead><tbody></tbody></table></div>

<h2>Closed round trips</h2>
<div class="scroll"><table id="trades"><thead><tr><th>Market</th><th>Direction</th>
<th>Entry &rarr; Exit</th><th>Net P&amp;L</th><th>Fees</th><th>Funding</th>
</tr></thead><tbody></tbody></table></div>

<h2>Current reasoning <span class="muted" id="rwin"></span></h2>
<div id="reasoning"></div>

<h2>What it has learned</h2>
<p class="muted" id="lsub"></p>
<div class="pnl" id="lcards"></div>
<div class="scroll"><table id="retrains"><thead><tr><th>Time</th><th>Outcome</th>
<th>Rows</th><th>Candidate</th><th>Incumbent</th><th>Why</th>
</tr></thead><tbody></tbody></table></div>

<footer>Simulated paper trading. No real capital at risk. Not financial advice.<br>
<span id="updated"></span></footer>
</div><script>
const usd=n=>(n<0?'-$':'$')+Math.abs(Number(n)).toLocaleString(undefined,
  {minimumFractionDigits:2,maximumFractionDigits:2});
const sgn=n=>(n>=0?'+':'')+usd(n).replace('-','');
const cls=n=>n>0?'up':(n<0?'down':'');
// Every time on this page is shown in TZ (the server's DISPLAY_TIMEZONE).
// The trader is UTC throughout; this is presentation only.
let TZ='UTC';
const fmt=(ms,opts)=>new Date(ms).toLocaleString('en-GB',
  Object.assign({timeZone:TZ,hour12:false},opts));
const tm=ms=>fmt(ms,{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
const hm=ms=>fmt(ms,{hour:'2-digit',minute:'2-digit'});
const tzLabel=()=>{
  const parts=new Intl.DateTimeFormat('en-GB',
    {timeZone:TZ,timeZoneName:'short'}).formatToParts(new Date());
  const z=parts.find(p=>p.type==='timeZoneName');
  return z?z.value:TZ;
};

function pnlCard(label,value,sub){
  if(value===null||value===undefined)
    return `<div class="card"><div class="label">${label}</div>
      <div class="value muted" style="font-size:17px">not enough history</div>
      <div class="sub">${sub||''}</div></div>`;
  return `<div class="card"><div class="label">${label}</div>
    <div class="value ${cls(value)}">${sgn(value)}</div><div class="sub">${sub||''}</div></div>`;
}

// The model refits itself on a schedule, so a change in behaviour is
// otherwise indistinguishable from a change in the market. Show which model
// is deciding, how its own past calls actually turned out, and when it next
// refits.
function statCard(label,value,sub){
  return `<div class="card"><div class="label">${label}</div>
    <div class="value" style="font-size:20px">${value}</div>
    <div class="sub">${sub||''}</div></div>`;
}

function renderLearning(l){
  const m=l.model, rt=l.retrain||{}, sc=l.scorecard||{}, mx=sc.metrics||{};
  const sh=l.shorting||{};
  const bits=[];
  if(m){
    bits.push(`${m.backend} · ${m.features} features · asks ${m.question}`);
    bits.push(`fitted on data through ${tm(m.trained_through_ms)}`);
    if(m.val_auc!=null) bits.push(`walk-forward AUC ${m.val_auc.toFixed(4)}`);
  } else {
    bits.push('No model artefact could be read.');
  }
  bits.push(sh.enabled
    ? `shorting ON — ${sh.question}`
      + (sh.val_auc!=null?` (AUC ${sh.val_auc.toFixed(4)})`:'')
    : 'shorting OFF — long-only, it can only sit out a falling market');
  if(rt.enabled && rt.next_ms) bits.push(`next refit ${tm(rt.next_ms)} ${tzLabel()}`);
  else if(!rt.enabled) bits.push('retraining is OFF — this model is frozen');
  document.getElementById('lsub').textContent = bits.join(' · ');

  const cards=[];
  const resolved=sc.resolved||0;
  cards.push(statCard('Calls marked', resolved.toLocaleString(),
    (sc.pending||0).toLocaleString()+' still inside their horizon'));
  cards.push(statCard('Live AUC',
    mx.auc!=null?mx.auc.toFixed(4):'—',
    sc.has_verdict?'0.50 is a coin flip':'too few calls to mean anything yet'));
  const acted=sc.acted||{};
  cards.push(statCard('When it said go',
    acted.hit_rate!=null?(acted.hit_rate*100).toFixed(1)+'%':'—',
    acted.rows?`${acted.rows.toLocaleString()} calls above the entry bar · `
      +`base rate ${(mx.base_rate*100).toFixed(1)}%`
      :'no resolved call cleared the entry bar'));
  document.getElementById('lcards').innerHTML=cards.join('');

  const hist=rt.history||[];
  document.querySelector('#retrains tbody').innerHTML = hist.length
    ? hist.map(h=>`<tr><td>${tm(h.ts_ms)}</td>
      <td class="${h.status==='promoted'?'up':(h.status==='failed'?'down':'muted')}">${h.status}</td>
      <td>${Number(h.rows||0).toLocaleString()}</td>
      <td>${h.candidate_auc!=null?Number(h.candidate_auc).toFixed(4):'—'}</td>
      <td>${h.incumbent_auc!=null?Number(h.incumbent_auc).toFixed(4):'—'}</td>
      <td class="muted">${h.reason||''}</td></tr>`).join('')
    : '<tr><td colspan="6" class="muted">No refit has run yet.</td></tr>';
}

// One failing endpoint must never blank the page. Promise.all rejects as
// soon as any member does, so a single 500 on any one API left every panel
// empty while the page itself still returned 200 -- the same "one bad
// component stops the rest" failure this system keeps having, moved into
// the browser. Each fetch now resolves to a usable fallback instead of
// rejecting, and says so out loud rather than failing silently.
const FAILED=[];
async function get(url, fallback){
  try{
    const res=await fetch(url);
    if(!res.ok) throw new Error(url+' returned '+res.status);
    return await res.json();
  }catch(err){
    console.error(err);
    FAILED.push(url);
    return fallback;
  }
}

// Likewise for rendering: a panel that throws on unexpected data must not
// take the panels after it down with it.
function safely(name, fn){
  try{ fn(); }
  catch(err){ console.error(name, err); FAILED.push(name); }
}

async function load(){
  FAILED.length=0;
  const [s,p,e,a,r,l]=await Promise.all([
    get('/api/state',{positions:[],markets:[],activity:{},closed_trades:0,fills:0,
                      liquidations:0,fees:0,slippage:0,funding:0}),
    get('/api/pnl',{readings:0}),
    get('/api/equity',{points:[]}),
    get('/api/activity',{fills:[],trades:[]}),
    get('/api/reasoning',{markets:[]}),
    get('/api/learning',{})]);

  TZ = s.timezone || 'UTC';

  document.getElementById('sub').textContent =
    `${s.profile} profile · max ${s.max_leverage}x · ${s.max_positions} positions · `
    +`${s.closed_trades} round trips · ${s.fills} fills · ${s.liquidations} liquidation(s)`;

  // An empty table should never leave you guessing whether the thing is
  // broken or simply waiting. Say which, and say when it acts next.
  const act = s.activity || {};
  const rules = [];
  if (s.markets && s.markets.length) {
    rules.push(`${s.markets.length} markets`);
  }
  if (s.max_position_usd) rules.push(`max ${usd(s.max_position_usd)} per market`);
  if (s.signal_threshold) rules.push(`enters at P(up) ≥ ${s.signal_threshold}`);
  if (act.min_hold_hours && act.max_hold_hours) {
    rules.push(`holds ${act.min_hold_hours}–${act.max_hold_hours}h`);
  }
  if (act.next_decision_ms) {
    rules.push(`next decision ${hm(act.next_decision_ms)} ${tzLabel()}`);
  }
  if (act.max_idle_hours) {
    if (s.positions.length) {
      rules.push(`forces an entry after ${act.max_idle_hours}h holding nothing`);
    } else if (act.idle_hours != null) {
      const when = act.forced_entry_at_ms
        ? hm(act.forced_entry_at_ms) + ' ' + tzLabel()
        : null;
      rules.push(`flat for ${act.idle_hours.toFixed(1)}h — forces an entry`
                 + (when ? ` at the ${when} decision` : '')
                 + ` unless a signal clears first`);
    }
  }
  document.getElementById('rules').textContent = rules.join(' · ');

  const notes=[];
  if(s.bankrupt) notes.push('<b>The account has been wiped out.</b> Trading has stopped.');
  if(p.readings<2) notes.push('Only just started — hourly and daily P&L need more readings.');
  if(FAILED.length) notes.push('<b>Some panels could not load:</b> '+FAILED.join(', ')
    +'. Everything else on this page is current. Check the service logs.');
  document.getElementById('banner').innerHTML =
    notes.map(n=>`<div class="banner">${n}</div>`).join('');

  document.getElementById('pnl').innerHTML=[
    `<div class="card"><div class="label">Equity</div>
      <div class="value">${usd(p.equity)}</div>
      <div class="sub">from ${usd(p.starting_capital)}</div></div>`,
    pnlCard('P&L — lifetime', p.lifetime, (p.lifetime_pct*100).toFixed(2)+'%'),
    pnlCard('P&L — today', p.day, 'since 00:00 '+tzLabel()),
    pnlCard('P&L — last hour', p.hour, 'rolling 60 min'),
    `<div class="card"><div class="label">Costs paid</div>
      <div class="value" style="font-size:18px">${usd(s.fees+s.slippage+s.funding)}</div>
      <div class="sub">fees ${usd(s.fees)} · slip ${usd(s.slippage)} · fund ${usd(s.funding)}</div></div>`,
  ].join('');

  const pts=e.points||[], svg=document.getElementById('chart');
  if(pts.length>1){
    const w=1000,h=200,pad=22, xs=pts.map(o=>o.ts_ms), ys=pts.map(o=>o.equity);
    const x0=Math.min(...xs),x1=Math.max(...xs);
    let y0=Math.min(...ys,p.starting_capital),y1=Math.max(...ys,p.starting_capital);
    if(y1-y0<1e-9){y0-=1;y1+=1;}
    const sx=v=>pad+(v-x0)/((x1-x0)||1)*(w-2*pad), sy=v=>h-pad-(v-y0)/((y1-y0)||1)*(h-2*pad);
    const d=pts.map((o,i)=>(i?'L':'M')+sx(o.ts_ms).toFixed(1)+' '+sy(o.equity).toFixed(1)).join(' ');
    const col=ys[ys.length-1]>=p.starting_capital?'var(--up)':'var(--down)';
    const base=sy(p.starting_capital).toFixed(1);
    svg.innerHTML=`<line x1="${pad}" y1="${base}" x2="${w-pad}" y2="${base}"
        stroke="var(--muted)" stroke-dasharray="3 3" stroke-width="1"/>
      <path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;
  } else {
    svg.innerHTML='<text x="12" y="28" fill="currentColor" font-size="13" opacity=".6">'
      +'Not enough history yet — one point is recorded per decision cycle.</text>';
  }

  document.querySelector('#pos tbody').innerHTML = s.positions.length
    ? s.positions.map(x=>`<tr><td><b>${x.coin}</b></td><td>${x.side}</td>
      <td>${x.size.toFixed(5)}</td><td>${x.entry_price.toFixed(2)}</td>
      <td>${usd(x.notional)}</td><td class="${cls(x.unrealized)}">${sgn(x.unrealized)}</td></tr>`).join('')
    : '<tr><td colspan="6" class="muted">No open positions.</td></tr>';

  document.querySelector('#fills tbody').innerHTML = (a.fills||[]).length
    ? a.fills.map(f=>`<tr><td>${tm(f.ts_ms)}</td><td><b>${f.coin}</b></td>
      <td class="side-${f.side}">${f.side.toUpperCase()}${f.is_liquidation?' ⚠':''}</td>
      <td>${f.size.toFixed(5)}</td><td>${f.price.toFixed(2)}</td><td>${usd(f.notional)}</td>
      <td>${usd(f.fee)}</td><td class="${cls(f.realized_pnl)}">${f.realized_pnl?sgn(f.realized_pnl):'—'}</td>
      <td class="muted">${f.reason||''}</td></tr>`).join('')
    : '<tr><td colspan="9" class="muted">Nothing bought or sold yet.</td></tr>';

  document.querySelector('#trades tbody').innerHTML = (a.trades||[]).length
    ? a.trades.map(t=>`<tr><td><b>${t.coin}</b>${t.liquidated?' <span class="down">LIQ</span>':''}</td>
      <td>${t.direction}</td><td>${t.entry_price.toFixed(2)} → ${t.exit_price.toFixed(2)}</td>
      <td class="${cls(t.net_pnl)}">${sgn(t.net_pnl)}</td><td>${usd(t.fees)}</td>
      <td>${sgn(-t.funding)}</td></tr>`).join('')
    : '<tr><td colspan="6" class="muted">No completed round trips yet.</td></tr>';

  document.getElementById('rwin').textContent =
    `— same text the ${r.window_hours}-hour email carries`;
  document.getElementById('reasoning').innerHTML = (r.markets||[]).length
    ? r.markets.map(m=>`<div class="reason"><h3>${m.coin}
        <span class="muted" style="font-weight:400">— leaning ${m.direction}</span></h3>
      <div class="muted">P(up) <b>${Number(m.probability).toFixed(3)}</b> ·
        confidence <b>${Number(m.confidence).toFixed(2)}</b> ·
        regime <b>${m.regime}</b> · target <b>${usd(m.target_notional)}</b></div>
      <p style="margin:8px 0 0">${m.reasoning}</p>
      <div class="chips">${(m.drivers||[]).map(d=>`<span class="chip">${d}</span>`).join('')}</div>
      <div class="muted" style="margin-top:8px;font-size:12px">Risk: ${m.risk_summary||'—'}</div>
      </div>`).join('')
    : '<p class="muted">No decisions recorded yet. The trader writes one per bar close.</p>';

  safely('learning panel', ()=>renderLearning(l));

  document.getElementById('updated').textContent =
    'Updated '+hm(Date.now())+' '+tzLabel()+' · refreshes every 30s';
}
load(); setInterval(load, 30000);
</script></body></html>"""
