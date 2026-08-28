"""Durable state for the live paper trader.

The process will be restarted -- for a deploy, a reboot, a crash. Positions,
cash and the trade ledger have to survive that, or every restart silently
resets the account to $100k and the track record is fiction.

State is written atomically (temp file, then rename) after every decision
cycle, so a kill at any moment leaves either the old state or the new one,
never a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from execution.paper_exchange import (
    ClosedTrade,
    DecisionContext,
    Fill,
    PaperExchange,
    Position,
)
from execution.simulator import Side

log = logging.getLogger(__name__)

STATE_VERSION = 1


def _context_to_dict(context: DecisionContext) -> dict[str, Any]:
    return asdict(context)


def _context_from_dict(data: dict[str, Any] | None) -> DecisionContext:
    if not data:
        return DecisionContext()
    return DecisionContext(**data)


class StateStore:
    """Reads and writes the paper account's state as JSON."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    # -- writing -----------------------------------------------------------

    def save(self, exchange: PaperExchange, *, extra: dict[str, Any] | None = None) -> Path:
        payload = {
            "version": STATE_VERSION,
            "saved_at_ms": int(time.time() * 1000),
            "starting_capital": exchange.starting_capital,
            "cash": exchange.cash,
            "total_fees": exchange.total_fees,
            "total_funding": exchange.total_funding,
            "total_slippage": exchange.total_slippage,
            "liquidation_count": exchange.liquidation_count,
            "bankrupt": exchange.bankrupt,
            "last_funding_hour": exchange._last_funding_hour,
            "positions": [
                {**asdict(p), "open_context": _context_to_dict(p.open_context)}
                for p in exchange.positions.values()
            ],
            "closed_trades": [
                {
                    **{k: v for k, v in asdict(t).items()
                       if k not in {"open_context", "close_context"}},
                    "open_context": _context_to_dict(t.open_context),
                    "close_context": _context_to_dict(t.close_context),
                }
                for t in exchange.closed_trades
            ],
            "fills": [
                {
                    "ts_ms": f.ts_ms, "coin": f.coin, "side": f.side.value,
                    "size": f.size, "price": f.price, "fee": f.fee,
                    "slippage_cost": f.slippage_cost, "realized_pnl": f.realized_pnl,
                    "is_maker": f.is_maker, "is_liquidation": f.is_liquidation,
                    "context": _context_to_dict(f.context),
                }
                for f in exchange.fills
            ],
            "extra": extra or {},
        }

        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, self.path)
        return self.path

    # -- reading -----------------------------------------------------------

    def load_into(self, exchange: PaperExchange) -> dict[str, Any]:
        """Restore a saved account into ``exchange``. Returns the ``extra`` blob."""
        if not self.exists():
            return {}
        with open(self.path) as handle:
            payload = json.load(handle)

        if payload.get("version") != STATE_VERSION:
            raise ValueError(
                f"state file version {payload.get('version')} does not match "
                f"{STATE_VERSION}; refusing to guess at its meaning"
            )
        if payload["starting_capital"] != exchange.starting_capital:
            log.warning(
                "saved state started from $%,.0f but this run is configured for $%,.0f",
                payload["starting_capital"], exchange.starting_capital,
            )

        exchange.cash = float(payload["cash"])
        exchange.total_fees = float(payload["total_fees"])
        exchange.total_funding = float(payload["total_funding"])
        exchange.total_slippage = float(payload["total_slippage"])
        exchange.liquidation_count = int(payload["liquidation_count"])
        exchange.bankrupt = bool(payload["bankrupt"])
        exchange._last_funding_hour = payload.get("last_funding_hour")

        exchange.positions = {}
        for record in payload["positions"]:
            context = _context_from_dict(record.pop("open_context", None))
            position = Position(**record)
            position.open_context = context
            exchange.positions[position.coin] = position

        exchange.closed_trades = [
            ClosedTrade(
                **{k: v for k, v in record.items()
                   if k not in {"open_context", "close_context"}},
                open_context=_context_from_dict(record.get("open_context")),
                close_context=_context_from_dict(record.get("close_context")),
            )
            for record in payload["closed_trades"]
        ]
        exchange.fills = [
            Fill(
                ts_ms=record["ts_ms"], coin=record["coin"], side=Side(record["side"]),
                size=record["size"], price=record["price"], fee=record["fee"],
                slippage_cost=record["slippage_cost"], realized_pnl=record["realized_pnl"],
                is_maker=record["is_maker"], is_liquidation=record["is_liquidation"],
                context=_context_from_dict(record.get("context")),
            )
            for record in payload["fills"]
        ]

        log.info(
            "restored paper account: cash $%.2f, %d open position(s), %d closed trade(s)",
            exchange.cash, len(exchange.open_positions()), len(exchange.closed_trades),
        )
        return payload.get("extra", {})
