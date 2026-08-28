"""Wires report generation to a schedule.

APScheduler fires a cron trigger at 00:00, 06:00, 12:00 and 18:00 local
time. Because a scheduler only fires while the process is alive, a
system-level fallback (cron or launchd) is documented in
``deploy/DEPLOY.md`` and can invoke ``python main.py report`` independently.

A report failure is contained here: the trading loop keeps running whatever
happens to email.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config.settings import SETTINGS, Settings
from execution.paper_trader import PaperTrader
from reporting.emailer import Emailer, SendResult, subject_line
from reporting.report_builder import ReportBuilder, render_html, render_text

log = logging.getLogger(__name__)


@dataclass
class ReportOutcome:
    subject: str
    text: str
    html: str
    send: SendResult


class ReportService:
    """Builds a report from the live trader's state and sends it."""

    def __init__(
        self,
        trader: PaperTrader,
        *,
        settings: Settings | None = None,
        emailer: Emailer | None = None,
        builder: ReportBuilder | None = None,
    ) -> None:
        self.trader = trader
        self.settings = settings or SETTINGS
        self.emailer = emailer or Emailer(self.settings.report)
        self.builder = builder or ReportBuilder(settings=self.settings)

    def current_marks(self) -> dict[str, float]:
        """Latest mark per market, falling back to the last stored close."""
        marks: dict[str, float] = {}
        try:
            mids = self.trader.client.all_mids()
            for coin in self.trader.coins:
                if coin in mids:
                    marks[coin] = float(mids[coin])
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch live mids for the report: %s", exc)

        for position in self.trader.exchange.open_positions():
            marks.setdefault(position.coin, position.entry_price)
        return marks

    def generate(self, *, now_ms: int | None = None) -> ReportOutcome:
        marks = self.current_marks()
        data = self.builder.build(
            exchange=self.trader.exchange,
            marks=marks,
            risk_engine=self.trader.risk,
            latest_decisions=self.trader.strategy.latest_by_coin(),
            now_ms=now_ms or int(time.time() * 1000),
        )
        subject = subject_line(data.equity, data.starting_capital, data.pnl_window)
        return ReportOutcome(
            subject=subject,
            text=render_text(data),
            html=render_html(data),
            send=SendResult(False, 0),
        )

    def send_now(self, *, now_ms: int | None = None) -> ReportOutcome:
        outcome = self.generate(now_ms=now_ms)
        outcome.send = self.emailer.send(outcome.subject, outcome.text, outcome.html)
        log.info("6-hour report: %s", outcome.send.describe())
        return outcome

    def safe_send(self) -> ReportOutcome | None:
        """Never raise. A broken report must not take the trader down."""
        try:
            return self.send_now()
        except Exception:  # noqa: BLE001
            log.exception("report generation failed; trading continues")
            return None


def build_scheduler(service: ReportService, *, hours: tuple[int, ...] | None = None):
    """An APScheduler configured for the 6-hourly cadence."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    hours = hours or SETTINGS.report.schedule_hours
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        service.safe_send,
        CronTrigger(hour=",".join(str(h) for h in hours), minute=0),
        id="six_hour_report",
        name="6-hour paper trading report",
        # If the machine slept through a fire time, send one catch-up report
        # rather than a burst of stale ones.
        coalesce=True,
        misfire_grace_time=1800,
        max_instances=1,
    )
    log.info("report scheduler armed for %s (local time)", hours)
    return scheduler
