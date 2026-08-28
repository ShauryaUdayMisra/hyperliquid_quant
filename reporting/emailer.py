"""Sending the 6-hourly report.

Two rules from the brief, both enforced here:

* **Credentials live in ``.env``, never in code.** ``EmailConfig`` reads
  them at construction; nothing is hardcoded and nothing is logged.
* **A failed send must never stop the trading loop.** Every send is
  retried with exponential backoff, and a final failure is logged and
  returned as ``False`` rather than raised.

The report is sent as multipart/alternative: a plain-text part for clients
that cannot render HTML, and the HTML part for those that can.
"""

from __future__ import annotations

import logging
import random
import smtplib
import ssl
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path

import pandas as pd

from config.settings import SETTINGS, ReportConfig

log = logging.getLogger(__name__)


class EmailNotConfigured(RuntimeError):
    """Raised when a send is attempted without the required settings."""


@dataclass
class SendResult:
    sent: bool
    attempts: int
    error: str | None = None
    saved_to: Path | None = None

    def describe(self) -> str:
        if self.sent:
            return f"report sent after {self.attempts} attempt(s)"
        detail = f": {self.error}" if self.error else ""
        saved = f" (saved locally to {self.saved_to})" if self.saved_to else ""
        return f"report NOT sent after {self.attempts} attempt(s){detail}{saved}"


class Emailer:
    def __init__(
        self,
        config: ReportConfig | None = None,
        *,
        max_attempts: int = 4,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        fallback_dir: Path | None = None,
    ) -> None:
        self.config = config or SETTINGS.report
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        #: A report that cannot be emailed is still written to disk, so the
        #: record is never lost just because SMTP was down.
        self.fallback_dir = fallback_dir or (SETTINGS.paths.logs / "reports")

    # -- configuration -----------------------------------------------------

    def missing_settings(self) -> list[str]:
        missing = []
        if not self.config.smtp_host:
            missing.append("SMTP_HOST")
        if not self.config.recipient:
            missing.append("REPORT_RECIPIENT")
        if not (self.config.sender or self.config.smtp_user):
            missing.append("REPORT_SENDER (or SMTP_USER)")
        if not self.config.smtp_app_password:
            missing.append("SMTP_APP_PASSWORD")
        return missing

    @property
    def configured(self) -> bool:
        return not self.missing_settings()

    @property
    def sender_address(self) -> str:
        return self.config.sender or self.config.smtp_user

    # -- message -----------------------------------------------------------

    def build_message(self, subject: str, text_body: str, html_body: str) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = formataddr(("Hyperliquid Paper Trader", self.sender_address))
        message["To"] = self.config.recipient
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid(domain="hyperliquid-quant.local")
        # Plain text first: multipart/alternative means "last part wins where
        # it can be rendered", so HTML must come second.
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")
        return message

    # -- sending -----------------------------------------------------------

    def send(self, subject: str, text_body: str, html_body: str) -> SendResult:
        saved = self._save_locally(subject, html_body)

        missing = self.missing_settings()
        if missing:
            error = f"email not configured; missing {', '.join(missing)} in .env"
            log.warning("%s - report saved to %s instead", error, saved)
            return SendResult(False, 0, error, saved)

        message = self.build_message(subject, text_body, html_body)
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._deliver(message)
            except Exception as exc:  # noqa: BLE001 - email must never kill the loop
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("email attempt %d/%d failed: %s", attempt, self.max_attempts, last_error)
                if attempt < self.max_attempts:
                    delay = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_max_s)
                    time.sleep(random.uniform(delay / 2, delay))
            else:
                log.info("report emailed to %s", self.config.recipient)
                return SendResult(True, attempt, None, saved)

        log.error("giving up on emailing the report: %s", last_error)
        return SendResult(False, self.max_attempts, last_error, saved)

    def _deliver(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        if self.config.smtp_port == 465:
            with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port,
                                  context=context, timeout=30) as server:
                server.login(self.config.smtp_user or self.sender_address,
                             self.config.smtp_app_password)
                server.send_message(message)
            return

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(self.config.smtp_user or self.sender_address,
                         self.config.smtp_app_password)
            server.send_message(message)

    def _save_locally(self, subject: str, html_body: str) -> Path:
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
        path = self.fallback_dir / f"report-{stamp}.html"
        path.write_text(html_body)
        return path


def subject_line(equity: float, starting_capital: float, pnl_window: float) -> str:
    total_return = equity / starting_capital - 1.0 if starting_capital else 0.0
    arrow = "+" if pnl_window >= 0 else "-"
    return (
        f"[Paper] ${equity:,.0f} ({total_return:+.2%} all time) | "
        f"{arrow}${abs(pnl_window):,.0f} last 6h"
    )
