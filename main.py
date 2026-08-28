#!/usr/bin/env python3
"""Command-line entry point.

    python main.py status                 # API reachability + safety checks
    python main.py backfill --days 180    # historical candles + funding
    python main.py collect                # live trades / book / funding
    python main.py verify                 # gap + future-timestamp audit
    python main.py summary                # what is on disk
    python main.py prove-accounting       # paper exchange arithmetic proof
    python main.py train                  # walk-forward model training
    python main.py backtest               # full system backtest + honest report
    python main.py paper                  # LIVE paper trading + 6-hour reports
    python main.py report --test-report   # send one report immediately
    python main.py dashboard              # local read-only web view

No subcommand here places a real order, and none ever will.

Original Phase 1 surface:

    python main.py status                 # API reachability + safety checks
    python main.py backfill --days 30     # historical candles + funding
    python main.py collect --minutes 10   # live trades / book / funding
    python main.py verify                 # gap + future-timestamp audit
    python main.py summary                # what is on disk
    python main.py prove-accounting       # paper exchange arithmetic proof

Later phases add train / backtest / paper / report subcommands. This file
must never gain a subcommand that places a real order.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from config.settings import SETTINGS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else getattr(logging, SETTINGS.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    from data.hyperliquid_client import HyperliquidInfoClient, TLSInterceptionError

    print("Hyperliquid paper-trading research system - status\n")
    print(f"  paper trading only : {SETTINGS.paper_trading_only}")
    print(f"  markets            : {', '.join(SETTINGS.data.markets)}")
    print(f"  intervals          : {', '.join(SETTINGS.data.candle_intervals)}")
    print(f"  info endpoint      : {SETTINGS.hyperliquid.info_url}")
    print(f"  ws endpoint        : {SETTINGS.hyperliquid.ws_url}")
    print(f"  TLS verify         : {SETTINGS.hyperliquid.ca_bundle or 'system trust store'}")
    print(f"  parquet root       : {SETTINGS.paths.parquet}")
    print("  signing keys       : none (verified at import)\n")

    try:
        with HyperliquidInfoClient() as client:
            meta = client.meta()
            mids = client.all_mids()
    except TLSInterceptionError as exc:
        print(f"  API reachability   : FAILED\n\n{exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"  API reachability   : FAILED ({type(exc).__name__}: {exc})")
        return 2

    universe = [a["name"] for a in meta.get("universe", [])]
    print(f"  API reachability   : OK ({len(universe)} perp markets listed)")
    missing = [m for m in SETTINGS.data.markets if m not in universe]
    if missing:
        print(f"  WARNING            : configured markets not listed: {missing}")
    for coin in SETTINGS.data.markets:
        if coin in mids:
            print(f"    {coin:<5} mid = {float(mids[coin]):,.2f}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from data.downloader import HistoricalDownloader

    coins = args.coins or list(SETTINGS.data.markets)
    intervals = args.intervals or list(SETTINGS.data.candle_intervals)
    downloader = HistoricalDownloader()
    print(f"Backfilling {coins} {intervals} ({args.days or SETTINGS.data.backfill_days} days)\n")
    try:
        results = downloader.backfill_all(
            coins, intervals, days=args.days, resume=not args.no_resume
        )
    finally:
        downloader.close()
    for result in results:
        print(result.describe())
    print(f"\nTotal new rows: {sum(r.rows_new for r in results):,}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from data.collector import LiveCollector

    duration = args.minutes * 60 if args.minutes else None
    collector = LiveCollector()
    print(
        f"Collecting live data for {SETTINGS.data.markets} "
        f"({'until interrupted' if duration is None else f'{args.minutes} minute(s)'})\n"
    )
    stats = asyncio.run(collector.run(duration_s=duration))
    print(f"\nCollector finished | {stats.describe()}")
    return 0 if stats.errors == 0 else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from data.database import MarketDatabase
    from data.quality import check_candles, check_funding

    tolerance_ms = int(SETTINGS.data.future_timestamp_tolerance_s * 1000)
    coins = args.coins or list(SETTINGS.data.markets)
    intervals = args.intervals or list(SETTINGS.data.candle_intervals)

    failures = 0
    with MarketDatabase() as db:
        if not db.store.has_data("candles"):
            print("No candle data on disk yet - run `python main.py backfill` first.")
            return 1
        print("Candle integrity\n")
        for coin in coins:
            for interval in intervals:
                df = db.query(
                    "SELECT * FROM candles WHERE coin = ? AND interval = ? ORDER BY ts_ms",
                    [coin, interval],
                )
                report = check_candles(
                    df, coin=coin, interval=interval, tolerance_ms=tolerance_ms
                )
                print(report.describe())
                failures += 0 if report.ok else 1

        if db.store.has_data("funding"):
            print("\nFunding integrity\n")
            for coin in coins:
                df = db.query(
                    "SELECT * FROM funding WHERE coin = ? ORDER BY ts_ms", [coin]
                )
                report = check_funding(df, coin=coin, tolerance_ms=tolerance_ms)
                print(report.describe())
                failures += 0 if report.ok else 1

    print(f"\n{'All checks passed.' if failures == 0 else f'{failures} slice(s) reported issues.'}")
    return 0 if failures == 0 else 1


def cmd_prove_accounting(args: argparse.Namespace) -> int:
    from backtest.proof import render_proof, run_proof

    print(render_proof())
    _, ok = run_proof()
    return 0 if ok else 1


def cmd_summary(args: argparse.Namespace) -> int:
    from data.database import MarketDatabase

    with MarketDatabase() as db:
        summary = db.table_summary()
        print("Stored datasets\n")
        print(summary.to_string(index=False))

        if db.store.has_data("candles"):
            print("\nCandles by coin / interval\n")
            print(
                db.query(
                    "SELECT coin, interval, count(*) AS bars, min(ts) AS first_bar, "
                    "max(ts) AS last_bar FROM candles GROUP BY 1, 2 ORDER BY 1, 2"
                ).to_string(index=False)
            )
    return 0



# --------------------------------------------------------------------------
# Phases 4-8
# --------------------------------------------------------------------------

def _load_market_data(args):
    from data.loader import load_bars, load_funding, load_order_books

    coins = args.coins or list(SETTINGS.data.markets)
    interval = getattr(args, "interval", "1h")
    bars = load_bars(coins, interval)
    return bars, load_funding(list(bars)), load_order_books(list(bars)), interval


def cmd_train(args: argparse.Namespace) -> int:
    from features.pipeline import FeatureConfig, build_universe
    from models.backend import ModelParams
    from models.dataset import SplitConfig, assemble
    from models.labels import LabelConfig
    from models.train import evaluate_holdout, render_report, train_walk_forward

    bars, funding, books, interval = _load_market_data(args)
    print(f"Building features for {list(bars)} at {interval}...")
    matrices = build_universe(
        bars, funding_by_coin=funding, book_by_coin=books,
        config=FeatureConfig(interval=interval),
    )

    label_config = LabelConfig(horizon_bars=args.horizon, threshold=args.threshold)
    dataset = assemble(matrices, label_config)
    print(f"Training on {len(dataset):,} rows, asking: {label_config.name}\n")

    model, holdout = train_walk_forward(
        dataset,
        label_config=label_config,
        split_config=SplitConfig(n_folds=args.folds, test_fraction=args.test_fraction),
        params=ModelParams(),
    )
    if not args.skip_holdout:
        # Called once, after the model is frozen. Anything tuned from here on
        # makes this number in-sample.
        evaluate_holdout(model, holdout)

    print(render_report(model))
    path = model.save(args.output or SETTINGS.paths.models / "model.pkl")
    print(f"\nModel written to {path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtest.runner import run_backtest
    from config.settings import resolve_risk_profile
    from models.train import TrainedModel

    model_path = args.model or SETTINGS.paths.models / "model.pkl"
    model = TrainedModel.load(model_path)
    bars, funding, books, interval = _load_market_data(args)

    limits = resolve_risk_profile(args.profile) if args.profile else SETTINGS.risk
    print(f"Backtesting {list(bars)} at {interval} on the '{limits.name}' profile...\n")

    report = run_backtest(
        bars, model,
        funding_by_coin=funding, book_by_coin=books,
        limits=limits, interval=interval,
        long_threshold=args.threshold,
        out_of_sample_from_ms=model.train_span[1],
    )
    print(report.render())

    if args.control:
        from backtest.runner import run_shuffled_control

        print("\n" + "=" * 78)
        print("SHUFFLED CONTROL - the same strategy with the model's predictions")
        print("randomly permuted. The real run must clearly beat this to mean anything.\n")
        control = run_shuffled_control(
            bars, model,
            funding_by_coin=funding, book_by_coin=books,
            limits=limits, interval=interval, long_threshold=args.threshold,
        )
        print(control.render())

        real, ctrl = report.metrics, control.metrics
        print("\n  Verdict:")
        if real.total_return <= ctrl.total_return:
            print("    The model did NOT beat a random signal. No edge has been shown.")
        else:
            print(f"    Model {real.total_return:+.2%} vs control {ctrl.total_return:+.2%} "
                  f"(difference {real.total_return - ctrl.total_return:+.2%}).")
            print("    A positive difference is necessary but not sufficient: check that")
            print("    it survives out-of-sample and is not just lower turnover.")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    import asyncio

    from execution.paper_trader import PaperTrader
    from reporting.scheduler import ReportService, build_scheduler

    trader = PaperTrader(
        model_path=args.model, interval=args.interval, long_threshold=args.threshold
    )
    service = ReportService(trader)

    print("PAPER TRADING - simulated fills against live Hyperliquid data.")
    print(f"  risk profile : {SETTINGS.risk.describe()}")
    print(f"  markets      : {', '.join(trader.coins)}")
    print(f"  reports      : {'on' if SETTINGS.report.enabled else 'OFF'} "
          f"-> {SETTINGS.report.recipient} at {SETTINGS.report.schedule_hours}")
    if not service.emailer.configured:
        print(f"  WARNING      : email not configured "
              f"(missing {', '.join(service.emailer.missing_settings())}); "
              f"reports will be written to {service.emailer.fallback_dir} instead")
    print("  No real orders can be placed.\n")

    async def main_loop():
        scheduler = None
        if SETTINGS.report.enabled:
            scheduler = build_scheduler(service)
            scheduler.start()
        try:
            await trader.run(max_cycles=args.cycles)
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    asyncio.run(main_loop())
    print(f"\nStopped after {trader.cycles} cycle(s), {trader.errors} error(s).")
    return 0 if trader.errors == 0 else 1


def cmd_report(args: argparse.Namespace) -> int:
    from execution.paper_trader import PaperTrader
    from reporting.scheduler import ReportService

    trader = PaperTrader(model_path=args.model, interval=args.interval)
    service = ReportService(trader)

    outcome = service.send_now() if not args.dry_run else service.generate()
    if args.print_text:
        print(outcome.text)
    print(f"\nSubject: {outcome.subject}")
    if args.dry_run:
        print("Dry run: nothing was sent.")
        return 0
    print(outcome.send.describe())
    if not outcome.send.sent and not service.emailer.configured:
        print(f"Configure SMTP in .env: {', '.join(service.emailer.missing_settings())}")
    return 0 if outcome.send.sent else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Dashboard on http://{args.host}:{args.port} (read-only, no auth - keep it local)")
    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperliquid-quant",
        description="Paper-trading quant research system. Simulated fills only; no real orders, ever.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="check config and API reachability").set_defaults(func=cmd_status)

    backfill = sub.add_parser("backfill", help="download historical candles and funding")
    backfill.add_argument("--days", type=int, default=None)
    backfill.add_argument("--coins", nargs="*", default=None)
    backfill.add_argument("--intervals", nargs="*", default=None)
    backfill.add_argument("--no-resume", action="store_true", help="ignore what is on disk")
    backfill.set_defaults(func=cmd_backfill)

    collect = sub.add_parser("collect", help="run the live collector")
    collect.add_argument("--minutes", type=float, default=None, help="stop after N minutes")
    collect.set_defaults(func=cmd_collect)

    verify = sub.add_parser("verify", help="audit stored data for gaps and future timestamps")
    verify.add_argument("--coins", nargs="*", default=None)
    verify.add_argument("--intervals", nargs="*", default=None)
    verify.set_defaults(func=cmd_verify)

    sub.add_parser(
        "prove-accounting", help="verify the paper exchange's arithmetic on dumb strategies"
    ).set_defaults(func=cmd_prove_accounting)

    sub.add_parser("summary", help="show what is stored on disk").set_defaults(func=cmd_summary)

    train = sub.add_parser("train", help="train the signal model with walk-forward validation")
    train.add_argument("--coins", nargs="*", default=None)
    train.add_argument("--interval", default="1h")
    train.add_argument("--horizon", type=int, default=4, help="label horizon in bars")
    train.add_argument("--threshold", type=float, default=0.003, help="label return threshold")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--skip-holdout", action="store_true",
                       help="do not score the locked test set (keeps it unused)")
    train.add_argument("--output", default=None)
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest", help="run the full system over history")
    backtest.add_argument("--coins", nargs="*", default=None)
    backtest.add_argument("--interval", default="1h")
    backtest.add_argument("--model", default=None)
    backtest.add_argument("--profile", default=None, choices=["conservative", "aggressive"])
    backtest.add_argument("--threshold", type=float, default=0.55,
                          help="probability above which the strategy goes long")
    backtest.add_argument("--control", action="store_true",
                          help="also run a shuffled-prediction control for comparison")
    backtest.set_defaults(func=cmd_backtest)

    paper = sub.add_parser("paper", help="run live paper trading (simulated fills, real data)")
    paper.add_argument("--interval", default="1h")
    paper.add_argument("--model", default=None)
    paper.add_argument("--threshold", type=float, default=0.55)
    paper.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    paper.set_defaults(func=cmd_paper)

    report = sub.add_parser("report", help="build and send one report now")
    report.add_argument("--test-report", dest="test_report", action="store_true",
                        help="send a report immediately to verify formatting")
    report.add_argument("--dry-run", action="store_true", help="build it but do not send")
    report.add_argument("--print-text", action="store_true", help="print the plain-text version")
    report.add_argument("--interval", default="1h")
    report.add_argument("--model", default=None)
    report.set_defaults(func=cmd_report)

    dashboard = sub.add_parser("dashboard", help="serve the read-only dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8000)
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
