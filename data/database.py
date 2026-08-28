"""Storage layer: Parquet on disk, DuckDB as the query engine.

Layout (all files also carry their partition columns, so DuckDB reads them
with plain globs and no hive-partition inference):

    storage/parquet/
      candles/coin=BTC/interval=1m/2026-08-27.parquet
      funding/coin=BTC/2026-08.parquet
      asset_ctx/coin=BTC/2026-08-27.parquet
      trades/coin=BTC/2026-08-27/part-<ms>-<n>.parquet
      orderbook/coin=BTC/2026-08-27/part-<ms>-<n>.parquet

Two write modes:

``upsert`` -- for data the exchange will hand us again identically (candles,
funding). The day's file is read, merged on a key, de-duplicated and
rewritten. Re-running a backfill is therefore idempotent.

``append`` -- for streaming data that is never re-served (trades, order-book
snapshots). Each flush writes a new part file; de-duplication happens at
read time.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import SETTINGS, Paths

log = logging.getLogger(__name__)

#: dataset -> (partition columns, primary key columns, day-partition granularity)
DATASET_SPECS: dict[str, dict[str, object]] = {
    "candles": {"partition": ("coin", "interval"), "key": ("coin", "interval", "ts_ms"), "grain": "day"},
    "funding": {"partition": ("coin",), "key": ("coin", "ts_ms"), "grain": "month"},
    "asset_ctx": {"partition": ("coin",), "key": ("coin", "ts_ms"), "grain": "day"},
    "trades": {"partition": ("coin",), "key": ("coin", "tid"), "grain": "day"},
    "orderbook": {"partition": ("coin",), "key": ("coin", "ts_ms", "side", "level"), "grain": "day"},
    # --- Phase 6: the paper-trading performance record ---
    "equity": {"partition": (), "key": ("ts_ms",), "grain": "day"},
    "paper_fills": {"partition": ("coin",), "key": ("coin", "ts_ms", "fill_id"), "grain": "day"},
    "paper_trades": {"partition": ("coin",), "key": ("coin", "closed_ts_ms", "opened_ts_ms"), "grain": "day"},
    "decisions": {"partition": ("coin",), "key": ("coin", "ts_ms"), "grain": "day"},
}


def _sanitise(value: object) -> str:
    """Make a partition value safe for a path component."""
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def _grain_key(ts_ms: pd.Series, grain: str) -> pd.Series:
    ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
    fmt = "%Y-%m-%d" if grain == "day" else "%Y-%m"
    return ts.dt.strftime(fmt)


class ParquetStore:
    """Thin, dependency-light Parquet writer with idempotent upserts."""

    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = (paths or SETTINGS.paths).ensure()
        self.root = self.paths.parquet
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._part_counter = 0

    # -- paths -------------------------------------------------------------

    def dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    def glob(self, dataset: str) -> str:
        return str(self.dataset_dir(dataset) / "**" / "*.parquet")

    def _partition_dir(self, dataset: str, row: Mapping[str, object]) -> Path:
        spec = DATASET_SPECS[dataset]
        parts = [f"{col}={_sanitise(row[col])}" for col in spec["partition"]]  # type: ignore[index]
        # An empty partition tuple means everything lives directly under the
        # dataset directory, which is right for account-level series.
        return self.dataset_dir(dataset).joinpath(*parts)

    def _lock_for(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(str(path), threading.Lock())

    # -- writes ------------------------------------------------------------

    def upsert(self, dataset: str, df: pd.DataFrame) -> int:
        """Merge ``df`` into the dataset. Returns the number of NEW rows."""
        if df.empty:
            return 0
        spec = DATASET_SPECS[dataset]
        key: Sequence[str] = spec["key"]  # type: ignore[assignment]
        grain: str = spec["grain"]  # type: ignore[assignment]

        df = df.copy()
        df["_grain"] = _grain_key(df["ts_ms"], grain)
        group_cols = list(spec["partition"]) + ["_grain"]  # type: ignore[operator]

        new_rows = 0
        for group_values, chunk in df.groupby(group_cols, sort=False, observed=True):
            values = dict(zip(group_cols, group_values if isinstance(group_values, tuple) else (group_values,)))
            target_dir = self._partition_dir(dataset, values)
            target = target_dir / f"{values['_grain']}.parquet"
            chunk = chunk.drop(columns=["_grain"])

            with self._lock_for(target):
                target_dir.mkdir(parents=True, exist_ok=True)
                before = 0
                if target.exists():
                    existing = pq.read_table(target).to_pandas()
                    before = len(existing)
                    merged = pd.concat([existing, chunk], ignore_index=True)
                else:
                    merged = chunk
                merged = (
                    merged.drop_duplicates(subset=list(key), keep="last")
                    .sort_values("ts_ms")
                    .reset_index(drop=True)
                )
                self._atomic_write(merged, target)
                new_rows += len(merged) - before
        return new_rows

    def append(self, dataset: str, df: pd.DataFrame) -> int:
        """Write ``df`` as a fresh part file. Returns rows written."""
        if df.empty:
            return 0
        spec = DATASET_SPECS[dataset]
        grain: str = spec["grain"]  # type: ignore[assignment]

        df = df.copy()
        df["_grain"] = _grain_key(df["ts_ms"], grain)
        group_cols = list(spec["partition"]) + ["_grain"]  # type: ignore[operator]

        written = 0
        for group_values, chunk in df.groupby(group_cols, sort=False, observed=True):
            values = dict(zip(group_cols, group_values if isinstance(group_values, tuple) else (group_values,)))
            target_dir = self._partition_dir(dataset, values) / str(values["_grain"])
            target_dir.mkdir(parents=True, exist_ok=True)
            self._part_counter += 1
            stamp = int(chunk["ts_ms"].min())
            target = target_dir / f"part-{stamp}-{os.getpid()}-{self._part_counter}.parquet"
            self._atomic_write(chunk.drop(columns=["_grain"]).reset_index(drop=True), target)
            written += len(chunk)
        return written

    @staticmethod
    def _atomic_write(df: pd.DataFrame, target: Path) -> None:
        """Write via a temp file + rename so a crash cannot leave a torn file."""
        tmp = target.with_suffix(target.suffix + ".tmp")
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, target)

    # -- reads -------------------------------------------------------------

    def has_data(self, dataset: str) -> bool:
        directory = self.dataset_dir(dataset)
        return directory.exists() and any(directory.rglob("*.parquet"))

    def file_count(self, dataset: str) -> int:
        directory = self.dataset_dir(dataset)
        return sum(1 for _ in directory.rglob("*.parquet")) if directory.exists() else 0


class MarketDatabase:
    """DuckDB session with a view over each Parquet dataset.

    Views are created only for datasets that actually have files; DuckDB
    errors on a glob matching nothing, and an empty dataset is a normal
    state before the first collection run.
    """

    def __init__(self, store: ParquetStore | None = None, *, persistent: bool = False) -> None:
        self.store = store or ParquetStore()
        path = str(self.store.paths.duckdb_file) if persistent else ":memory:"
        self.con = duckdb.connect(path)
        self.refresh_views()

    def refresh_views(self) -> list[str]:
        created = []
        for dataset in DATASET_SPECS:
            if not self.store.has_data(dataset):
                self.con.execute(f"DROP VIEW IF EXISTS {dataset}")
                continue
            pattern = self.store.glob(dataset).replace("'", "''")
            self.con.execute(
                f"CREATE OR REPLACE VIEW {dataset} AS "
                f"SELECT *, epoch_ms(ts_ms) AS ts FROM read_parquet('{pattern}', union_by_name=true)"
            )
            created.append(dataset)
        return created

    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def table_summary(self) -> pd.DataFrame:
        """One row per dataset: rows, coins, time span, file count."""
        rows = []
        for dataset in DATASET_SPECS:
            if not self.store.has_data(dataset):
                rows.append(
                    {"dataset": dataset, "rows": 0, "coins": 0, "start": None, "end": None, "files": 0}
                )
                continue
            summary = self.query(
                f"SELECT count(*) AS rows, count(DISTINCT coin) AS coins, "
                f"min(ts) AS start, max(ts) AS end FROM {dataset}"
            ).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "rows": int(summary["rows"]),
                    "coins": int(summary["coins"]),
                    "start": summary["start"],
                    "end": summary["end"],
                    "files": self.store.file_count(dataset),
                }
            )
        return pd.DataFrame(rows)

    def last_candle_ts(self, coin: str, interval: str) -> int | None:
        if not self.store.has_data("candles"):
            return None
        result = self.query(
            "SELECT max(ts_ms) AS m FROM candles WHERE coin = ? AND interval = ?",
            [coin, interval],
        )
        value = result.iloc[0]["m"]
        return None if pd.isna(value) else int(value)

    def last_funding_ts(self, coin: str) -> int | None:
        if not self.store.has_data("funding"):
            return None
        result = self.query("SELECT max(ts_ms) AS m FROM funding WHERE coin = ?", [coin])
        value = result.iloc[0]["m"]
        return None if pd.isna(value) else int(value)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "MarketDatabase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
