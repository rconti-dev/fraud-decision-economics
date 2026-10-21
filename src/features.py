"""Run the SQL feature layer, in order, against the DuckDB build."""

import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "raw" / "fraud.duckdb"
STEPS = ["01_velocity_features.sql", "02_entity_aggregates.sql", "03_training_table.sql"]


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("data/raw/fraud.duckdb not found; run src/ingest.py first")

    con = duckdb.connect(str(DB_PATH))
    for name in STEPS:
        t0 = time.perf_counter()
        print(f"sql/{name} ...", end=" ", flush=True)
        con.execute((ROOT / "sql" / name).read_text())
        print(f"{time.perf_counter() - t0:.1f}s")

    n_rows = con.execute("SELECT count(*) FROM training_table").fetchone()[0]
    n_cols = len(con.execute("DESCRIBE training_table").fetchall())
    n_base = con.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
    con.close()

    print(f"\ntraining_table: {n_rows:,} rows x {n_cols} columns")
    if n_rows != n_base:  # a join that dropped or duplicated rows
        raise SystemExit(f"row count changed: {n_base:,} raw -> {n_rows:,} features")


if __name__ == "__main__":
    main()
