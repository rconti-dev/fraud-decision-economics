"""Set the approve/decline threshold that keeps the portfolio's fraud ratio under each VAMP cap.

    uv run python src/policy.py
"""

from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB_PATH = ROOT / "data" / "raw" / "fraud.duckdb"
MART_PATH = ROOT / "data" / "mart.duckdb"

SQL = ROOT / "sql"


def build(con: duckdb.DuckDBPyConnection, econ: dict) -> None:
    """Expects `eval_set` and `predictions` on the connection; `econ` is config's economics."""
    caps = econ["vamp_caps"]
    con.register("caps", pd.DataFrame({
        "cap_name": list(caps), "cap": list(caps.values()),
        "binds_predicted": [name in econ["predicted_ratio_caps"] for name in caps],
        "fee_per_event": [float(econ["vamp_fee_per_event"][name]) for name in caps],
    }))
    rates = econ["take_rates"]
    con.register("take_rates", pd.DataFrame({
        "scenario": list(rates), "take_rate": [float(r) for r in rates.values()],
    }))
    con.execute((SQL / "05_policy_curve.sql").read_text())
    con.execute((SQL / "06_policy_thresholds.sql").read_text())


def publish(con: duckdb.DuckDBPyConnection, path: Path = MART_PATH) -> pd.DataFrame:
    """Copies policy_thresholds to the mart; a re-run replaces the run's rows."""
    con.execute(f"ATTACH IF NOT EXISTS '{path}' AS mart")
    con.execute((SQL / "mart.sql").read_text())
    con.execute("INSERT OR REPLACE INTO mart.policy_thresholds BY NAME "
                "SELECT * FROM policy_thresholds")
    return con.execute("SELECT * FROM policy_thresholds").df()


def split_matches_config(con: duckdb.DuckDBPyConnection) -> bool:
    row = con.execute("SELECT lo, hi FROM split_windows WHERE name = 'eval'").fetchone()
    return list(row or ()) == CONFIG["split"]["eval_days"]


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("data/raw/fraud.duckdb not found; run src/ingest.py first")
    con = duckdb.connect(str(DB_PATH))
    if not con.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'predictions'"
    ).fetchone()[0]:
        raise SystemExit("predictions not found; run src/train.py first")
    if not split_matches_config(con):
        raise SystemExit("split_windows disagrees with split.eval_days; re-run src/split.py")

    run_ids = [r[0] for r in con.execute("SELECT DISTINCT run_id FROM predictions").fetchall()]
    if len(run_ids) != 1:
        raise SystemExit("predictions carry no single run_id; re-run src/train.py")

    build(con, CONFIG["economics"])
    n = int(con.execute(
        "SELECT approved FROM policy_curve WHERE threshold = 'inf'::DOUBLE"
    ).fetchone()[0])
    n_eval = con.execute("SELECT count(*) FROM eval_set").fetchone()[0]
    assert n == n_eval, f"{n_eval - n:,} eval_set rows have no prediction; re-run src/train.py"
    publish(con)
    con.close()
    print(f"run {run_ids[0]}: {n:,} txn -> data/mart.duckdb (policy_thresholds)")


if __name__ == "__main__":
    main()
