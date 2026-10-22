"""Create the chronological train_set, calib_set and eval_set views over training_table.

    uv run python src/split.py
"""

from pathlib import Path
from string import Template

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB_PATH = ROOT / "data" / "raw" / "fraud.duckdb"
WINDOWS = ("train", "calib", "eval")


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("data/raw/fraud.duckdb not found; run src/ingest.py first")
    con = duckdb.connect(str(DB_PATH))
    if not con.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'training_table'"
    ).fetchone()[0]:
        raise SystemExit("training_table not found; run src/features.py first")

    days = {f"{name}_{end}": d for name in WINDOWS
            for end, d in zip(("lo", "hi"), CONFIG["split"][f"{name}_days"])}
    sql = Template((ROOT / "sql" / "04_split.sql").read_text()).substitute(days)

    con.begin()  # a rejected split must not replace the previous views
    if bad := con.execute(sql).fetchall():
        con.rollback()
        raise SystemExit(f"split windows in config.yaml rejected: {bad}")
    con.commit()

    for name in WINDOWS:
        n, fraud, amt = con.execute(
            f"SELECT count(*), avg(isFraud), sum(TransactionAmt) FROM {name}_set"
        ).fetchone()
        print(f"{name + '_set':<10} days {days[name + '_lo']:>3}-{days[name + '_hi']:<3}  "
              f"{n:>7,} txn  {fraud:.2%} fraud  ${amt / 1e6:.1f}M")
    con.close()


if __name__ == "__main__":
    main()
