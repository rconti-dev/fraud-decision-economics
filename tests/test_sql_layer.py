"""Check the SQL feature layer against a synthetic fixture.

Runs without a Kaggle account and without the 1.3 GB download, so the window functions
in sql/ can be verified by anyone who clones this. The fixture reproduces the awkward
parts of the source schema: nullable card and address keys, a numeric addr1, missing
email domains, and identity rows for about a quarter of transactions.

The assertions are about leakage and join integrity, the two ways this layer can be
wrong and still look plausible downstream.

    uv run python tests/test_sql_layer.py
"""

import csv
import random
import sys
import tempfile
from pathlib import Path
from string import Template

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql"

TX_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "C1", "C2", "V1", "V2", "D1",
]
ID_COLS = ["TransactionID", "id_01", "id_31", "DeviceType", "DeviceInfo"]


def write_fixture(tmp: Path, n: int = 4000, seed: int = 0) -> tuple[Path, Path]:
    random.seed(seed)
    rows = []
    for i in range(n):
        rows.append([
            1000 + i,
            int(random.random() < 0.035),
            int(i * 400 + random.randint(0, 300)),
            round(random.lognormvariate(4, 1), 3),
            random.choice("WCRH"),
            random.choice([None] + list(range(1, 40))),     # card1 nullable
            random.randint(100, 600), 150,
            random.choice(["visa", "mastercard"]),
            random.randint(100, 240),
            random.choice(["debit", "credit"]),
            random.choice([None] + list(range(100, 140))),  # addr1 numeric and nullable
            87,
            random.choice([None, "gmail.com", "yahoo.com", "aol.com"]),
            random.choice([None, "gmail.com"]),
            random.randint(0, 5), random.randint(0, 5),
            random.random(), random.random(), random.randint(0, 300),
        ])

    tx, idf = tmp / "train_transaction.csv", tmp / "train_identity.csv"
    with tx.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(TX_COLS); w.writerows(rows)
    with idf.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(ID_COLS)
        for r in rows:
            if random.random() < 0.24:  # matches the real ~24% identity coverage
                w.writerow([r[0], random.randint(-100, 0), "chrome 62.0",
                            random.choice(["desktop", "mobile"]),
                            random.choice(["Windows", "iOS"])])
    return tx, idf


def build(con: duckdb.DuckDBPyConnection, tx: Path, idf: Path) -> None:
    con.execute(Template((SQL / "00_load_raw.sql").read_text())
                .substitute(tx_csv=str(tx), id_csv=str(idf)))
    for step in ("01_velocity_features.sql", "02_entity_aggregates.sql",
                 "03_training_table.sql"):
        con.execute((SQL / step).read_text())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tx, idf = write_fixture(tmp)
        con = duckdb.connect(str(tmp / "test.duckdb"))
        build(con, tx, idf)

        checks = con.execute((ROOT / "tests" / "sql_layer_checks.sql").read_text()).fetchall()

        cov = con.execute(
            "SELECT avg(CAST(has_identity AS INTEGER)) FROM training_table"
        ).fetchone()[0]
        con.close()

        failed = 0
        for _, label, violations in checks:
            ok = violations == 0
            failed += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
                  + ("" if ok else f"  ({violations} violations)"))
        print(f"  [INFO] identity coverage in fixture: {cov:.0%}")
        print("\n" + ("all checks passed" if not failed else f"{failed} check(s) FAILED"))
        return 1 if failed else 0


def test_sql_layer() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
