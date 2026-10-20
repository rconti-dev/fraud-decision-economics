"""Download the labelled IEEE-CIS files and load them into DuckDB.

Needs a Kaggle token and acceptance of the competition rules at
https://www.kaggle.com/c/ieee-fraud-detection/rules; the API returns 403 until then.
kagglehub caches the download outside this repo, so re-running does not download again.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import duckdb
import kagglehub
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB_PATH = ROOT / "data" / "raw" / "fraud.duckdb"
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def download() -> dict[str, Path]:
    slug = CONFIG["data"]["competition"]
    files = {}
    for name in CONFIG["data"]["files"]:
        print(f"fetching {name} ...", flush=True)
        files[name] = Path(kagglehub.competition_download(slug, path=name))
        print(f"  {files[name]}  {files[name].stat().st_size / 1e6:.0f} MB")
    return files


def write_manifest(files: dict[str, Path]) -> None:
    """Record what was fetched. The only file under data/ that is committed."""
    MANIFEST.write_text(
        json.dumps(
            {
                "competition": CONFIG["data"]["competition"],
                "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "files": [
                    {"file": name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                    for name, p in sorted(files.items())
                ],
            },
            indent=2,
        )
        + "\n"
    )


def load(files: dict[str, Path]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        Template((ROOT / "sql" / "00_load_raw.sql").read_text()).substitute(
            tx_csv=files["train_transaction.csv"], id_csv=files["train_identity.csv"]
        )
    )
    n, fraud = con.execute("SELECT count(*), avg(isFraud) FROM raw_transactions").fetchone()
    n_id = con.execute("SELECT count(*) FROM raw_identity").fetchone()[0]
    con.close()
    print(f"\n{n:,} transactions, {fraud:.2%} fraud, identity on {n_id / n:.0%}")
    print(f"-> data/raw/fraud.duckdb ({DB_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    files = download()
    write_manifest(files)
    load(files)
