"""Fit on train_set, calibrate on calib_set, write eval_set probabilities to predictions.

    uv run python src/train.py [--random-split]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB_PATH = ROOT / "data" / "raw" / "fraud.duckdb"
MART_PATH = ROOT / "data" / "mart.duckdb"
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"

# Columns the model must not learn from. TransactionDT and day_index are just the date: a model
# that learned "fraud was high in week 10" would be useless next month. uid identifies the
# account, and Vesta labelled whole accounts at once, so it would let the model memorise answers.
DROP = ["TransactionID", "isFraud", "TransactionDT", "day_index", "uid"]


def cast(X: pd.DataFrame, cats: dict[str, pd.CategoricalDtype] | None = None) -> pd.DataFrame:
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(X[c]):
            if cats:
                # A category the training window never saw is missing to the model either way.
                X[c] = X[c].where(X[c].isin(cats[c].categories)).astype(cats[c])
            else:
                X[c] = X[c].astype("category")
        else:
            # Int64 and boolean carry pd.NA, which does not survive a direct float32 cast.
            X[c] = X[c].astype("Float32").to_numpy("float32", na_value=np.nan)
    return X


def frame(con: duckdb.DuckDBPyConnection, view: str, cats=None) -> tuple:
    df = con.execute(f"SELECT * FROM {view}").df()
    ids = df["TransactionID"].to_numpy()
    y = df["isFraud"].to_numpy("int8")
    day = df["day_index"].to_numpy()
    X = cast(df.drop(columns=DROP), cats)
    return ids, X, y, day


def fit(X_fit, y_fit, X_val, y_val, seed: int) -> lgb.Booster:
    params = dict(CONFIG["model"]) | {"seed": seed}
    params.pop("seeds")
    rounds = params.pop("num_boost_round")
    stopping = params.pop("early_stopping_rounds")
    # No is_unbalance or scale_pos_weight. Frauds are only 3.5% of rows, and both options make the
    # model count each fraud as several. That inflates every score: 2% would stop meaning 2% risk.
    params |= {"metric": "average_precision", "verbosity": -1}

    train = lgb.Dataset(X_fit, y_fit)
    valid = lgb.Dataset(X_val, y_val, reference=train)
    return lgb.train(
        params, train, num_boost_round=rounds, valid_sets=[valid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(100)],
    )


def calibrator(p_raw: np.ndarray, y: np.ndarray):
    return IsotonicRegression(out_of_bounds="clip").fit(p_raw, y).predict


def reliability_plot(y, p_raw, p_cal) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 0.6], [0, 0.6], lw=1, ls="--", color="#999")
    for p, name, colour in [(p_raw, "raw score", "#777"), (p_cal, "calibrated", "#c1440e")]:
        obs, pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
        ax.plot(pred, obs, "o-", ms=3, lw=1.2, color=colour, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted fraud probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Reliability, evaluation window", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    print("  figure -> outputs/figures/calibration.png")


def random_split_check(con: duckdb.DuckDBPyConnection) -> float:
    _, X, y, day = frame(con, "training_table")
    cfg = CONFIG["split"]
    fit_hi = cfg["train_days"][1] - cfg["valid_days"]
    n_fit = int((day < fit_hi).sum())
    n_val = int(((day >= fit_hi) & (day < cfg["train_days"][1])).sum())
    n_eval = int((day >= cfg["eval_days"][0]).sum())

    seed = CONFIG["model"]["seeds"][0]
    order = np.random.default_rng(seed).permutation(len(y))
    i_fit, i_val, rest = np.split(order, [n_fit, n_fit + n_val])
    i_eval = rest[:n_eval]

    booster = fit(X.iloc[i_fit], y[i_fit], X.iloc[i_val], y[i_val], seed)
    return average_precision_score(y[i_eval], booster.predict(X.iloc[i_eval]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-split", action="store_true",
                        help="also fit under a random partition, to size the leakage")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit("data/raw/fraud.duckdb not found; run src/ingest.py first")
    con = duckdb.connect(str(DB_PATH))
    if not con.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = 'eval_set'"
    ).fetchone()[0]:
        raise SystemExit("split views not found; run src/split.py first")

    run_at = datetime.now(UTC).replace(microsecond=0)
    run_id = run_at.strftime("%Y%m%d-%H%M%S")
    cfg = CONFIG["split"]
    _, X_tr, y_tr, day_tr = frame(con, "train_set")
    cats = {c: t for c in X_tr.columns
            if isinstance(t := X_tr[c].dtype, pd.CategoricalDtype)}

    # Not calib_set: the round count must not be chosen on the rows the calibrator uses.
    held = day_tr >= cfg["train_days"][1] - cfg["valid_days"]
    print(f"fitting on {(~held).sum():,} txn, early stopping on {held.sum():,}")
    # One booster per seed, averaged: a single seed moved approval at the cap by ten points.
    boosters = []
    for seed in CONFIG["model"]["seeds"]:
        booster = fit(X_tr[~held], y_tr[~held], X_tr[held], y_tr[held], seed)
        booster.save_model(str(DATA / f"model_{seed}.txt"), num_iteration=booster.best_iteration)
        print(f"  seed {seed}: {booster.best_iteration} rounds -> data/model_{seed}.txt")
        boosters.append(booster)
    n_features = X_tr.shape[1]
    del X_tr

    _, X_ca, y_ca, _ = frame(con, "calib_set", cats)
    ids_ev, X_ev, y_ev, _ = frame(con, "eval_set", cats)
    raw_ca = np.mean([b.predict(X_ca) for b in boosters], axis=0)
    raw_ev = np.mean([b.predict(X_ev) for b in boosters], axis=0)
    cal_ev = calibrator(raw_ca, y_ca)(raw_ev)

    reliability_plot(y_ev, raw_ev, cal_ev)

    preds = pd.DataFrame({"TransactionID": ids_ev, "p_fraud": cal_ev, "run_id": run_id})
    con.register("preds_df", preds)
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM preds_df")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_id ON predictions (TransactionID)")

    pr_auc = average_precision_score(y_ev, cal_ev)
    run = pd.DataFrame([{
        "run_id": run_id, "run_at": run_at,
        "seeds": CONFIG["model"]["seeds"], "rounds": [b.best_iteration for b in boosters],
        "n_features": n_features,
        "config": json.dumps({k: CONFIG[k] for k in ("split", "model")}),
        "eval_n": len(y_ev), "pr_auc": pr_auc, "roc_auc": roc_auc_score(y_ev, cal_ev),
        "brier_raw": brier_score_loss(y_ev, raw_ev),
        "brier_calibrated": brier_score_loss(y_ev, cal_ev),
        "mean_predicted": cal_ev.mean(), "observed_rate": y_ev.mean(),
        "pr_auc_random_split": random_split_check(con) if args.random_split else None,
    }])
    con.execute(f"ATTACH IF NOT EXISTS '{MART_PATH}' AS mart")
    con.execute((ROOT / "sql" / "mart.sql").read_text())
    con.register("run_df", run)
    con.execute("INSERT OR REPLACE INTO mart.model_runs BY NAME SELECT * FROM run_df")
    con.close()
    print(f"\nrun {run_id}: PR-AUC {pr_auc:.4f} on {len(y_ev):,} txn "
          f"-> data/mart.duckdb (model_runs)")


if __name__ == "__main__":
    main()
