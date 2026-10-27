"""Check the threshold curve and the VAMP-cap rule against brute force, on a synthetic sample.

Needs neither the dataset nor a trained model: the SQL runs on an in-memory DuckDB.
"""

import duckdb
import numpy as np
import pandas as pd

import policy

CAPS = {"tight": 0.005, "loose": 0.01, "looser": 0.02, "lax": 0.05}
PREDICTED = ["loose", "lax"]
FEES = {"tight": 4, "loose": 8, "looser": 10, "lax": 20}
TAKE = 0.002


def sample(n: int = 4000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    p = np.round(rng.beta(0.4, 8, n), 3)  # rounded, so ties exist as they do under isotonic
    y = (rng.random(n) < p).astype(int)
    return p, rng.lognormal(4, 1, n), y


def fee(ratio: float) -> float:
    over = [c for c in CAPS.values() if ratio > c]
    return FEES[next(k for k, c in CAPS.items() if c == max(over))] if over else 0.0


def connected(p, amt, y, caps=CAPS, stray=0, predicted=PREDICTED) -> duckdb.DuckDBPyConnection:
    """`stray` predictions have no eval_set row, as calibration-window scores would not."""
    con = duckdb.connect()
    ids = np.arange(len(p))
    con.register("eval_set", pd.DataFrame({"TransactionID": ids, "TransactionAmt": amt,
                                           "isFraud": y}))
    con.register("predictions", pd.DataFrame({
        "TransactionID": np.append(ids, len(p) + np.arange(stray)),
        "p_fraud": np.append(p, np.full(stray, 0.0)),
        "run_id": "test",
    }))
    policy.build(con, {"vamp_caps": caps, "headline_cap": next(iter(caps)), "take_rate": TAKE,
                       "predicted_ratio_caps": predicted,
                       "vamp_fee_per_event": {k: FEES.get(k, 0) for k in caps}})
    return con


def built(p, amt, y, caps=CAPS, stray=0, predicted=PREDICTED) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = connected(p, amt, y, caps, stray, predicted)
    return (con.execute("SELECT * FROM policy_curve").df(),
            con.execute("SELECT * FROM policy_thresholds").df())


def test_curve_matches_a_naive_loop() -> None:
    p, amt, y = sample()
    curve, _ = built(p, amt, y)
    assert len(curve) == len(np.unique(p)) + 1
    for r in curve.sample(25, random_state=1).itertuples():
        a = p < r.threshold
        assert r.approved == a.sum()
        assert r.frauds == y[a].sum()
        assert np.isclose(r.expected_frauds, p[a].sum())
        assert np.isclose(r.legit_value, amt[a & (y == 0)].sum())
        assert np.isclose(r.fraud_value, amt[a & (y == 1)].sum())


def brute_force(p, amt, y, cap, binds_predicted):
    best, best_value = None, -1.0
    for t in np.append(np.unique(p), np.inf):
        a = p < t
        ok = (y[a].mean() if a.any() else 0.0) <= cap
        if binds_predicted:
            ok &= (p[a].mean() if a.any() else 0.0) <= cap
        if ok and amt[a & (y == 0)].sum() >= best_value:
            best, best_value = t, amt[a & (y == 0)].sum()
    return best


def test_the_predicted_ratio_binds_only_where_configured() -> None:
    p, amt, y = sample()
    over = np.minimum(np.round(p * 1.5, 3), 1)  # labels drawn from p, so the model over-predicts
    _, free = built(over, amt, y, predicted=[])
    _, bound = built(over, amt, y)
    free, bound = free.set_index("cap_name"), bound.set_index("cap_name")
    for name, cap in CAPS.items():
        assert bound.threshold[name] == brute_force(over, amt, y, cap, name in PREDICTED)
        if name not in PREDICTED:
            assert bound.threshold[name] == free.threshold[name]
    assert (bound.threshold[PREDICTED] < free.threshold[PREDICTED]).any()
    assert (bound.expected_ratio[PREDICTED] <= pd.Series(CAPS)[PREDICTED]).all()


def test_predictions_outside_eval_set_are_not_priced() -> None:
    p, amt, y = sample()
    curve, _ = built(p, amt, y, stray=500)
    assert curve.approved.iloc[-1] == len(p)


def test_published_rows_are_priced_and_replaced_on_rerun(tmp_path) -> None:
    p, amt, y = sample()
    con = connected(p, amt, y)
    rows = policy.publish(con, tmp_path / "mart.duckdb")
    policy.publish(con, tmp_path / "mart.duckdb")
    stored = con.execute("SELECT * FROM mart.policy_thresholds").df()
    con.close()
    assert len(stored) == len(CAPS) and (stored.run_id == "test").all()
    assert (stored.vamp_fees > 0).any()
    assert stored.is_headline.sum() == 1
    points = rows.set_index("cap_name")
    for name in CAPS:
        a = p < points.threshold[name]
        assert np.isclose(points.revenue[name], TAKE * amt[a & (y == 0)].sum())
        assert points.vamp_events[name] == y[a].sum()
        assert np.isclose(points.vamp_fees[name], fee(y[a].mean()) * y[a].sum())
        assert np.isclose(points.net_revenue[name], points.revenue[name] - points.vamp_fees[name])
        assert np.isclose(points.forgone[name], TAKE * amt[~a & (y == 0)].sum())
        assert np.isclose(points.approved_share[name], a.mean())
