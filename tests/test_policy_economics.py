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
TAKES = {"low": 0.002, "high": 0.005}


def sample(n: int = 4000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    p = np.round(rng.beta(0.4, 8, n), 3)  # rounded, so ties exist as they do under isotonic
    y = (rng.random(n) < p).astype(int)
    return p, rng.lognormal(4, 1, n), y


def fee(ratio: float) -> float:
    over = [c for c in CAPS.values() if ratio > c]
    return FEES[next(k for k, c in CAPS.items() if c == max(over))] if over else 0.0


def connected(p, amt, y, caps=CAPS, stray=0, predicted=PREDICTED, fees=FEES,
              takes=TAKES) -> duckdb.DuckDBPyConnection:
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
    policy.build(con, {"vamp_caps": caps, "take_rates": takes,
                       "predicted_ratio_caps": predicted,
                       "vamp_fee_per_event": {k: fees.get(k, 0) for k in caps}})
    return con


def built(p, amt, y, caps=CAPS, stray=0, predicted=PREDICTED, fees=FEES,
          takes=TAKES) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = connected(p, amt, y, caps, stray, predicted, fees, takes)
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
    free, bound = (d.drop_duplicates("cap_name").set_index("cap_name") for d in (free, bound))
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
    assert len(stored) == len(CAPS) * len(TAKES) and (stored.run_id == "test").all()
    assert (stored.vamp_fees > 0).any()
    assert (stored.groupby("cap_name").threshold.nunique() == 1).all()
    for r in rows.itertuples():
        a, take = p < r.threshold, TAKES[r.scenario]
        assert r.take_rate == take
        assert np.isclose(r.revenue, take * amt[a & (y == 0)].sum())
        assert r.vamp_events == y[a].sum()
        assert np.isclose(r.vamp_fees, fee(y[a].mean()) * y[a].sum())
        assert np.isclose(r.net_revenue, r.revenue - r.vamp_fees)
        assert np.isclose(r.forgone, take * amt[~a & (y == 0)].sum())
        assert np.isclose(r.approved_share, a.mean())


def best_net_in_band(p, amt, y, lo, hi, take, fee_per_event):
    """Brute-force argmax of net revenue over the thresholds whose ratio is in (lo, hi]."""
    best, best_net, steps_pay, prev = None, -np.inf, True, None
    for t in np.append(np.unique(p), np.inf):
        a = p < t
        if not a.any() or not lo < y[a].mean() <= hi:
            continue
        legit, frauds = amt[a & (y == 0)].sum(), y[a].sum()
        if prev and frauds > prev[1]:
            steps_pay &= (legit - prev[0]) / (frauds - prev[1]) > fee_per_event / take
        prev = legit, frauds
        if take * legit - fee_per_event * frauds >= best_net:
            best, best_net = t, take * legit - fee_per_event * frauds
    return best, steps_pay


def test_top_of_the_band_is_best_paid_only_while_every_step_pays() -> None:
    p, amt, y = sample()
    caps, take = {"lo": 0.01, "hi": 0.02}, 0.05
    for fee_per_event, pays in [(4, True), (5000, False)]:
        _, points = built(p, amt, y, caps, predicted=[], fees={"lo": fee_per_event, "hi": 8},
                          takes={"only": take})
        chosen = points.set_index("cap_name").threshold["hi"]
        best, steps_pay = best_net_in_band(p, amt, y, 0.01, 0.02, take, fee_per_event)
        assert steps_pay == pays
        assert (chosen == best) == pays
