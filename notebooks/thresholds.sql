-- The mart's rows for the run behind predictions: one per cap and take-rate scenario, read
-- once by results.ipynb for every section. Read-only.

SELECT t.cap, t.threshold, t.approved_share, t.value_share, t.fraud_ratio,
       t.expected_ratio, t.vamp_events, t.vamp_fees, t.scenario, t.take_rate, t.revenue,
       t.net_revenue, t.forgone
FROM mart.policy_thresholds t
JOIN (SELECT DISTINCT run_id FROM predictions) USING (run_id)
ORDER BY t.cap, t.take_rate
