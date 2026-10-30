-- Key metrics at the highest threshold each VAMP cap allows, one row per cap and take-rate
-- scenario: the share of transactions and of value approved; the realised fraud ratio and the
-- predicted one (mean p of approved); the VAMP events and the fees Visa charges on them; revenue
-- at the take rate on approved legitimate value, net of fees, and forgone on declined legitimate
-- value. Only the model run in predictions; results.ipynb loads it once, as `thresholds`.

SELECT t.cap, t.threshold, t.approved_share, t.value_share, t.fraud_ratio,
       t.expected_ratio, t.vamp_events, t.vamp_fees, t.scenario, t.take_rate, t.revenue,
       t.net_revenue, t.forgone
FROM mart.policy_thresholds t
JOIN (SELECT DISTINCT run_id FROM predictions) USING (run_id)
ORDER BY t.cap, t.take_rate
