-- The highest threshold under each VAMP cap, priced as the mart stores it. Expects `caps`
-- (cap_name, cap, is_headline, binds_predicted, fee_per_event), policy_curve and predictions;
-- $take_rate is bound by policy.py.
-- The fraud ratio is not monotone in the threshold, so this keeps the last threshold under the
-- cap, not the first one over it. Approved legitimate value only grows as the threshold rises,
-- so the last one under the cap is also the best paid.

CREATE OR REPLACE TEMP TABLE policy_thresholds AS
WITH everything AS (SELECT * FROM policy_curve WHERE threshold = 'inf'::DOUBLE),
chosen AS (
    SELECT k.cap_name, k.cap, k.is_headline, k.binds_predicted, c.*
    FROM caps k JOIN policy_curve c
      ON c.fraud_ratio <= k.cap AND (NOT k.binds_predicted OR c.expected_ratio <= k.cap)
    QUALIFY row_number() OVER (PARTITION BY k.cap_name ORDER BY c.threshold DESC) = 1
),
priced AS (
    -- Every counted event pays the fee of the highest cap the realised ratio is over.
    SELECT c.*, coalesce((SELECT arg_max(f.fee_per_event, f.cap) FROM caps f
                          WHERE c.fraud_ratio > f.cap), 0) * c.frauds AS vamp_fees
    FROM chosen c
)
SELECT (SELECT any_value(run_id) FROM predictions) AS run_id,
       c.cap_name, c.cap, c.is_headline, c.binds_predicted, c.threshold,
       c.approved / e.approved AS approved_share,
       (c.legit_value + c.fraud_value) / (e.legit_value + e.fraud_value) AS value_share,
       c.fraud_ratio, c.expected_ratio,
       c.fraud_value / e.fraud_value AS fraud_value_share,
       $take_rate::DOUBLE AS take_rate,
       $take_rate::DOUBLE * c.legit_value AS revenue,
       c.frauds::INTEGER AS vamp_events, c.vamp_fees,
       $take_rate::DOUBLE * c.legit_value - c.vamp_fees AS net_revenue,
       $take_rate::DOUBLE * (e.legit_value - c.legit_value) AS forgone,
       now() AS built_at
FROM priced c CROSS JOIN everything e
ORDER BY c.cap;
