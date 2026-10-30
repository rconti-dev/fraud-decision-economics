-- Each threshold step inside a fee band that approves at least one more fraud: the frauds and
-- legitimate value it adds, and min_take_rate, the take rate below which its VAMP fees outweigh
-- its revenue. 

WITH run AS (
    SELECT t.* FROM mart.policy_thresholds t
    JOIN (SELECT DISTINCT run_id FROM predictions) USING (run_id)
),
chosen AS (
    SELECT DISTINCT h.cap, h.threshold, h.vamp_fees / h.vamp_events AS fee,
           (SELECT max(l.cap) FROM run l WHERE l.cap < h.cap) AS lo
    FROM run h WHERE h.vamp_fees > 0
),
band AS (
    SELECT k.cap, c.threshold, k.fee,
           c.frauds - lag(c.frauds) OVER w AS extra_frauds,
           c.legit_value - lag(c.legit_value) OVER w AS extra_legit_value
    FROM policy_curve c JOIN chosen k ON c.fraud_ratio > k.lo AND c.threshold <= k.threshold
    WINDOW w AS (PARTITION BY k.cap ORDER BY c.threshold)
)
SELECT cap, threshold, extra_frauds, extra_legit_value,
       extra_legit_value / extra_frauds AS legit_per_fraud,
       fee * extra_frauds / extra_legit_value AS min_take_rate
-- lag() is NULL on a band's first row, so the step into the band is left out: it switches the
-- fee on for every approved fraud, not only the new ones.
FROM band WHERE extra_frauds > 0 ORDER BY cap, threshold
