-- Realised outcome of every candidate threshold. See README.md in this directory.

CREATE OR REPLACE TABLE policy_curve AS
WITH scored AS (
    SELECT p.p_fraud, e.TransactionAmt, e.isFraud
    FROM eval_set e JOIN predictions p USING (TransactionID)
),
levels AS (
    SELECT p_fraud AS threshold, count(*) AS n, sum(isFraud) AS frauds,
           sum(p_fraud) AS expected_frauds,
           sum(TransactionAmt * (1 - isFraud)) AS legit_value,
           sum(TransactionAmt * isFraud) AS fraud_value
    FROM scored GROUP BY p_fraud
    UNION ALL
    SELECT 'inf'::DOUBLE, 0, 0, 0, 0, 0
),
running AS (
    -- Ends at 1 PRECEDING: approval is p < t, so a level is declined at its own threshold.
    SELECT threshold,
           coalesce(sum(n) OVER w, 0) AS approved,
           coalesce(sum(frauds) OVER w, 0) AS frauds,
           coalesce(sum(expected_frauds) OVER w, 0) AS expected_frauds,
           coalesce(sum(legit_value) OVER w, 0) AS legit_value,
           coalesce(sum(fraud_value) OVER w, 0) AS fraud_value
    FROM levels
    WINDOW w AS (ORDER BY threshold ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT *, frauds / greatest(approved, 1) AS fraud_ratio,
       expected_frauds / greatest(approved, 1) AS expected_ratio
FROM running ORDER BY threshold;
