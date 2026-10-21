-- One row per invariant, (n, check_name, violations). Every check passes at zero.

WITH recount AS (
    -- The check that matters: a RANGE frame that silently became a ROWS frame
    -- would still produce plausible-looking numbers.
    SELECT a.card_n24h,
           (SELECT count(*) FROM training_table b
            WHERE b.card1 = a.card1
              AND b.TransactionDT BETWEEN a.TransactionDT - 86400
                                      AND a.TransactionDT) AS truth
    FROM training_table a WHERE a.card1 IS NOT NULL LIMIT 500
)
SELECT 1 AS n, 'rolling 24h count matches brute-force recount' AS check_name,
       count(*) AS violations
FROM recount WHERE card_n24h IS DISTINCT FROM truth

UNION ALL
SELECT 2, 'no window frame reaches into the future', count(*)
FROM training_table a
WHERE a.card1 IS NOT NULL AND a.card_n24h > (
    SELECT count(*) FROM training_table b
    WHERE b.card1 = a.card1 AND b.TransactionDT <= a.TransactionDT)

UNION ALL
SELECT 3, 'NULL entity keys excluded, not grouped', count(*)
FROM training_table WHERE card1 IS NULL AND card_n24h IS NOT NULL

UNION ALL
SELECT 4, 'LEFT joins preserve the row count',
       abs((SELECT count(*) FROM training_table) - (SELECT count(*) FROM raw_transactions))

UNION ALL
SELECT 5, 'inter-arrival gaps are non-negative', count(*)
FROM training_table WHERE card_gap < 0

UNION ALL
SELECT 6, 'entity age is non-negative', count(*)
FROM training_table WHERE uid_age < 0

ORDER BY n;
