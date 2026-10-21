-- Rolling-window velocity per entity. See README.md in this directory.
--
-- Two rules govern every frame below:
--   1. Never FOLLOWING. At authorisation the acquirer knows this transaction and the
--      earlier ones, nothing else.
--   2. RANGE, not ROWS. The windows are elapsed seconds of TransactionDT. ROWS would
--      give "the last n transactions" whether they span a minute or a month.

CREATE OR REPLACE TABLE velocity AS
WITH entity_long AS (
    -- NULL keys are dropped, not partitioned on: grouping them would collapse every
    -- unknown-email transaction into one partition and manufacture velocity.
    SELECT TransactionID, TransactionDT, TransactionAmt,
           'card'  AS entity, CAST(card1 AS VARCHAR) AS entity_key
    FROM base WHERE card1 IS NOT NULL
  UNION ALL
    SELECT TransactionID, TransactionDT, TransactionAmt,
           'addr', CAST(addr1 AS VARCHAR)
    FROM base WHERE addr1 IS NOT NULL
  UNION ALL
    SELECT TransactionID, TransactionDT, TransactionAmt,
           'email', P_emaildomain
    FROM base WHERE P_emaildomain IS NOT NULL
  UNION ALL
    SELECT TransactionID, TransactionDT, TransactionAmt,
           'uid', uid
    FROM base WHERE uid IS NOT NULL
),
windowed AS (
    SELECT
        TransactionID,
        entity,
        count(*)            OVER w1h  AS n1h,
        count(*)            OVER w24h AS n24h,
        count(*)            OVER w7d  AS n7d,
        sum(TransactionAmt) OVER w24h AS amt24h,
        sum(TransactionAmt) OVER w7d  AS amt7d,
        -- This amount against the entity's own recent norm. A card that normally runs
        -- $40 suddenly attempting $900 is the escalation pattern the raw amount misses.
        TransactionAmt / nullif(avg(TransactionAmt) OVER w7d, 0) AS amtratio7d
    FROM entity_long
    WINDOW
        w1h  AS (PARTITION BY entity, entity_key ORDER BY TransactionDT
                 RANGE BETWEEN 3600   PRECEDING AND CURRENT ROW),
        w24h AS (PARTITION BY entity, entity_key ORDER BY TransactionDT
                 RANGE BETWEEN 86400  PRECEDING AND CURRENT ROW),
        w7d  AS (PARTITION BY entity, entity_key ORDER BY TransactionDT
                 RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW)
)
-- Back to one row per transaction, as <entity>_<stat>: uid_n24h, card_amtratio7d.
-- Each aggregate collapses a single row, since (TransactionID, entity) is unique here.
PIVOT windowed
    ON entity
    USING max(n1h)        AS n1h,
          max(n24h)       AS n24h,
          max(n7d)        AS n7d,
          max(amt24h)     AS amt24h,
          max(amt7d)      AS amt7d,
          max(amtratio7d) AS amtratio7d
    GROUP BY TransactionID;

CREATE UNIQUE INDEX IF NOT EXISTS idx_velocity_id ON velocity (TransactionID);
