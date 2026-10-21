-- All-history-to-date aggregates and inter-arrival timing. See README.md.
--
-- Deliberately absent: any per-entity fraud rate, the share of an entity's transactions
-- labelled fraud. Vesta labelled by account: one chargeback marks the account and every
-- later transaction linked to it, so an entity's fraud rate largely restates the label of
-- the row it sits on. Everything here is built from amounts and timestamps, never isFraud.

CREATE OR REPLACE TABLE entity_aggregates AS
WITH entity_long AS (
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
expanding AS (
    SELECT
        TransactionID,
        entity,
        -- First sighting: the entity's first transaction in the data (age 0, nseen 1). It
        -- carries a different risk from an entity with two months of history behind it.
        TransactionDT - first_value(TransactionDT) OVER wall   AS age,
        count(*)                                   OVER wall   AS nseen,
        -- Card testing shows up as a collapse in this gap. NULL on first sighting,
        -- which is meaningful and is left as NULL.
        TransactionDT - lag(TransactionDT)         OVER worder AS gap,
        avg(TransactionAmt)                        OVER wall   AS amtmean,
        (TransactionAmt - avg(TransactionAmt) OVER wall)
            / nullif(stddev_samp(TransactionAmt) OVER wall, 0) AS amtz
    FROM entity_long
    WINDOW
        worder AS (PARTITION BY entity, entity_key ORDER BY TransactionDT),
        wall   AS (PARTITION BY entity, entity_key ORDER BY TransactionDT
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
PIVOT expanding
    ON entity
    USING max(age)     AS age,
          max(nseen)   AS nseen,
          max(gap)     AS gap,
          max(amtmean) AS amtmean,
          max(amtz)    AS amtz
    GROUP BY TransactionID;

CREATE UNIQUE INDEX IF NOT EXISTS idx_entagg_id ON entity_aggregates (TransactionID);
