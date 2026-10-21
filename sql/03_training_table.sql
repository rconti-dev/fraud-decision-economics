-- Assemble the modelling table. Nothing here is filtered by label or by date: the
-- chronological split happens in src/split.py, from boundaries in config.yaml.
-- See README.md.

CREATE OR REPLACE TABLE training_table AS
SELECT
    b.*,
    v.* EXCLUDE (TransactionID),
    e.* EXCLUDE (TransactionID),

    ln(b.TransactionAmt + 1) AS amt_log,

    -- Fractional cents: a currency-of-origin proxy, not a throwaway. Converted
    -- foreign amounts leave a long decimal; domestic retail clusters on .00 and .99.
    b.TransactionAmt - floor(b.TransactionAmt)   AS amt_cents,
    (b.TransactionAmt = floor(b.TransactionAmt)) AS amt_is_round
FROM base b
LEFT JOIN velocity v          USING (TransactionID)
LEFT JOIN entity_aggregates e USING (TransactionID);

CREATE UNIQUE INDEX IF NOT EXISTS idx_training_id ON training_table (TransactionID);
