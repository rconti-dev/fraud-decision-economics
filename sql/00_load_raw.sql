-- Load the labelled files verbatim, then define the view every feature query builds on.
-- What each query in this directory does, and why: README.md.

CREATE OR REPLACE TABLE raw_transactions AS
SELECT * FROM read_csv('$tx_csv', header = true);

CREATE OR REPLACE TABLE raw_identity AS
SELECT * FROM read_csv('$id_csv', header = true);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_id ON raw_transactions (TransactionID);
CREATE UNIQUE INDEX IF NOT EXISTS idx_id_id ON raw_identity (TransactionID);

-- The join must stay LEFT: only few transactions carry an identity row, and an
-- inner join would discard a lot of data and move the fraud rate.
-- TransactionDT is seconds from an unknown reference, so day and hour are derived by
-- arithmetic rather than converted to a date.
CREATE OR REPLACE VIEW base AS
SELECT
    t.*,
    i.* EXCLUDE (TransactionID),
    CAST(t.TransactionDT / 86400 AS INTEGER)          AS day_index,
    t.TransactionDT % 86400                           AS seconds_of_day,
    CAST((t.TransactionDT % 86400) / 3600 AS INTEGER) AS hour_of_day,
    CAST(t.TransactionDT / 86400 AS INTEGER) % 7      AS dow_index,
    i.TransactionID IS NOT NULL                       AS has_identity,
    -- Composite account key: card, billing region, purchaser email domain. Labels were
    -- propagated per account, so fraud clusters on it; the split built later must therefore be
    -- chronological, or one account's frauds land on both sides and the model memorises it.
    -- concat_ws skips NULLs, so the '?' sentinel holds each field's position and keeps a missing
    -- addr1 apart from a missing email. addr1 is numeric, hence the cast before the coalesce.
    -- No card1, no account: uid stays NULL and the row is left out of per-entity features.
    CASE
        WHEN t.card1 IS NULL THEN NULL
        ELSE concat_ws('|', CAST(t.card1 AS VARCHAR),
                            coalesce(CAST(t.addr1 AS VARCHAR), '?'),
                            coalesce(CAST(t.P_emaildomain AS VARCHAR), '?'))
    END                                               AS uid
FROM raw_transactions t
LEFT JOIN raw_identity i USING (TransactionID);
