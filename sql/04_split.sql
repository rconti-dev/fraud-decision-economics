-- Chronological split, as views over training_table. A string.Template: src/split.py fills
-- in the day boundaries from config.yaml.
-- Never random: labels were propagated by account, so a random split puts the same
-- fraudster on both sides and the model is scored on accounts it has already seen.

CREATE OR REPLACE TABLE split_windows AS
SELECT * FROM (VALUES (1, 'train', $train_lo, $train_hi),
                      (2, 'calib', $calib_lo, $calib_hi),
                      (3, 'eval',  $eval_lo,  $eval_hi)) AS w (ord, name, lo, hi);

CREATE OR REPLACE VIEW train_set AS
SELECT t.* FROM training_table t JOIN split_windows w
    ON w.name = 'train' AND t.day_index >= w.lo AND t.day_index < w.hi;

CREATE OR REPLACE VIEW calib_set AS
SELECT t.* FROM training_table t JOIN split_windows w
    ON w.name = 'calib' AND t.day_index >= w.lo AND t.day_index < w.hi;

CREATE OR REPLACE VIEW eval_set AS
SELECT t.* FROM training_table t JOIN split_windows w
    ON w.name = 'eval' AND t.day_index >= w.lo AND t.day_index < w.hi;

-- Windows that are empty or do not start where the previous one ends; none on a valid split.
SELECT name
FROM (SELECT name, lo, hi, lag(hi) OVER (ORDER BY ord) AS prev_hi FROM split_windows)
WHERE lo <> prev_hi OR lo >= hi;
