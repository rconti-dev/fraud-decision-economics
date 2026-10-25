-- The data mart, attached as `mart` by train.py and policy.py. Keyed by run, and kept across
-- rebuilds of the main database, so runs stay comparable.

CREATE TABLE IF NOT EXISTS mart.model_runs (
    run_id VARCHAR PRIMARY KEY,
    run_at TIMESTAMPTZ,
    seeds INTEGER[],
    rounds INTEGER[],
    n_features INTEGER,
    config JSON,
    eval_n INTEGER,
    pr_auc DOUBLE,
    roc_auc DOUBLE,
    brier_raw DOUBLE,
    brier_calibrated DOUBLE,
    mean_predicted DOUBLE,
    observed_rate DOUBLE,
    -- NULL unless train.py ran with --random-split.
    pr_auc_random_split DOUBLE
);

CREATE TABLE IF NOT EXISTS mart.policy_thresholds (
    run_id VARCHAR,
    cap_name VARCHAR,
    cap DOUBLE,
    is_headline BOOLEAN,
    binds_predicted BOOLEAN,
    threshold DOUBLE,
    approved_share DOUBLE,
    value_share DOUBLE,
    fraud_ratio DOUBLE,
    expected_ratio DOUBLE,
    fraud_value_share DOUBLE,
    take_rate DOUBLE,
    revenue DOUBLE,
    vamp_events INTEGER,
    vamp_fees DOUBLE,
    net_revenue DOUBLE,
    forgone DOUBLE,
    built_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, cap_name)
);
