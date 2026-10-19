-- Option-data contract: the minimum tables and columns the dashboard reads.
-- Plain PostgreSQL; idempotent (safe to run on every start). A provider-specific
-- database may hold more columns; the dashboard ignores the ones it doesn't know.
-- Meaning, units and sign conventions: docs/DATA_CONTRACT.md.

-- One row per contract per collection day.
CREATE TABLE IF NOT EXISTS option_chain_data (
    collection_date  DATE NOT NULL,
    symbol           VARCHAR NOT NULL,
    expiration_date  DATE NOT NULL,
    strike           DOUBLE PRECISION NOT NULL,
    option_type      VARCHAR NOT NULL CHECK (option_type IN ('call', 'put')),
    price            DOUBLE PRECISION,
    day_open_price   DOUBLE PRECISION,
    day_high_price   DOUBLE PRECISION,
    day_low_price    DOUBLE PRECISION,
    oi               INTEGER,
    day_volume       INTEGER,
    volatility       DOUBLE PRECISION,
    delta            DOUBLE PRECISION,
    gamma            DOUBLE PRECISION,
    theta            DOUBLE PRECISION,
    vega             DOUBLE PRECISION,
    vanna            DOUBLE PRECISION,
    charm            DOUBLE PRECISION,
    vomma            DOUBLE PRECISION,
    volm_buy         INTEGER,
    volm_sell        INTEGER,
    gxoi             DOUBLE PRECISION,
    dxoi             DOUBLE PRECISION,
    vxoi             DOUBLE PRECISION,
    vannaxoi         DOUBLE PRECISION,
    charmxoi         DOUBLE PRECISION,
    PRIMARY KEY (collection_date, symbol, expiration_date, strike, option_type)
);
-- Per-symbol listings, expirations, OI by strike and contract history. The
-- primary key (leading collection_date) serves the latest-day queries.
CREATE INDEX IF NOT EXISTS idx_option_chain_data_symbol_expiration
    ON option_chain_data (symbol, expiration_date, collection_date);

-- One row per underlying per collection day: aggregates of that day's chain.
CREATE TABLE IF NOT EXISTS underlying_data (
    symbol            VARCHAR NOT NULL,
    collection_date   DATE NOT NULL,
    volatility        DOUBLE PRECISION,
    front_volatility  DOUBLE PRECISION,
    back_volatility   DOUBLE PRECISION,
    call_volume       INTEGER,
    put_volume        INTEGER,
    option_volume     INTEGER,
    put_call_ratio    DOUBLE PRECISION,
    flownet           DOUBLE PRECISION,
    vflownet          DOUBLE PRECISION,
    flowratio         DOUBLE PRECISION,
    vflowratio        DOUBLE PRECISION,
    volm_call_ratio   DOUBLE PRECISION,
    volm_put_ratio    DOUBLE PRECISION,
    deltas            DOUBLE PRECISION,
    gammas            DOUBLE PRECISION,
    dxoi              DOUBLE PRECISION,
    gxoi              DOUBLE PRECISION,
    vxoi              DOUBLE PRECISION,
    vannaxoi          DOUBLE PRECISION,
    charmxoi          DOUBLE PRECISION,
    txoi              DOUBLE PRECISION,
    PRIMARY KEY (symbol, collection_date)
);
-- "Latest day" scans filter on the date across all symbols.
CREATE INDEX IF NOT EXISTS idx_underlying_data_date ON underlying_data (collection_date);

-- One row per underlying per day: trailing 30-session mean of each numeric
-- underlying_data column, over the 30 sessions before `date` (not including it).
CREATE TABLE IF NOT EXISTS underlying_data_avg (
    symbol                VARCHAR NOT NULL,
    date                  DATE NOT NULL,
    volatility_avg        DOUBLE PRECISION,
    front_volatility_avg  DOUBLE PRECISION,
    back_volatility_avg   DOUBLE PRECISION,
    call_volume_avg       DOUBLE PRECISION,
    put_volume_avg        DOUBLE PRECISION,
    option_volume_avg     DOUBLE PRECISION,
    put_call_ratio_avg    DOUBLE PRECISION,
    flownet_avg           DOUBLE PRECISION,
    vflownet_avg          DOUBLE PRECISION,
    flowratio_avg         DOUBLE PRECISION,
    vflowratio_avg        DOUBLE PRECISION,
    volm_call_ratio_avg   DOUBLE PRECISION,
    volm_put_ratio_avg    DOUBLE PRECISION,
    deltas_avg            DOUBLE PRECISION,
    gammas_avg            DOUBLE PRECISION,
    dxoi_avg              DOUBLE PRECISION,
    gxoi_avg              DOUBLE PRECISION,
    vxoi_avg              DOUBLE PRECISION,
    vannaxoi_avg          DOUBLE PRECISION,
    charmxoi_avg          DOUBLE PRECISION,
    txoi_avg              DOUBLE PRECISION,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_underlying_data_avg_date ON underlying_data_avg (date);
