"""Column registry for the option-data contract (mirrors schema/option_data.sql).

The generator builds frames with exactly these columns, in this order, and the
loader inserts them by name. Tests check this registry against the DDL.
"""

from typing import Final

DATE: Final = "date"
TEXT: Final = "varchar"
INT: Final = "integer"
FLOAT: Final = "double precision"

OPTION_CHAIN_COLUMNS: Final[dict[str, str]] = {
    "collection_date": DATE,
    "symbol": TEXT,
    "expiration_date": DATE,
    "strike": FLOAT,
    "option_type": TEXT,
    "price": FLOAT,
    "day_open_price": FLOAT,
    "day_high_price": FLOAT,
    "day_low_price": FLOAT,
    "oi": INT,
    "day_volume": INT,
    "volatility": FLOAT,
    "delta": FLOAT,
    "gamma": FLOAT,
    "theta": FLOAT,
    "vega": FLOAT,
    "vanna": FLOAT,
    "charm": FLOAT,
    "vomma": FLOAT,
    "volm_buy": INT,
    "volm_sell": INT,
    "gxoi": FLOAT,
    "dxoi": FLOAT,
    "vxoi": FLOAT,
    "vannaxoi": FLOAT,
    "charmxoi": FLOAT,
}

UNDERLYING_COLUMNS: Final[dict[str, str]] = {
    "symbol": TEXT,
    "collection_date": DATE,
    "volatility": FLOAT,
    "front_volatility": FLOAT,
    "back_volatility": FLOAT,
    "call_volume": INT,
    "put_volume": INT,
    "option_volume": INT,
    "put_call_ratio": FLOAT,
    "flownet": FLOAT,
    "vflownet": FLOAT,
    "flowratio": FLOAT,
    "vflowratio": FLOAT,
    "volm_call_ratio": FLOAT,
    "volm_put_ratio": FLOAT,
    "deltas": FLOAT,
    "gammas": FLOAT,
    "dxoi": FLOAT,
    "gxoi": FLOAT,
    "vxoi": FLOAT,
    "vannaxoi": FLOAT,
    "charmxoi": FLOAT,
    "txoi": FLOAT,
}

UNDERLYING_KEY_COLUMNS: Final = ("symbol", "collection_date")
UNDERLYING_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in UNDERLYING_COLUMNS if c not in UNDERLYING_KEY_COLUMNS
)

UNDERLYING_AVG_COLUMNS: Final[dict[str, str]] = {
    "symbol": TEXT,
    "date": DATE,
    **{f"{c}_avg": FLOAT for c in UNDERLYING_VALUE_COLUMNS},
}

TABLES: Final[dict[str, dict[str, str]]] = {
    "option_chain_data": OPTION_CHAIN_COLUMNS,
    "underlying_data": UNDERLYING_COLUMNS,
    "underlying_data_avg": UNDERLYING_AVG_COLUMNS,
}
