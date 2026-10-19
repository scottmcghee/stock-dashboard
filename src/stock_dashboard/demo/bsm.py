"""Black-Scholes-Merton prices and greeks, vectorized over numpy arrays.

Inputs: spot S, strike K, time to expiry T in years, volatility sigma (annual,
as a fraction), risk-free rate r and continuous dividend yield q.

Output units (the ones the option-data contract uses):
- delta: per $1 of underlying; gamma: delta per $1.
- theta, charm: per calendar day (annual rate / 365).
- vega, vanna: per volatility point (sigma change of 0.01).
- vomma: change in vega (per point) per volatility point.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]

DAYS_PER_YEAR: Final = 365.0
VOL_POINT: Final = 0.01
_INV_SQRT_2PI: Final = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: FloatArray) -> FloatArray:
    return np.asarray(ndtr(x), dtype=np.float64)


def _norm_pdf(x: FloatArray) -> FloatArray:
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * x * x), dtype=np.float64)


@dataclass(frozen=True)
class Greeks:
    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    theta: FloatArray
    vega: FloatArray
    vanna: FloatArray
    charm: FloatArray
    vomma: FloatArray


def bsm_price(
    spot: FloatArray,
    strike: FloatArray,
    t: FloatArray,
    sigma: FloatArray,
    is_call: npt.NDArray[np.bool_],
    r: float,
    q: float,
) -> FloatArray:
    """Option price only (cheaper than `bsm_greeks` when greeks aren't needed)."""
    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q = np.exp(-q * t)
    disc_r = np.exp(-r * t)
    call = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    put = strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)
    return np.where(is_call, call, put)


def bsm_greeks(
    spot: FloatArray,
    strike: FloatArray,
    t: FloatArray,
    sigma: FloatArray,
    is_call: npt.NDArray[np.bool_],
    r: float,
    q: float,
) -> Greeks:
    """Price and greeks for each element; arrays broadcast against each other."""
    sqrt_t = np.sqrt(t)
    sig_sqrt_t = sigma * sqrt_t
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    disc_q = np.exp(-q * t)
    disc_r = np.exp(-r * t)
    pdf_d1 = _norm_pdf(d1)
    cdf_d1, cdf_d2 = _norm_cdf(d1), _norm_cdf(d2)
    cdf_md1, cdf_md2 = _norm_cdf(-d1), _norm_cdf(-d2)

    call = spot * disc_q * cdf_d1 - strike * disc_r * cdf_d2
    put = strike * disc_r * cdf_md2 - spot * disc_q * cdf_md1
    delta = np.where(is_call, disc_q * cdf_d1, -disc_q * cdf_md1)
    gamma = disc_q * pdf_d1 / (spot * sig_sqrt_t)

    decay = -spot * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
    theta_call = decay - r * strike * disc_r * cdf_d2 + q * spot * disc_q * cdf_d1
    theta_put = decay + r * strike * disc_r * cdf_md2 - q * spot * disc_q * cdf_md1

    vega = spot * disc_q * pdf_d1 * sqrt_t
    vanna = -disc_q * pdf_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma

    # Charm = d(delta)/d(calendar time), i.e. -d(delta)/dT.
    charm_common = disc_q * pdf_d1 * (2.0 * (r - q) * t - d2 * sig_sqrt_t) / (2.0 * t * sig_sqrt_t)
    charm_call = q * disc_q * cdf_d1 - charm_common
    charm_put = -q * disc_q * cdf_md1 - charm_common

    return Greeks(
        price=np.where(is_call, call, put),
        delta=delta,
        gamma=gamma,
        theta=np.where(is_call, theta_call, theta_put) / DAYS_PER_YEAR,
        vega=vega * VOL_POINT,
        vanna=vanna * VOL_POINT,
        charm=np.where(is_call, charm_call, charm_put) / DAYS_PER_YEAR,
        vomma=vomma * VOL_POINT * VOL_POINT,
    )
