# Reference Black-Scholes implementation. Correctness over speed.
# Used as the oracle for validating every other implementation.

import math
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, flag):
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if flag == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    
def bs_delta(S, K, T, r, sigma, flag):
    d1 = _d1_d2(S, K, T, r, sigma)[0]
    if flag == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
    
def bs_gamma(S, K, T, r, sigma):
    d1 = _d1_d2(S, K, T, r, sigma)[0]
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    d1 = _d1_d2(S, K, T, r, sigma)[0]
    return S * norm.pdf(d1) * math.sqrt(T)

def bs_theta(S, K, T, r, sigma, flag):
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if flag == 'call':
        return (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2))
    
def bs_rho(S, K, T, r, sigma, flag):
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if flag == 'call':
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * norm.cdf(-d2)


def _d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return d1, d2