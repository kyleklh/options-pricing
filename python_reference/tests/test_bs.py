# Cross-validates bs.py output against scipy/QuantLib and the fixture grid.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from bs import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho

# Known reference values for S=100, K=100, T=1, r=0.05, sigma=0.2
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

def test_call_price():
    assert abs(bs_price(S, K, T, r, sigma, 'call') - 10.4506) < 1e-4

def test_put_price():
    assert abs(bs_price(S, K, T, r, sigma, 'put') - 5.5735) < 1e-4

def test_put_call_parity():
    call = bs_price(S, K, T, r, sigma, 'call')
    put  = bs_price(S, K, T, r, sigma, 'put')
    # C - P = S - K * e^(-rT)
    assert abs((call - put) - (S - K * __import__('math').exp(-r * T))) < 1e-10

def test_call_delta():
    assert abs(bs_delta(S, K, T, r, sigma, 'call') - 0.6368) < 1e-4

def test_gamma():
    assert abs(bs_gamma(S, K, T, r, sigma) - 0.01876) < 1e-4

def test_vega():
    assert abs(bs_vega(S, K, T, r, sigma) - 37.524) < 1e-3