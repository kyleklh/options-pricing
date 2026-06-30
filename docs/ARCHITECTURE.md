# Architecture

## Layer diagram

```
python_reference/   ← correctness oracle, not on the hot path
        ↓ fixture grid (JSON)
core/               ← pure math, no I/O, no allocation on hot path
  numerics          ← normal CDF/PDF, stable helpers
  bs                ← Black-Scholes price + Greeks
  iv                ← implied vol solver
  tree              ← American option tree
  pde               ← Crank-Nicolson FD solver
        ↓
bindings/           ← PyO3 Python extension for prototyping
service/            ← async runtime, WebSocket/gRPC, market data feed
```

## Key decisions

TODO
