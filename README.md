# Options Pricing Engine

Sub-5ms European and American options pricing with full analytical Greeks, written in Rust.

<!-- TODO: add demo GIF here -->

## Scope (v1)

| Decision | Value |
|----------|-------|
| **Latency target** | p99 < 5ms for a single option price + full Greek set, measured at the API boundary |
| **Instrument coverage** | European vanilla (calls + puts) first; American second; no exotics in v1 |
| **Precision target** | Price accurate to 1e-10 vs. Python reference for vanilla inputs; error bounds documented for edge cases (near-zero time, near-zero vol, extreme moneyness) |
| **Benchmark hardware** | See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |

## Latency

| Benchmark | p50 | p99 |
|-----------|-----|-----|
| TODO (run `cargo bench`) | | |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full numbers and methodology.

## Numerical notes

See [docs/NUMERICS.md](docs/NUMERICS.md) for a documented inventory of numerical failure modes found, root-caused, and fixed — including catastrophic cancellation near expiry, IV solver divergence for deep OTM options, and binomial tree oscillation.

## Structure

```
core/              Rust crate: pricing math, no I/O, no heap allocation on hot path
service/           Async WebSocket service wrapping core
bindings/          PyO3 Python extension for prototyping
python_reference/  Correctness oracle; all core/ outputs validated against this
docs/              NUMERICS.md, BENCHMARKS.md, ARCHITECTURE.md
```

## Build

```bash
cargo build --release
cargo test --workspace
cargo bench
```

## Python reference

```bash
pip install -r python_reference/requirements.txt
pytest python_reference/tests/
```
