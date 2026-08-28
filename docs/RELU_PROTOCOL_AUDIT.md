# ReLU protocol audit

## GROTTO

`GrottoReLU` is the repository's dedicated GROTTO backend and does not call the
arithmetic-DCF `Relu` implementation. It evaluates the exact two-piece
degree-one spline, with coefficients `0` and `x`, using rotated DPF prefix
parity. XOR shares are lifted into signed arithmetic slope and correction
shares, then a one-round ternary masked multiplication produces `ReLU(x)`.
The merged implementation is documented in detail in `docs/GROTTO_RELU.md`.

## Other benchmark entries

- `SIGMAReLU` is `SlothDrelu` followed by the framework's generic `Select`.
- `GTDCFReLU` directly evaluates a two-component arithmetic GTDCF payload and
  reconstructs the masked arithmetic ReLU output in one online round.
- `OblivGNNReLU` uses the framework's two-round arithmetic DCF comparison and
  masked multiplication implementation. It is an implementation of that
  protocol shape, not imported OblivGNN source code.
- `CrypTenReLU` uses the same secure comparison/multiplication core plus explicit
  barriers to reproduce a nine-round schedule. It is a latency adapter and is
  not a source-level port of CrypTen's Python binary protocol. The benchmark now
  labels it accordingly.

All entries are checked by reconstructing the client output outside the measured
region and comparing every ring element against cleartext ReLU. A valid run must
report `Errors = 0` for every row.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target relu_comparison -j"$(nproc)"
cd build
bash ../tests/bench/run_relu_comparison.sh cora 127.0.0.1 ./relu_comparison
```

Repeat with `citeseer` and `pubmed`. If a previous process still owns the
framework port, stop that process before rerunning; do not start a second server
on the same port.
