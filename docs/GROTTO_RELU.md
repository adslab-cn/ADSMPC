# Grotto ReLU implementation notes

This implementation evaluates the exact two-piece polynomial

```text
ReLU(x) = 0  when x < 0
          x  when x >= 0
```

using the Grotto construction from Storrier et al. It deliberately remains a
separate backend from `GTDCFReLU`, so both protocols can be tested against the
same cleartext reference.

## Protocol mapping

1. The dealer samples a point `i` and generates the repository's existing
   128-bit-leaf `DPFETKeyPack` pair for the one-hot vector `e_i`.
2. The online parties open `delta = x - i`. In this repository, evaluators
   hold the public masked value `x + r`, so the dealer shares `r + i`; the
   reconstruction is still exactly `x - i`.
3. The signed-domain ReLU pieces are `[0, 2^(n-1))` and
   `[2^(n-1), 2^n)`. Their endpoints are rotated left by `delta`.
4. Two DPF prefix-parity queries select the cyclic non-negative segment. Each
   query performs one half-PRG traversal per non-leaf level; at 64 bits this
   is 57 traversals per endpoint, or 114 for ReLU.
5. XOR shares are lifted using Equation (3) of Grotto: party 0 maps a set bit
   to `+1`, while party 1 maps it to `-1`. This produces a signed slope
   `a = +/-[x >= 0]`.
6. From the two segment-share bits, the parties form additive shares
   `(U_0, -U_1)` of the matching correction sign `u in {-1, +1}`.
7. A one-round ternary Beaver/ABY2-style multiplication computes
   `u * a * x = ReLU(x)`. The dealer supplies additive shares of random
   `U, A, X` and the correlated terms `AX, UX, UA, UAX`.

All arithmetic after the XOR-share lift is in `Z_(2^n)`. Thus `-1` is encoded
as `2^n - 1`, and two's-complement signs require no host-language signed
arithmetic.

## Repository adaptations

These adaptations preserve the evaluated function and the semi-honest
two-online-party-plus-dealer security model, but they are not literal API or
wire-format reproductions of the authors' implementation:

- The repository exposes masked public values (`x + r`) between nonlinear
  layers, whereas the paper describes additive input and output shares. The
  key contains shares of `i` and `r + i` to convert between representations
  without online dealer participation.
- The paper's online result remains additively shared. This repository's next
  layer expects both evaluators to hold the same masked output, so a final
  reconstruction of `ReLU(x) + r_out` is required. It adds one framework-level
  online round. The Grotto computation itself consists of opening `x-i` and
  one ABY2-style multiplication round.
- The ternary product is the zero-intercept specialization of Appendix E.1.2.
  For ReLU, `a0=0`, so the general linear-polynomial mask `A0` and its opening
  can be omitted. The remaining `U`, `A1`, `X`, `UX`, `UA1`, `A1X`, and
  `UA1X` values are exactly the direct expansion used by this implementation.
- `evalDPFET_LT` predates this work and returns an XOR share of
  `query < alpha`. The new helpers explicitly convert that convention to
  half-open prefix and cyclic-segment parity. The underlying AES expansion,
  correction words, 128-bit leaves, and DPF keys are reused unchanged.
- The high-level protocol is chunked at 10,000 elements to bound evaluator
  key memory. Each chunk independently performs the three online
  reconstructions, so inputs larger than one chunk use three rounds per chunk.

## Code layout

- `api.cpp` contains the complete dealer/server/client protocol: key
  distribution, online synchronization, the three reconstructions, output
  handling, chunking, and statistics.
- `grotto.cpp` contains only communication-free Grotto building blocks: key
  generation, the DPF-based local share evaluation, and key cleanup.
- `mult.cpp` provides the generic one-round ternary multiplication used for
  `u*a*x`; Grotto only prepares and opens its three operands.
- `tests/bench/Grotto-ReLU.cpp` only prepares deterministic masked inputs,
  calls `GrottoReLU`, and checks the revealed result on the client.

## Validation

- 64-bit three-party test with 10,000 deterministic positive and negative
  inputs plus `0`, `INT64_MAX`, `INT64_MIN`, and `-1`; expected accuracy is
  `10000/10000`.
- End-to-end BPGNN execution with `--relu=grotto`.

Generate dealer keys, then start the two online parties in separate processes:

```bash
cd build-vscode
./Grotto-ReLU 1
./Grotto-ReLU 2
./Grotto-ReLU 3 127.0.0.1
```

The VS Code compound launch does the same sequence automatically. The complete
BPGNN example remains available with:

```bash
./run-local.sh --relu=grotto
```
