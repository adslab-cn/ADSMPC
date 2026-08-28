# Simplified dual-FSS authenticated comparison

This experiment wraps the framework's original optimized DCF, DPFET and GTDCF
implementations; it no longer contains a substitute `PlainArithmeticFSS` tree.
For DCF it implements the deliberately simple construction

```
value keys <- FSS.Gen(alpha, beta)
tag keys   <- FSS.Gen(alpha, Delta * beta)
```

and evaluates the two independent native keys before a deferred random-linear
MAC check. It is an experimental SHARK-style adapter, not a claim to reproduce
the complete SHARK proof or enlarged-ring construction.

All three rows end with authenticated arithmetic shares. GROTTO uses two
independent native DPFET evaluations. Dealer-preprocessed B2A converts the first
Boolean result with multiplier 1 and the second with multiplier Delta, costing
four opened masked bits per comparison.

GTDCF follows exactly the same dual-FSS rule: one independent native GTDCF key
uses payload `(1,0)` for the comparison value, and a second independent native
GTDCF key uses `(Delta,0)` for the MAC. Online evaluation therefore invokes
`evalGTDCF` twice per party. Although native GTDCF supports a two-lane payload,
this benchmark intentionally does not place `(1,Delta)` in one key, because
that fused optimization would not match the two-independent-FSS baseline.

Build and run:

```bash
cmake -S . -B build
cmake --build build --target dual_fss_malicious_comparison -j
./build/dual_fss_malicious_comparison 40 100000 1000 3 \
  | tee dual_fss_malicious.csv
```

Arguments are `stat_bits`, `largest_batch`, `chunk_size`, and odd
`repetitions`. `stat_bits` is retained for command-line compatibility; this
simplified experiment authenticates in the native ell-bit ring rather than the
enlarged ell+s ring. The program first tests boundary correctness and tamper
rejection, then reports the median of three runs. The GROTTO row is explicitly
reported as `Dual-FSS-GROTTO+B2A`, and `B2A(ms)` isolates its conversion cost;
that cost is already included in `Eval(ms)`. `Online(ms)` is
`Eval(ms)+BatchCheck(ms)`. Key generation is offline and listed separately.

Use the `Online(ms)` values to fill the following rows:

```latex
\multirow{2}{*}{Dual-FSS DCF (SHARK-style)}
& \multirow{2}{*}{Arithmetic}
& 32 & -- & -- & -- \\
&& 64 & -- & -- & -- \\
\midrule
\multirow{2}{*}{Dual-FSS GROTTO+B2A (LightShark-style)}
& \multirow{2}{*}{Arithmetic}
& 32 & -- & -- & -- \\
&& 64 & -- & -- & -- \\
\midrule
\multirow{2}{*}{\textbf{Dual-FSS GTDCF}}
& \multirow{2}{*}{Arithmetic}
& 32 & -- & -- & -- \\
&& 64 & -- & -- & -- \\
```
