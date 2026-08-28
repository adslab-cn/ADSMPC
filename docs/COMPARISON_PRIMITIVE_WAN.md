# WAN comparison-primitive table

`comparison_primitive` directly benchmarks the rows required by
`tab:GTDCF_l_compare` for `ell={32,64}` and
`B={1000,10000,100000}`.  GROTTO includes dealer-preprocessed B2A so that its
result can be consumed as an arithmetic share.

The WAN model is fixed to 400 Mbps bandwidth and 60 ms one-way latency.  DCF
and GTDCF evaluation are local FSS operations.  B2A opens one masked bit per
party, so GROTTO adds one 120 ms RTT and `2B` transmitted bits:

```
GROTTO-WAN = GROTTO-core + B2A-local + 120 ms + (2B / 400 Mbps)
```

Build and run the complete table:

```bash
cmake -S . -B build
cmake --build build --target comparison_primitive -j
./build/comparison_primitive 1000 3 100000 | tee comparison_wan.csv
```

Arguments are `chunk_size`, odd `repetitions`, and `largest_batch`.  The
default values are `1000 3 100000`.  The executable prints CSV measurements
followed by LaTeX rows that can be copied into the paper table.
