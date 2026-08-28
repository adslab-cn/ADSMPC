# CryptGNN-style two-party GCN benchmark

This port uses two online computing parties (`2` and `3`) and one offline
dealer (`1`). The dealer replaces CryptGNN's client-side preprocessing and
provides graph-index shares, AES-PRF noise seeds, mask-conversion material, and
the Beaver material consumed by the framework's `MatMul2D` implementation.

## Implemented online path

Each GCN layer executes:

1. dealer-preprocessed Beaver matrix multiplication (`MatMul2D`);
2. CryptGNN Algorithm 6 style batched message passing (`CryptMPL2P1`);
3. the CrypTen-compatible activation/Softmax adapters already provided by the
   framework.

`CryptMPL2P1` divides the edge list into 20 batches. The first source and
destination indices of each batch are additively shared, while the remaining
indices use CryptGNN's relative-index representation. All batch matrices are
concatenated so secure read uses one online exchange and secure write uses one
online exchange. AES-PRF masks hide transferred feature matrices. A final
batched reconstruction converts the arithmetic output shares back to the
framework's masked-value representation.

## Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) CryptGNN_Cora CryptGNN_Citeseer CryptGNN_Pubmed
```

## Run one dataset

Run the dealer first and wait for it to finish:

```bash
./CryptGNN_Cora 1
```

Then start the two online parties in separate terminals:

```bash
./CryptGNN_Cora 2
./CryptGNN_Cora 3 127.0.0.1
```

Replace `CryptGNN_Cora` with `CryptGNN_Citeseer` or `CryptGNN_Pubmed`.
The client process prints the end-to-end metrics and the first secure MPL
metrics. The First MPL accounting boundary includes the layer-1 `MatMul2D`
feature transformation followed by layer-1 `CryptMPL2P1`; the subsequent
ReLU is excluded. Time, key-read time, and online communication all use this
same boundary:

```text
Scope             Dataset         Online(ms)         Online(s)      OnlineComm(MB)
End-to-end        Cora               ...                ...                ...
First MPL         Cora               ...                ...                ...
```

For both rows, `Online(ms)` excludes dealer key-file reading time.
`OnlineComm(MB)` contains
only bytes sent and received between the two computing parties; graph sharing,
PRF seeds, conversion masks, and Beaver/FSS keys are offline traffic.

## Dataset/model dimensions

- Cora: `2708 x 1433`, 5429 edges, `1433 -> 64 -> 7`.
- Citeseer: `3327 x 3703`, 4732 edges, `3703 -> 64 -> 6`.
- Pubmed: `19717 x 500`, 44338 edges, `500 -> 64 -> 3`.

The test uses deterministic synthetic feature/model values and a deterministic
sparse edge list with the real dataset dimensions. Substitute the dataset
loader if accuracy evaluation on the original graph/model is required; online
cost dimensions remain unchanged.
