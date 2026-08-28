[
# BPGNN Artifact

This repository contains the code and scripts for reproducing the main experimental results of **BPGNN**, a privacy-preserving framework for inductive GNN inference in the standard two-server setting.

## Overview

The artifact includes:

* the implementation of the BPGNN framework,
* setup scripts for building the required environment,
* executables for reproducing the main experiments in the paper.

The framework is designed for a two-party setting and can be tested locally by simulating the client and the two servers on a single machine.

---

## Environment Setup

### Platform

The framework is developed and tested on:

* **Operating System:** Ubuntu 22.04 LTS

### Build Dependencies

The project uses **CMake 3.17 or higher**.

To simplify installation, we provide an automated setup script that checks the local environment and installs or builds the required dependencies when needed.

---

## Automated Setup

From the repository root, run:

```bash
sudo ./1-base.sh
```

This script configures the environment automatically.

---

## Compilation

After the environment is ready, compile the framework with standard CMake commands:

```bash
mkdir build && cd build
cmake ..
make -j
```

All generated binaries, including test and experiment executables, will be placed in the `build/` directory.

---

## Reproducing Experimental Results

The completed BPGNN path and its implementation/security scope are documented
in [`BPGNN_IMPLEMENTATION.md`](BPGNN_IMPLEMENTATION.md). The end-to-end test now
prints measured runtime, online communication, and Dealer preprocessing
communication separately for every GCN stage. Dataset-specific executables
`BPGNN_Cora`, `BPGNN_Citeseer`, and `BPGNN_Pubmed` each use one static round
followed by five dynamic rounds and report their aggregate comparison. Splitting
the experiments and streaming Dealer keys avoids loading the combined
preprocessing files for all datasets into memory.

The build also provides `softmax_comparison`, a 2+1 benchmark for BPGNN,
the framework's SIGMA-compatible Softmax path, BumbleBee's clipped limit
approximation, and a CrypTen-style limit-approximation baseline. Protocol
mapping, accuracy metrics, and run commands are in `BPGNN_IMPLEMENTATION.md`.

`relu_comparison` compares the OblivGNN, CrypTen schedule, GROTTO spline,
SIGMA/SlothRelu, and BPGNN GTDCF ReLU paths on each dataset's `nodes x 64`
hidden tensor.

Benchmark timing is split into total wall time, preprocessing-key read time,
and pure online time (`total - key read`) for the Softmax/ReLU comparisons and
for every stage, round, and summary row of the end-to-end BPGNN experiments.
The BPGNN Softmax clamp batches its two GTDCF threshold branches into one
reconstruction, uses three exponent-restoration squares, and applies one-round
probabilistic truncation to its scaling-and-squaring path.

We provide executables corresponding to the key results reported in the paper.

### Local Two-Party Execution

The framework is designed to run over a network. For local testing, the client and the two servers can be simulated on the same machine.

* Set the IP address in the source code to:

```text
127.0.0.1
```

### Network Setting Used in Our Experiments

Our reported experiments were conducted in a simulated LAN environment with:

* **Bandwidth:** 1 Gbps
* **Latency:** 0.15 ms

---

## Running an Experiment

The dealer (party 1) must finish first because it generates `server.dat` and
`client.dat`. After that, parties 2 and 3 run concurrently. For a local run,
the repository provides a helper that performs this sequence automatically:

```bash
./run-local.sh
```

The default uses the existing GTDCF ReLU. To run the Grotto
DPF/prefix-parity ReLU backend locally, use:

```bash
./run-local.sh --relu=grotto
```

The equivalent manual commands are:

```bash
./BPGNN 1 --relu=grotto
./BPGNN 2 --relu=grotto
./BPGNN 3 127.0.0.1 --relu=grotto
```

Protocol details, validation coverage, and adaptations required by this
repository's masked-value interface are documented in
[`docs/GROTTO_RELU.md`](docs/GROTTO_RELU.md).

The commands below show the equivalent manual sequence.

To run an experiment, start three separate processes corresponding to:

* the **client**,
* **server 0**,
* **server 1**.

Below is the example for running the end-to-end BPGNN experiment.

### Terminal 1: Client

```bash
cd build/
./BPGNN 1
```

### Terminal 2: Server 0

```bash
cd build/
./BPGNN 2
```

### Terminal 3: Server 1

```bash
cd build/
./BPGNN 3
```

---

## Mapping from Paper Results to Source Files

The following source files generate the main results in the paper.

### Table: Microbenchmark of secure MPL under static and dynamic settings

* **Source file:** `BPGNN.cpp`

### Table: Online runtime of secure comparison paths

* **Source file:** `comparison_primitive.cpp`

### Table: Secure ReLU comparison

* **Source file:** `GTDCF-ReLU.cpp`

### Table: End-to-end performance under static and dynamic settings

* **Source file:** `BPGNN.cpp`

### Table: End-to-end breakdown

* **Source file:** `BPGNN.cpp`

---

## Notes

* The artifact is intended to provide a reproducible environment for verifying the implementation and evaluating the main experimental claims of the paper.
* All experiments are run from the compiled binaries in `build/`.
* For local testing, make sure that all three processes are launched in the correct order and that the configured IP address matches the local setup.

---

## Reproducibility Statement

We provide this artifact to support verification of the reported results and to facilitate future research on privacy-preserving graph machine learning.


](https://anonymous.4open.science/r/SMPC-D510)
