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
