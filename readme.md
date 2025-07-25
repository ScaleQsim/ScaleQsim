# ScaleQsim
**ScaleQsim: A Highly Scalable Quantum Circuit Simulation Framework for Exascale HPC Systems**  
(Submitted to SIGMETRICS'26)

## Introduction
ScaleQsim is a distributed full-state quantum circuit simulator designed for large-scale HPC systems.  
To efficiently support multi-Node/GPUs, ScaleQsim introduces:
- A two-phase state vector partitioning scheme (inter-node and intra-node),
- A target index generation and mapping system for efficient memory access,
- An adaptive kernel tuning mechanism that dynamically adjusts workloads.

The framework is built upon Google’s Qsim, but redesigned to support multi-node, multi-GPU distributed simulation with low synchronization overhead and scalable memory layout.  
Our evaluation demonstrates significant speedups (up to 6.15×) over existing state-of-the-art simulators like cuStateVec and Qsim.

## Key Features
- Scalable to hundreds of GPUs with efficient memory distribution and task mapping.
- Supports bitmask-based target index enumeration for precise gate application.
- Provides adaptive CUDA kernel scheduling to balance performance and memory usage.
- Integrates MPI + CUDA P2P Communication.

## Modified Components
ScaleQsim modifies and extends the following core Qsim modules:
- `simulator_cuda.h`  
- `vectorspace_cuda.h`  
- `simulator_cuda_kernel.h`
- `pybind_cuda.cpp`
  
