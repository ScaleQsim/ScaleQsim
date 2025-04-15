// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMULATOR_CUDA_H_
#define SIMULATOR_CUDA_H_
//kcj
#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //KCJ
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif



#include "bits.h"
#include "statespace_cuda.h"
#include "simulator_cuda_kernels.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

//kcj
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <stdexcept>
#include <bitset>
#include <random>
#include </opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include/mpi.h>
#include <chrono>
#include <unordered_map>



namespace qsim {
    
/**
 * Quantum circuit simulator with GPU vectorization.
 */

    
/* //Original
template <typename FP = float>
class SimulatorCUDA final {
 public:
  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using idx_type = uint64_t;
  using Complex = qsim::Complex<double>;

  // 멤버 변수 추가
  //NCCLContext nccl_context_;
  StateSpace state_space_;
  void* scratch_;
  uint64_t scratch_size_;
  mutable State state_; 

  SimulatorCUDA() 
        : state_space_(typename StateSpace::Parameter{512, 16}),  // 괄호로 초기화
          scratch_(nullptr),
          scratch_size_(0) {
      cudaMalloc(&d_ws, max_buf_size);
      printf("KCJ: cudaMalloc: &d_ws: %p, max_buf_size: %llu\n", &d_ws, max_buf_size);
    }


  ~SimulatorCUDA() {
    cudaFree(d_ws);
    if (scratch_ != nullptr) {
      cudaFree(scratch_);
    }
  }
  
  
    static constexpr unsigned max_buf_size = 8192 * sizeof(FP)
      + 128 * sizeof(idx_type) + 96 * sizeof(unsigned);
*/
    
template <typename FP = float>
class SimulatorCUDA final {
 public:
  using StateSpace = StateSpaceCUDA<FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using idx_type = uint64_t;
  using Complex = qsim::Complex<double>;
  std::vector<fp_type*> state_parts_;

  // 다중 GPU 멤버 변수
  std::vector<StateSpace> state_spaces_;    // 각 GPU의 상태 공간
  std::vector<ncclComm_t> nccl_comms_;      // NCCL communicator 리스트
  std::vector<char*> d_ws_list_;        // 각 GPU별 작업 공간
  //char* d_ws_list;
  void* scratch_;
  uint64_t scratch_size_;
  mutable State state_; 

  static double total_index_calc_time;
  static double total_apply_gate_time;
  static double total_cpu_time_H;
  static float total_gpu_time_H;
  static double total_cpu_time;
  static float total_gpu_time;
    static double total_cpu_time_copy_h;   
    static float  total_gpu_time_copy_h; 
    static double total_cpu_time_copy_l;   
    static float total_gpu_time_copy_l;
  
SimulatorCUDA(const std::vector<StateSpace>& state_spaces,
              const std::vector<ncclComm_t>& nccl_comms)
    : state_spaces_(state_spaces),
      nccl_comms_(nccl_comms),
      scratch_size_(0) {

    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Barrier(MPI_COMM_WORLD);
        
    size_t num_gpus = state_spaces_.size();


    unsigned num_qubits = 38;

    size_t total_size = MinSize(num_qubits);

    // KCJ: Multi-node.
    size_t size_per_node = 0;
    size_t size_per_gpu = 0;
    size_t Real_size_per_gpu = 0;
    
    if (total_nodes > 1) {
        size_per_node = total_size / total_nodes;  
        size_per_gpu = size_per_node / num_gpus;   
        Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

    } else {
        size_per_node = total_size;                
        size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;
    }

        
    d_ws_list_.resize(num_gpus);

    cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
        printf("ERROR: Simulator 2 - CUDA error before kernel launch on GPU %lu: %s\n", 0, cudaGetErrorString(err_before));
    }

        
    for (size_t i = 0; i < num_gpus; ++i) {
        auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
        VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::CreateMultiGPU(num_qubits, num_gpus); 

        cudaSetDevice(i);
        fp_type* gpu_state_ptr = multi_gpu_pointers[i];
        state_parts_.push_back(gpu_state_ptr);
         
        cudaDeviceSynchronize();
     }
        
    AllocateWorkspace(num_gpus);
    
    SetAllZeros();
    
    InitializeState();

    }
    ~SimulatorCUDA() {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  
    
        printf("DEBUG: Node %d - Starting SimulatorCUDA destructor.\n", world_rank);
    
        // Free workspace memory for each GPU on the current node
        for (size_t i = 0; i < d_ws_list_.size(); ++i) {
            cudaSetDevice(i); // Set the appropriate GPU for the current node only
            if (d_ws_list_[i] != nullptr) {
                cudaError_t err = cudaFree(d_ws_list_[i]);
                if (err != cudaSuccess) {
                    printf("ERROR: Node %d - cudaFree failed for d_ws_list_[%lu]: %s\n", world_rank, i, cudaGetErrorString(err));
                } else {
                    printf("DEBUG: Node %d - Successfully freed memory for d_ws_list_[%lu].\n",  world_rank, i);
                }
            }
        }
    
        // Free scratch memory if allocated (only for the current node)
        if (scratch_ != nullptr) {
            cudaError_t err = cudaFree(scratch_);
            if (err != cudaSuccess) {
                printf("ERROR: Node %d - cudaFree failed for scratch_: %s\n",
                       world_rank, cudaGetErrorString(err));
            } else {
                printf("DEBUG: Node %d - Successfully freed scratch_ memory.\n", world_rank);
            }
        }
    
        // Synchronization to ensure all nodes have freed their memory before exiting
        MPI_Barrier(MPI_COMM_WORLD);
    }

  

  static constexpr unsigned max_buf_size = 8192 * sizeof(FP)
      + 128 * sizeof(idx_type) + 96 * sizeof(unsigned);
      
  char* d_ws;
  char h_ws0[max_buf_size];
  char* h_ws = (char*) h_ws0;



void AllocateWorkspace(size_t num_gpus) {
    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    

    for (unsigned i = 0; i < num_gpus; ++i) {
        cudaError_t status = cudaSetDevice(i);
        if (status != cudaSuccess) {
            printf("ERROR: Failed to set device %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }

        size_t free_mem = 0, total_mem = 0;
        status = cudaMemGetInfo(&free_mem, &total_mem);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMemGetInfo failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }

        status = cudaMalloc(&d_ws_list_[i], max_buf_size);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMalloc failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            d_ws_list_[i] = nullptr;
            continue;
        }


        
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
           printf("ERROR: cudaDeviceSynchronize failed on GPU %u: %s\n", i, cudaGetErrorString(status));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

}


void InitializeState() {



  //  printf("KCJ: Initializing multi-GPU state |0⟩\n");

    /*
    for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        size_t global_index = gpu_id * size_per_gpu;
        printf("DEBUG: Initializing GPU %lu, Global index %lu\n", gpu_id, global_index);
        
        SetAmpl(global_index, 1.0, 0.0);
    }
    */
   // SetAmpl(0, 1.0, 0.0);

  //  printf("KCJ: Initialization complete.\n");
  //  MPI_Barrier(MPI_COMM_WORLD);

}








static uint64_t MinSize(unsigned num_qubits) {
    num_qubits = 38;
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    return result;
      
};


// KCJ Multiver.
void SetAllZeros() {
    int node_id, total_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus = multi_gpu_pointers.size();
    unsigned num_qubits = 38;
    size_t total_size = MinSize(num_qubits);

    size_t size_per_node = 0;
    size_t size_per_gpu = 0;
    size_t Real_size_per_gpu = 0;
    
    if (total_nodes > 1) {
        size_per_node = total_size / total_nodes;  
        size_per_gpu = size_per_node / num_gpus;   
        Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

    } else {
        size_per_node = total_size;              
        size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;
    }


    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaMemset(multi_gpu_pointers[i], 0, Real_size_per_gpu);
    }

    if(node_id == 0){
    fp_type one[1] = {1};
    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(0);
        cudaMemcpy(multi_gpu_pointers[0], one, sizeof(fp_type), cudaMemcpyHostToDevice);
 
       }
    }
}
 
/*
 * CalculateAffectedIndices:
 * Computes affected global indices for a given multi-qubit task using bitmask enumeration.
 * Filters indices based on node and GPU boundaries defined by static partitioning.
 * Only indices assigned to each GPU are returned to enable consistent and independent execution.
*/



std::vector<size_t> CalculateAffectedIndices(
    size_t num_qubits, const std::vector<unsigned>& qs,
    size_t num_nodes, size_t num_gpus_per_node, size_t size_per_gpu) const {


    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    
    
    size_t total_gpus = num_nodes * num_gpus_per_node;
    size_t size_per_node = size_per_gpu * num_gpus_per_node;
    size_t num_indices = 1ULL << qs.size();

    std::vector<size_t> indices_global;
    indices_global.reserve(num_indices);

    std::vector<size_t> xs(qs.size());
    xs[0] = size_t{1} << (qs[0] + 1);
    for (size_t i = 1; i < qs.size(); ++i) {
        xs[i] = size_t{1} << (qs[i] + 1);
    }


    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        size_t node_offset = node_id * size_per_node;

        for (size_t i = 0; i < num_indices; ++i) {
            size_t index = 0;
            for (size_t j = 0; j < qs.size(); ++j) {
                index += xs[j] * ((i >> j) & 1);  
            }
            index += node_offset;

            if (index < node_offset || index >= node_offset + size_per_node) {
                continue; 
            }


            size_t index_in_node = index % size_per_node;
            size_t gpu_id_local = index_in_node / size_per_gpu;
            size_t local_index = index_in_node % size_per_gpu;
            size_t global_gpu_id = node_id * num_gpus_per_node + gpu_id_local;

            if (global_gpu_id >= total_gpus) continue;

            indices_global.push_back(index);
        }
    }

    auto end_time = high_resolution_clock::now();  
    double elapsed_time = duration<double, std::milli>(end_time - start_time).count();  
    total_index_calc_time += elapsed_time;

    
    return indices_global;
}


/*
 * ApplyGate:
 * Executes a quantum gate by computing affected indices and distributing work across GPUs.
 * Node 0 generates global affected indices, which are broadcast to all nodes via MPI.
 * Each node filters relevant indices and assigns local tasks to GPUs based on data partitioning.
*/

void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now(); 

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t num_nodes = world_size;  
    
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();

    unsigned num_qubits = 38;
    size_t num_gpus_per_node = multi_gpu_pointers.size();
    size_t total_gpus = num_nodes * num_gpus_per_node;

    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = total_size / total_gpus;
    size_t size_per_node = size_per_gpu * num_gpus_per_node;
    size_t Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

   std::vector<size_t> affected_indices;

    if (world_rank == 0) {
        affected_indices = CalculateAffectedIndices(num_qubits, qs, num_nodes, num_gpus_per_node, Real_size_per_gpu);
    }
    
    size_t indices_size = affected_indices.size();
    MPI_Bcast(&indices_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        affected_indices.resize(indices_size);  
    }
    
    size_t total_bytes = indices_size * sizeof(uint64_t);
    static size_t total_transmitted_bytes = 0;
    total_transmitted_bytes += total_bytes;

    if (world_rank == 0) {
        printf("MPI_COMM N0 to All Node > Cumulative total: %zu bytes (%.4f MB)\n",  total_transmitted_bytes, total_transmitted_bytes / (1024.0 * 1024.0));
    }
    
    
    MPI_Bcast(affected_indices.data(), indices_size, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    size_t gpu_task_count[num_gpus_per_node] = {0};
    std::map<size_t, std::vector<size_t>> gpu_indices_map;

    for (size_t global_index : affected_indices) {
        size_t node_id = global_index / size_per_node;
        if (node_id != world_rank) continue;
        
        size_t index_in_node = global_index - (node_id * size_per_node);
        size_t gpu_id_local = index_in_node / Real_size_per_gpu;
        size_t local_index = index_in_node % Real_size_per_gpu;
        size_t global_gpu_id = node_id * num_gpus_per_node + gpu_id_local;    

        gpu_task_count[gpu_id_local]++;
        gpu_indices_map[gpu_id_local].push_back(local_index);

    }

    
    
            for (size_t gpu_id = 0; gpu_id < num_gpus_per_node; ++gpu_id) {
                if (gpu_task_count[gpu_id] == 0) continue;
    
    
                cudaSetDevice(gpu_id);
    
                for (const auto& local_index : gpu_indices_map[gpu_id]) {
                }
    
                if (qs.size() == 0) {
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state, gpu_id);
                } else if (qs[0] > 4) {
                    switch (qs.size()) {
                        case 1: const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state, gpu_id); break;
                        case 2: const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state, gpu_id); break;
                        case 3: const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state, gpu_id); break;
                        case 4: const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state, gpu_id); break;
                        case 5: const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state, gpu_id); break;
                        case 6: const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state, gpu_id); break;
                    }
                } else {
                    switch (qs.size()) {
                        case 1: const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state, gpu_id); break;
                        case 2: const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state, gpu_id); break;
                        case 3: const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state, gpu_id); break;
                        case 4: const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state, gpu_id); break;
                        case 5: const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state, gpu_id); break;
                        case 6: const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state, gpu_id); break;
                    }
                }
            }

    
    
    auto end_time = high_resolution_clock::now(); 
    double elapsed_time = duration<double, std::milli>(end_time - start_time).count();  

    total_apply_gate_time += elapsed_time;

}


  /**
   * Applies a controlled gate using CUDA instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
   
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    printf("KCJ: ApplyControlledGate\n");
    if (cqs.size() == 0) {
      //ApplyGate(qs, matrix, state);
      return;
    }

    if (cqs[0] < 5) {
      switch (qs.size()) {
      case 0:
        ApplyControlledGateL<0>(qs, cqs, cvals, matrix, state);
        break;
      case 1:
        ApplyControlledGateL<1>(qs, cqs, cvals, matrix, state);
        break;
      case 2:
        ApplyControlledGateL<2>(qs, cqs, cvals, matrix, state);
        break;
      case 3:
        ApplyControlledGateL<3>(qs, cqs, cvals, matrix, state);
        break;
      case 4:
        ApplyControlledGateL<4>(qs, cqs, cvals, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    } else {
      if (qs.size() == 0) {
        ApplyControlledGateHH<0>(qs, cqs, cvals, matrix, state);
      } else if (qs[0] > 4) {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      } else {
        switch (qs.size()) {
        case 1:
          ApplyControlledGateLH<1>(qs, cqs, cvals, matrix, state);
          break;
        case 2:
          ApplyControlledGateLH<2>(qs, cqs, cvals, matrix, state);
          break;
        case 3:
          ApplyControlledGateLH<3>(qs, cqs, cvals, matrix, state);
          break;
        case 4:
          ApplyControlledGateLH<4>(qs, cqs, cvals, matrix, state);
          break;
        default:
          // Not implemented.
          break;
        }
      }
    }
  }
  

  /**
   * Computes the expectation value of an operator using CUDA instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
    printf("KCJ: ExpectationValue\n");

    if (qs[0] > 4) {
      switch (qs.size()) {
      case 1:
        return ExpectationValueH<1>(qs, matrix, state);
      case 2:
        return ExpectationValueH<2>(qs, matrix, state);
      case 3:
        return ExpectationValueH<3>(qs, matrix, state);
      case 4:
        return ExpectationValueH<4>(qs, matrix, state);
      case 5:
        return ExpectationValueH<5>(qs, matrix, state);
      case 6:
        return ExpectationValueH<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
    } else {
      switch (qs.size()) {
      case 1:
        return ExpectationValueL<1>(qs, matrix, state);
      case 2:
        return ExpectationValueL<2>(qs, matrix, state);
      case 3:
        return ExpectationValueL<3>(qs, matrix, state);
      case 4:
        return ExpectationValueL<4>(qs, matrix, state);
      case 5:
        return ExpectationValueL<5>(qs, matrix, state);
      case 6:
        return ExpectationValueL<6>(qs, matrix, state);
      default:
        // Not implemented.
        break;
      }
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 32;
  }

 private:

// Original 
  template <unsigned G>
  struct IndicesH {
    static constexpr unsigned gsize = 1 << G;
    unsigned int gsize_t;
    static constexpr unsigned matrix_size = 2 * gsize * gsize * sizeof(fp_type);
    static constexpr unsigned xss_size = 32 * sizeof(idx_type) * (1 + (G == 6));
    static constexpr unsigned ms_size = 32 * sizeof(idx_type);
    static constexpr unsigned xss_offs = matrix_size;
    static constexpr unsigned ms_offs = xss_offs + xss_size;
    static constexpr unsigned buf_size = ms_offs + ms_size;

//IndicesH(float* p)
//        : xss((idx_type*) (p + xss_offs)), ms((idx_type*) (p + ms_offs)) {}
    IndicesH(char* p)
        : xss((idx_type*) (p + xss_offs)), ms((idx_type*) (p + ms_offs)) {}
   /* IndicesH(fp_type* p)
        : xss((idx_type*) (p + xss_offs)), ms((idx_type*) (p + ms_offs)) {} */

    idx_type* xss;
    idx_type* ms;
  };

  template <unsigned G>
  struct IndicesL : public IndicesH<G> {
    using Base = IndicesH<G>;
    static constexpr unsigned qis_size = 32 * sizeof(unsigned) * (1 + (G == 6));
    static constexpr unsigned tis_size = 32 * sizeof(unsigned);
    static constexpr unsigned qis_offs = Base::buf_size;
    static constexpr unsigned tis_offs = qis_offs + qis_size;
    static constexpr unsigned buf_size = tis_offs + tis_size;


//    IndicesL(float* p)
//        : Base(p), qis((unsigned*) (p + qis_offs)),
//          tis((unsigned*) (p + tis_offs)) {}
          
    IndicesL(char* p)
      : Base(p), qis((unsigned*) (p + qis_offs)),
         tis((unsigned*) (p + tis_offs)) {}

  /*  IndicesL(fp_type* p)
      : Base(p), qis((unsigned*) (p + qis_offs)),
         tis((unsigned*) (p + tis_offs)) {} */

    unsigned* qis;
    unsigned* tis;
  };


  template <unsigned G>
  struct IndicesLC : public IndicesL<G> {
    using Base = IndicesL<G>;
    static constexpr unsigned cis_size = 32 * sizeof(idx_type);
    static constexpr unsigned cis_offs = Base::buf_size;
    static constexpr unsigned buf_size = cis_offs + cis_size;

    IndicesLC(char* p) : Base(p), cis((idx_type*) (p + cis_offs)) {}

    idx_type* cis;
  };

  struct DataC {
    idx_type cvalsh;
    unsigned num_aqs;
    unsigned num_effective_qs;
    unsigned remaining_low_cqs;
  };





struct StateVectorLocation {
    size_t start_index;
    size_t end_index;
    int gpu_id;
    int node_rank;
};

std::unordered_map<size_t, StateVectorLocation> state_location_table;
std::vector<StateVectorLocation> state_location_vector;
int num_gpus_per_node;
int num_nodes;

void InitializeStateLocationTable(int num_gpus, int num_nodes, size_t total_size) {
    size_t chunk_size = total_size / (num_gpus * num_nodes);
    for (int node = 0; node < num_nodes; ++node) {
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            int index = node * num_gpus + gpu;
            StateVectorLocation loc = {
                index * chunk_size,
                (index + 1) * chunk_size - 1,
                gpu,
                node
            };
            state_location_table[loc.start_index] = loc;
            state_location_vector.push_back(loc);  
        }
    }
}

/* 
 * Adaptive Kernel Parameter Adjustment:
 * These functions dynamically configure CUDA block and thread parameters 
 * based on the number of qubits and gate width (G). 
 * To improve execution efficiency and memory safety, ScaleQsim enables 
 * safe mode for large-scale gates or high-qubit circuits. 
 * In safe mode, the number of blocks is constrained by real-time 
 * available GPU memory to avoid over-allocation and OOM errors. 
 * This adaptive mechanism ensures performance across varying workload size. 
*/


template <unsigned G>
void parameterConf_H(unsigned num_qubits, int gpuID,
                     unsigned long long& blocks, unsigned& threads) {
    unsigned k = 5 + G;
    unsigned n = (num_qubits > k) ? (num_qubits - k) : 0;
    unsigned long long size = 1ULL << n;
    threads = 64U;
    unsigned long long max_blocks = (1ULL << 30);


    bool safe_mode = (num_qubits >= 44 || G >= 7);  

    if (!safe_mode) {
        blocks = std::min(max_blocks, std::max(1ULL, size / threads));
        return;
    } else {

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (gpuID >= device_count) {
            printf("ERROR: Invalid GPU ID = %d (device_count = %d)\n", gpuID, device_count);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    
        cudaError_t err = cudaSetDevice(gpuID);
        if (err != cudaSuccess) {
            printf("cudaSetDevice(%d) failed: %s\n", gpuID, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    
        cudaFree(0);  
    
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
    
        size_t mem_per_thread = 2 * sizeof(fp_type);
        size_t mem_per_block = threads * mem_per_thread;
    
        unsigned long long mem_bound_blocks = free_mem / mem_per_block;
        unsigned long long size_bound_blocks = std::max(1ULL, size / threads);
    
        blocks = std::min({max_blocks, mem_bound_blocks, size_bound_blocks});
    }
}


void parameterConf_L(unsigned num_qubits, unsigned num_effective_qs, int gpuID,
                     unsigned long long& blocks, unsigned& threads) {
    unsigned k = 5 + num_effective_qs;
    unsigned n = (num_qubits > k) ? (num_qubits - k) : 0;
    unsigned long long size = 1ULL << n;
    threads = 32;  
    unsigned long long max_blocks = (1ULL << 30);

    bool safe_mode = (num_qubits >= 44 || num_effective_qs >= 7);  

    if (!safe_mode) {
        blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    } else {
        size_t free_mem, total_mem;
        cudaSetDevice(gpuID);
        cudaMemGetInfo(&free_mem, &total_mem);

        size_t mem_per_thread = 2 * sizeof(fp_type);
        size_t mem_per_block = threads * mem_per_thread;

        unsigned long long mem_bound_blocks = free_mem / mem_per_block;
        unsigned long long size_bound_blocks = std::max(1ULL, size / threads);

        blocks = std::min({max_blocks, mem_bound_blocks, size_bound_blocks});
    }
}

/*
 * ApplyGateH / ApplyGateL:
 * Launches a gate kernel on the assigned GPU using adaptive parameter configuration.
 * Performs local index generation, matrix transfer, and kernel execution based on target qubit range.
 * Execution parameters (block/thread) are adjusted according to qubit width and available memory.
*/
 

template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    using namespace std::chrono;

    auto start_time = high_resolution_clock::now();
    auto start_cpu = std::chrono::high_resolution_clock::now();

    unsigned num_qubits = 38;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    size_t num_nodes = world_size;

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus_per_node = multi_gpu_pointers.size();
    size_t total_gpus = num_nodes * num_gpus_per_node;

    size_t total_size = MinSize(num_qubits);
    size_t size_per_node = total_size / num_nodes;
    size_t size_per_gpu = size_per_node / num_gpus_per_node;
    size_t Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

    size_t global_gpu_id = world_rank * num_gpus_per_node + gpu_id;

    size_t node_offset = world_rank * size_per_node;
    size_t global_offset = global_gpu_id * Real_size_per_gpu;


    InitializeStateLocationTable(num_gpus_per_node, num_nodes, total_size); 

    
    cudaSetDevice(gpu_id);

    IndicesH<G> h_i(h_ws);
    GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);

    
    // GPU 타이머 시작
    static cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    cudaEvent_t start_gpu_copy, stop_gpu_copy;
    cudaEventCreate(&start_gpu_copy);
    cudaEventCreate(&stop_gpu_copy);
    
    cudaEventRecord(start_gpu_copy, 0);

    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);


    cudaEventRecord(stop_gpu_copy, 0);
    cudaEventSynchronize(stop_gpu_copy); 
    
    float gpu_time_copy = 0;
    cudaEventElapsedTime(&gpu_time_copy, start_gpu_copy, stop_gpu_copy);
    total_gpu_time_copy_h += gpu_time_copy;

    auto end_cpu_copy = std::chrono::high_resolution_clock::now();
    double cpu_time_copy = std::chrono::duration<double, std::milli>(end_cpu_copy - start_cpu).count();
    total_cpu_time_copy_h += cpu_time_copy;


    /*
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned long long size = 1ULL << n;
    unsigned threads = 64U;
    unsigned long long max_blocks = (1ULL << 30);
    unsigned long long blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    */


    unsigned long long blocks;
    unsigned threads;

    parameterConf_H<G>(num_qubits, gpu_id, blocks, threads);    

    IndicesH<G> d_i(d_ws_list_[gpu_id]);


    
 //   printf("DEBUG: Node%u, GPU%u --> Launching ApplyGateH_Kernel\n", world_rank, gpu_id);
    ApplyGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        (fp_type*) multi_gpu_pointers[gpu_id],  
        global_offset,                    
        Real_size_per_gpu,
        gpu_id,
        world_size,
        global_gpu_id
    );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateH failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    auto end_time = high_resolution_clock::now();
    double cpu_time = duration<double, std::milli>(end_time - start_time).count();

    total_cpu_time_H += cpu_time;
    total_gpu_time_H += gpu_time;


    
}



    
template <unsigned G>
void ApplyGateL(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 38;

    using namespace std::chrono;

    auto start_time = high_resolution_clock::now();
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    size_t num_nodes = world_size;  // 노드 개수
    
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus_per_node = multi_gpu_pointers.size();
    size_t total_gpus = num_nodes * num_gpus_per_node;

    size_t total_size = MinSize(num_qubits);
    size_t size_per_node = total_size / num_nodes;
    size_t size_per_gpu = size_per_node / num_gpus_per_node;

    size_t global_gpu_id = world_rank * num_gpus_per_node + gpu_id;

    size_t node_offset = world_rank * size_per_node;
    size_t Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;
    size_t global_offset = global_gpu_id * Real_size_per_gpu;



    cudaSetDevice(gpu_id);
 

    IndicesL<G> h_i(h_ws);
    

    auto num_effective_qs = GetIndicesL_m(num_qubits, qs, h_i, global_offset);

    static cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    cudaEvent_t start_gpu_copy, stop_gpu_copy;
    cudaEventCreate(&start_gpu_copy);
    cudaEventCreate(&stop_gpu_copy);
    
    cudaEventRecord(start_gpu_copy, 0);

    
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

  //  printf("DEBUG: ApplyGateL Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",  h_i.matrix_size, h_ws, d_ws_list_[gpu_id]);

    cudaEventRecord(stop_gpu_copy, 0);
    cudaEventSynchronize(stop_gpu_copy);  
    
    float gpu_time_copy = 0;
    cudaEventElapsedTime(&gpu_time_copy, start_gpu_copy, stop_gpu_copy);
    total_gpu_time_copy_l += gpu_time_copy;

    auto end_cpu_copy = std::chrono::high_resolution_clock::now();
    double cpu_time_copy = std::chrono::duration<double, std::milli>(end_cpu_copy - start_cpu).count();
    total_cpu_time_copy_l += cpu_time_copy;


    /*
    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned long long size = 1ULL << n;
    unsigned threads = 32;
    unsigned long long max_blocks = (1ULL << 30);
    unsigned long long blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    */


    unsigned long long blocks;
    unsigned threads;

    parameterConf_L(num_qubits, num_effective_qs, gpu_id, blocks, threads);
    

    
    IndicesL<G> d_i(d_ws_list_[gpu_id]);


  //  printf("DEBUG: Preparing to launch ApplyGateL_Kernel\n");
    ApplyGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   // v0: 행렬 데이터
        d_i.xss,                         // xss: 인덱스 변환 테이블
        d_i.ms,                          // mss: 비트마스크 정보
        d_i.qis,                         // qis: 큐비트 정보
        d_i.tis,                         // tis: 큐비트 정보
        1 << num_effective_qs,           // esize: 활성 큐비트 수
        (fp_type*) multi_gpu_pointers[gpu_id],  // rstate: 현재 GPU의 상태 벡터
        global_offset,                    // global_offset: 현재 GPU의 메모리 시작점
        Real_size_per_gpu,                      // size_per_gpu: 각 GPU가 담당하는 메모리 크기      
        gpu_id,
        world_size,
        global_gpu_id
        );

    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);




    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateL failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    auto end_time = std::chrono::high_resolution_clock::now();  
    double cpu_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  
    total_cpu_time += cpu_time;
    total_gpu_time += gpu_time;
        
 
}

  template <unsigned G>
  void ApplyControlledGateHH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, idx_type cvals,
                             const fp_type* matrix, State& state) const {
    unsigned aqs[64];
    idx_type cmaskh = 0;
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);

    unsigned num_aqs = GetHighQubits(qs, 0, cqs, 0, 0, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, h_i.ms);
    GetXss(num_qubits, qs, qs.size(), h_i.xss);

    idx_type cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);
    printf("KCJ: ApplyControlledGateHH start\n");
    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);

        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    IndicesH<G> d_i(d_ws);

    ApplyControlledGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_aqs + 1, cvalsh, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateLH(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs, uint64_t cvals,
                             const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto d = GetIndicesLC(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    printf("KCJ: ApplyControlledGateLH start\n");

        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesL<G> d_i(d_ws);

    ApplyControlledGateLH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs, state.get());
  }

  template <unsigned G>
  void ApplyControlledGateL(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs, uint64_t cvals,
                            const fp_type* matrix, State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesLC<G> h_i(h_ws);
    auto d = GetIndicesLCL(num_qubits, qs, cqs, cvals, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    unsigned k = 5 + G + cqs.size();
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesLC<G> d_i(d_ws);

    ApplyControlledGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis, d_i.cis,
        d.num_aqs + 1, d.cvalsh, 1 << d.num_effective_qs,
        1 << (5 - d.remaining_low_cqs), state.get());
  }

  template <unsigned G>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    //ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
        
    printf("KCJ: ExpectationValueH - cudaMemcpyAsync\n");
    fflush(stdout);
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 14 ? n - 14 : 0, 4U);
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, (size / 2) >> s);
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;

    printf("KCJ: ExpectationValueH - AllocScratch");
    //fflush(stdout);

    IndicesH<G> d_i(d_ws);

    ExpectationValueH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, num_iterations_per_block,
        state.get(), Plus<double>(), d_res1);

    printf("KCJ: ExpectationValueH - after Kernel Execution");
    
    double mul = size == 1 ? 0.5 : 1.0;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned G>
  std::complex<double> ExpectationValueL(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    unsigned num_qubits = state.num_qubits();

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);

    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    //ErrorCheck(
        cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    printf("KCJ: ExpectationValueL - cudaMemcpyAsync\n");

    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;

    unsigned s = std::min(n >= 13 ? n - 13 : 0, 5U);
    unsigned threads = 32;
    unsigned blocks = size >> s;
    unsigned num_iterations_per_block = 1 << s;

    constexpr unsigned m = 16;

    Complex* d_res1 = (Complex*) AllocScratch((blocks + m) * sizeof(Complex));
    Complex* d_res2 = d_res1 + blocks;
    
    printf("KCJ: ExpectationValueL - AllocScratch");

    IndicesL<G> d_i(d_ws);

    ExpectationValueL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        num_iterations_per_block, state.get(), Plus<double>(), d_res1);
        
    printf("KCJ: ExpectationValueL - after Kernel Execution");


    double mul = double(1 << (5 + num_effective_qs - G)) / 32;

    return ExpectationValueReduceFinal<m>(blocks, mul, d_res1, d_res2);
  }

  template <unsigned m>
  std::complex<double> ExpectationValueReduceFinal(
      unsigned blocks, double mul,
      const Complex* d_res1, Complex* d_res2) const {
    Complex res2[m];
    
    printf("KCJ: ExpectationValueReduceFinal start\n");

    if (blocks <= 16) {
      //ErrorCheck(
      cudaMemcpy(res2, d_res1, blocks * sizeof(Complex), cudaMemcpyDeviceToHost);
      printf("KCJ: cudaMemcpyD2H right?\n ");

    } else {
      unsigned threads2 = std::min(1024U, blocks);
      unsigned blocks2 = std::min(m, blocks / threads2);

      unsigned dblocks = std::max(1U, blocks / (blocks2 * threads2));
      unsigned bytes = threads2 * sizeof(Complex);

      Reduce2Kernel<Complex><<<blocks2, threads2, bytes>>>(
          dblocks, blocks, Plus<Complex>(), Plus<double>(), d_res1, d_res2);

      //ErrorCheck(
      cudaMemcpy(res2, d_res2, blocks2 * sizeof(Complex),
                            cudaMemcpyDeviceToHost);
      
      printf("KCJ: cudaMemcpyD2H right?\n ");

      blocks = blocks2;
    }

    double re = 0;
    double im = 0;

    for (unsigned i = 0; i < blocks; ++i) {
      re += res2[i].re;
      im += res2[i].im;
    }

    return {mul * re, mul * im};
  }

  template <typename AQ>
  unsigned GetHighQubits(const std::vector<unsigned>& qs, unsigned qi,
                         const std::vector<unsigned>& cqs, unsigned ci,
                         unsigned ai, idx_type& cmaskh, AQ& aqs) const {
    while (1) {
      if (qi < qs.size() && (ci == cqs.size() || qs[qi] < cqs[ci])) {
        aqs[ai++] = qs[qi++];
      } else if (ci < cqs.size()) {
        cmaskh |= idx_type{1} << cqs[ci];
        aqs[ai++] = cqs[ci++];
      } else {
        break;
      }
    }
    
    printf("KCJ: GetHighQubits- ai: %u\n", ai);

    return ai;
  }

  template <typename QS>
  void GetMs(unsigned num_qubits, const QS& qs, unsigned qs_size,
             idx_type* ms) const {
    printf("KCJ: GetMs start\n");
    if (qs_size == 0) {
      ms[0] = idx_type(-1);
    } else {
      idx_type xs = idx_type{1} << (qs[0] + 1);
      ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < qs_size; ++i) {
        ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs - 1);
        xs = idx_type{1} << (qs[i] + 1);
      }
      ms[qs_size] = ((idx_type{1} << num_qubits) - 1) ^ (xs - 1);
    }
  }

  template <typename QS>
  void GetXss(unsigned num_qubits, const QS& qs, unsigned qs_size,
              idx_type* xss) const {
    printf("KCJ: GetXss start\n");
    if (qs_size == 0) {
      xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
      }

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        xss[i] = a;
      }
    }
  }



// Original
  template <unsigned G, typename qs_type>
  void GetIndicesH(unsigned num_qubits, const qs_type& qs, unsigned qs_size,
                   IndicesH<G>& indices) const {
    printf("KCJ: GetIndicesH start\n");
    num_qubits = 38;
    if (qs_size == 0) {
      indices.ms[0] = idx_type(-1);
      indices.xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      indices.ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
        indices.ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs[i - 1] - 1);
      }
      indices.ms[g] = ((idx_type{1} << num_qubits) - 1) ^ (xs[g - 1] - 1);

      for (unsigned i = 0; i < gsize; ++i) {
        idx_type a = 0;
        for (unsigned k = 0; k < g; ++k) {
          a += xs[k] * ((i >> k) & 1);
        }
        indices.xss[i] = a;
      }
    }
  }

  template <unsigned G>
  void GetIndicesL(unsigned num_effective_qs, unsigned qmask,
                   IndicesL<G>& indices) const {
     printf("KCJ: GetIndicesL start\n");
    for (unsigned i = num_effective_qs + 1; i < (G + 1); ++i) {
      indices.ms[i] = 0;
    }

    for (unsigned i = (1 << num_effective_qs); i < indices.gsize; ++i) {
      indices.xss[i] = 0;
    }

    for (unsigned i = 0; i < indices.gsize; ++i) {
      indices.qis[i] = bits::ExpandBits(i, 5 + num_effective_qs, qmask);
    }

    unsigned tmask = ((1 << (5 + num_effective_qs)) - 1) ^ qmask;
    for (unsigned i = 0; i < 32; ++i) {
      indices.tis[i] = bits::ExpandBits(i, 5 + num_effective_qs, tmask);
    }
  }


  template <unsigned G>
  unsigned GetIndicesL(unsigned num_qubits, const std::vector<unsigned>& qs,
                       IndicesL<G>& indices) const {
                       
    num_qubits = 38;
      
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits);
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    if (qs.size() == num_low_qs) {
      while (ei < num_effective_qs && l++ < num_low_qs) {
        eqs[ei] = ei + 5;
        ++ei;
      }
    } else {
      while (ei < num_effective_qs && l < num_low_qs) {
        unsigned ei5 = ei + 5;
        eqs[ei] = ei5;
        if (qi < qs.size() && qs[qi] == ei5) {
          ++qi;
          qmaskh |= 1 << ei5;
        } else {
          ++l;
        }
        ++ei;
      }

      while (ei < num_effective_qs) {
        eqs[ei] = qs[qi++];
        qmaskh |= 1 << (ei + 5);
        ++ei;
      }
    }
    
    GetIndicesH(num_qubits, eqs, num_effective_qs, indices);
    //printf("KCJ: Call GetIndicesH ! \n");
    
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);
    //printf("KCJ: Call GetIndicesL ! \n");

    return num_effective_qs;
  }



template <unsigned G, typename qs_type>
void GetIndicesH_m(unsigned num_qubits, const qs_type& qs, unsigned qs_size,
                    IndicesH<G>& indices, size_t global_offset) {

  //  printf("KCJ: GetIndicesH_M start\n");
    num_qubits = 38;
    
   if (qs_size == 0) {
      indices.ms[0] = idx_type(-1);
      indices.xss[0] = 0;
    } else {
      unsigned g = qs_size;
      unsigned gsize = 1 << qs_size;

      idx_type xs[64];

      xs[0] = idx_type{1} << (qs[0] + 1);
      indices.ms[0] = (idx_type{1} << qs[0]) - 1;
      for (unsigned i = 1; i < g; ++i) {
        xs[i] = idx_type{1} << (qs[i] + 1);
        indices.ms[i] = ((idx_type{1} << qs[i]) - 1) ^ (xs[i - 1] - 1);
      }
      indices.ms[g] = ((idx_type{1} << num_qubits) - 1) ^ (xs[g - 1] - 1);

        //size_t local_index_wrapped = 0;
        for (unsigned i = 0; i < gsize; ++i) {
            idx_type a = 0;
            for (unsigned k = 0; k < g; ++k) {
                a += xs[k] * ((i >> k) & 1);
            }
  
            indices.xss[i] = a;

        //    printf("DEBUG: GPU offset=%lu, xs[0]=%lu, xs[1]=%lu, xs[2]=%lu\n", global_offset, xs[0], xs[1], xs[2]);  
        //    printf("DEBUG: indices.ms[%u]: %u\n", i, indices.ms[i]);
            //  로그 추가
        //    printf("DEBUG: i=%u, a=%lu, global_offset=%lu, xss[%u] = %lu (global_index)\n",           i, a, global_offset, i, indices.xss[i]);
        }
    }

}


template <unsigned G>
void GetIndicesL_m(unsigned num_effective_qs, unsigned qmask,
                    IndicesL<G>& indices, size_t global_offset) {
   // printf("KCJ: GetIndicesL_M start\n");

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    unsigned num_qubits = 38;
    size_t num_gpus = multi_gpu_pointers.size();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = total_size / num_gpus;

    //  `H`에서 `xss`, `ms`를 설정했으므로, `L`에서는 0으로 초기화
      for (unsigned i = num_effective_qs + 1; i < (G + 1); ++i) {
      indices.ms[i] = 0;
    }

    for (unsigned i = (1 << num_effective_qs); i < indices.gsize; ++i) {
      indices.xss[i] = 0;
    }

    for (unsigned i = 0; i < indices.gsize; ++i) {
      indices.qis[i] = bits::ExpandBits(i, 5 + num_effective_qs, qmask);
    }

    unsigned tmask = ((1 << (5 + num_effective_qs)) - 1) ^ qmask;
    for (unsigned i = 0; i < 32; ++i) {
      indices.tis[i] = bits::ExpandBits(i, 5 + num_effective_qs, tmask);
    }

    /*
    
    //  디버깅 로그 추가 (GPU별 `qis`, `tis`, `xss` 값 검증)
    for (unsigned i = 0; i < std::min(size_t(10), static_cast<size_t>(indices.gsize)); ++i) {
        printf("DEBUG: GetIndicesL_m - indices.xss[%u] = %lu, indices.qis[%u] = %lu, indices.tis[%u] = %lu\n",
               i, indices.xss[i], i, indices.qis[i], i, indices.tis[i]);
    } */
}



template <unsigned G>
unsigned GetIndicesL_m(unsigned num_qubits, const std::vector<unsigned>& qs,
                      IndicesL<G>& indices, size_t global_offset)  {
    num_qubits = 38;

    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
        qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits);
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    if (qs.size() == num_low_qs) {
        while (ei < num_effective_qs && l++ < num_low_qs) {
            eqs[ei] = ei + 5;
            ++ei;
        }
    } else {
        while (ei < num_effective_qs && l < num_low_qs) {
            unsigned ei5 = ei + 5;
            eqs[ei] = ei5;
            if (qi < qs.size() && qs[qi] == ei5) {
                ++qi;
                qmaskh |= 1 << ei5;
            } else {
                ++l;
            }
            ++ei;
        }

        while (ei < num_effective_qs) {
            eqs[ei] = qs[qi++];
            qmaskh |= 1 << (ei + 5);
            ++ei;
        }
    }

  //  printf("DEBUG: qmaskh = %u, qmaskl = %u, qmaskh | qmaskl = %u\n", qmaskh, qmaskl, qmaskh | qmaskl);

    
    GetIndicesH_m(num_qubits, eqs, num_effective_qs, indices, global_offset);
  //  printf("KCJ: Call GetIndicesH_M ! \n");
       GetIndicesL_m(num_effective_qs, qmaskh | qmaskl, indices, global_offset);
 //   printf("KCJ: Call GetIndicesL_M ! \n");

    return num_effective_qs;
}




  template <unsigned G>
  DataC GetIndicesLC(unsigned num_qubits, const std::vector<unsigned>& qs,
                     const std::vector<unsigned>& cqs, uint64_t cvals,
                     IndicesL<G>& indices) const {
                     
    printf("KCJ: GetIndicesLC start\n");
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;

    unsigned qi = 0;

    while (qi < qs.size() && qs[qi] < 5) {
      qmaskl |= 1 << qs[qi++];
    }

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ci = 0;
    unsigned ei = 0;
    unsigned num_low_qs = qi;

    while (ai < num_qubits && l < num_low_qs) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        eqs[ei++] = ai;
        qmaskh |= 1 << (ai - ci);
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        eqs[ei++] = ai;
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = qi;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);

    return {cvalsh, num_aqs, num_effective_qs};
  }

  template <unsigned G>
  DataC GetIndicesLCL(unsigned num_qubits, const std::vector<unsigned>& qs,
                      const std::vector<unsigned>& cqs, uint64_t cvals,
                      IndicesLC<G>& indices) const {
    printf("KCJ: GetIndicesLCL start\n");
    unsigned aqs[64];
    unsigned eqs[32];

    unsigned qmaskh = 0;
    unsigned qmaskl = 0;
    idx_type cmaskh = 0;
    idx_type cmaskl = 0;
    idx_type cis_mask = 0;

    unsigned qi = 0;
    unsigned ci = 0;

    for (unsigned k = 0; k < 5; ++k) {
      if (qi < qs.size() && qs[qi] == k) {
        qmaskl |= 1 << (k - ci);
        ++qi;
      } else if (ci < cqs.size() && cqs[ci] == k) {
        cmaskl |= idx_type{1} << k;
        ++ci;
      }
    }

    unsigned num_low_qs = qi;
    unsigned num_low_cqs = ci;

    unsigned nq = std::max(5U, num_qubits - unsigned(cqs.size()));
    unsigned num_effective_qs = std::min(nq - 5, unsigned(qs.size()));

    unsigned l = 0;
    unsigned ai = 5;
    unsigned ei = 0;
    unsigned num_low = num_low_qs + num_low_cqs;
    unsigned remaining_low_cqs = num_low_cqs;
    unsigned effective_low_qs = num_low_qs;
    unsigned highest_cis_bit = 0;

    while (ai < num_qubits && l < num_low) {
      aqs[ai - 5] = ai;
      if (qi < qs.size() && qs[qi] == ai) {
        ++qi;
        if ((ai - ci) > 4) {
          eqs[ei++] = ai;
          qmaskh |= 1 << (ai - ci);
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          qmaskl |= 1 << (ai - ci);
          --remaining_low_cqs;
          ++effective_low_qs;
        }
      } else if (ci < cqs.size() && cqs[ci] == ai) {
        ++ci;
        cmaskh |= idx_type{1} << ai;
      } else {
        ++l;
        if (remaining_low_cqs == 0) {
          eqs[ei++] = ai;
        } else {
          highest_cis_bit = ai;
          cis_mask |= idx_type{1} << ai;
          --remaining_low_cqs;
        }
      }
      ++ai;
    }

    unsigned i = ai;
    unsigned j = effective_low_qs;

    while (ei < num_effective_qs) {
      eqs[ei++] = qs[j++];
      qmaskh |= 1 << (i++ - ci);
    }

    unsigned num_aqs = GetHighQubits(qs, qi, cqs, ci, ai - 5, cmaskh, aqs);
    GetMs(num_qubits, aqs, num_aqs, indices.ms);
    GetXss(num_qubits, eqs, num_effective_qs, indices.xss);
    GetIndicesL(num_effective_qs, qmaskh | qmaskl, indices);

    idx_type cvalsh = bits::ExpandBits(idx_type(cvals), num_qubits, cmaskh);
    idx_type cvalsl = bits::ExpandBits(idx_type(cvals), 5, cmaskl);

    cis_mask |= 31 ^ cmaskl;
    highest_cis_bit = highest_cis_bit < 5 ? 5 : highest_cis_bit;
    for (idx_type i = 0; i < 32; ++i) {
      auto c = bits::ExpandBits(i, highest_cis_bit + 1, cis_mask);
      indices.cis[i] = 2 * (c & 0xffffffe0) | (c & 0x1f) | cvalsl;
    }

    return {cvalsh, num_aqs, num_effective_qs, remaining_low_cqs};
  }


  //void* AllocScratch(uint64_t size) const {
  //KCJ: Debug.
    void* AllocScratch(uint64_t size) const __attribute__((noinline)) {
    if (size > scratch_size_) {
            std::cout << "Allocating memory with cudaMalloc of size: 1" << size << std::endl;

      if (scratch_ != nullptr) {
              std::cout << "Allocating memory with cudaMalloc of size: 2" << size << std::endl;

        //ErrorCheck(cudaFree(scratch_));
        cudaFree(scratch_);
      }
              std::cout << "Allocating memory with cudaMalloc of size: 3" << size << std::endl;

      //ErrorCheck(cudaMalloc(const_cast<void**>(&scratch_), size));
      cudaMalloc(const_cast<void**>(&scratch_), size);
      printf("KCJ: cudaMalloc check size: %llu", size);
      
      const_cast<uint64_t&>(scratch_size_) = size;
      
    }

    
    return scratch_;
  }

  /* KCJ: 
d_ws: Device workspace pointer for storing gate and index data during CUDA operations.
h_ws (buf_size): Host-side workspace buffer with a maximum size, used to prepare data before copying to device.
h_ws: Pointer to host workspace, pointing to the start of h_ws0 buffer for easier handling.
scratch: Pointer to a dynamically allocated scratch buffer on the device, used for temporary storage.
scratch_size: Size of the scratch buffer in bytes, updated whenever more space is needed for computations.

*/


  
};


template <>
double SimulatorCUDA<float>::total_index_calc_time = 0.0;
template <>
double SimulatorCUDA<float>::total_apply_gate_time = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_H = 0.0;
template <>
float SimulatorCUDA<float>::total_gpu_time_H = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time = 0.0;
template <>
float SimulatorCUDA<float>::total_gpu_time = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_copy_h = 0.0;   
template <>
float SimulatorCUDA<float>::total_gpu_time_copy_h = 0.0;
template <>
double SimulatorCUDA<float>::total_cpu_time_copy_l = 0.0;   
template <>
float SimulatorCUDA<float>::total_gpu_time_copy_l = 0.0;
}  // namespace qsim

#endif  // SIMULATOR_CUDA_H_

