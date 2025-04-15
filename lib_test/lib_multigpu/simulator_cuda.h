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
#include </opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include/mpi.h>





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
  
SimulatorCUDA(const std::vector<StateSpace>& state_spaces,
              const std::vector<ncclComm_t>& nccl_comms)
    : state_spaces_(state_spaces),
      nccl_comms_(nccl_comms),
      scratch_size_(0) {

    
    size_t num_gpus = state_spaces_.size();
    printf("KCJ DEBUG: Number of GPUs detected: %lu\n", num_gpus);


    // num_qubits 설정 (예: 8로 고정)
    unsigned num_qubits = 29;
    printf("KCJ: SimulatorCUDA > num_qubits: %u\n", num_qubits);
    //size_t total_size = sizeof(fp_type) * MinSize(num_qubits);
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;
        //size_t size_per_gpu = MinSize(num_qubits);
    printf("KCJ DEBUG: Initializing state parts across %lu GPUs.\n", num_gpus);
        
    // d_ws_list_ 크기 조정
    d_ws_list_.resize(num_gpus);
    printf("KCJ DEBUG: Resized d_ws_list_ to hold %lu entries\n", d_ws_list_.size());

    cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
        printf("ERROR: Simulator 2 - CUDA error before kernel launch on GPU %lu: %s\n", 0, cudaGetErrorString(err_before));
    }

        
    for (size_t i = 0; i < num_gpus; ++i) {
        // MultiGPU 상태 벡터 생성
        auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
        VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::CreateMultiGPU(num_qubits, num_gpus); 

        cudaSetDevice(i);
        //multi ver. - Multi pointer of number of GPUs.
        fp_type* gpu_state_ptr = multi_gpu_pointers[i];
        state_parts_.push_back(gpu_state_ptr);
         
         //d_ws_list_[i] = nullptr;
         printf("KCJ: Initiaized Buffer GPU: d_ws_list[%u], %p \n", i, d_ws_list_[i]);
         cudaDeviceSynchronize();
     }
        
    AllocateWorkspace(num_gpus);
    
    // 모든 상태 벡터를 0으로 초기화
    SetAllZeros();
    
    InitializeState();
  
        //InitializeStatespace(num_gpus, total_size);

    }
    ~SimulatorCUDA() {
        // Free workspace memory for each GPU
        
        for (size_t i = 0; i < d_ws_list_.size(); ++i) {
            cudaSetDevice(i); // Set the appropriate GPU
            if (d_ws_list_[i] != nullptr) {
                cudaError_t err = cudaFree(d_ws_list_[i]);
                if (err != cudaSuccess) {
                    printf("ERROR: cudaFree failed for d_ws_list_[%lu]: %s\n", i, cudaGetErrorString(err));
                } else {
                    printf("DEBUG: Successfully freed memory for d_ws_list_[%lu].\n", i);
                }
            }
        }
    
        // Free scratch memory if allocated
        if (scratch_ != nullptr) {
            cudaError_t err = cudaFree(scratch_);
            if (err != cudaSuccess) {
                printf("ERROR: cudaFree failed for scratch_: %s\n", cudaGetErrorString(err));
            } else {
                printf("DEBUG: Successfully freed scratch_ memory.\n");
            }
        }
    }
  

  static constexpr unsigned max_buf_size = 8192 * sizeof(FP)
      + 128 * sizeof(idx_type) + 96 * sizeof(unsigned);
      
  char* d_ws;
  char h_ws0[max_buf_size];
  char* h_ws = (char*) h_ws0;



void AllocateWorkspace(size_t num_gpus) {
    printf("DEBUG: Allocating workspace for %lu GPUs\n", num_gpus);

    for (unsigned i = 0; i < num_gpus; ++i) {
        cudaError_t status = cudaSetDevice(i);
        printf("KCJ: Access GPU %u\n", i);
        if (status != cudaSuccess) {
            printf("ERROR: Failed to set device %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }

        // 메모리 할당 상태 점검
        size_t free_mem = 0, total_mem = 0;
        status = cudaMemGetInfo(&free_mem, &total_mem);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMemGetInfo failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            continue;
        }
        printf("INFO: GPU %u - Free memory: %lu bytes, Total memory: %lu bytes\n", i, free_mem, total_mem);

        // 메모리 할당
        status = cudaMalloc(&d_ws_list_[i], max_buf_size);
        if (status != cudaSuccess) {
            printf("ERROR: cudaMalloc failed on GPU %u: %s\n", i, cudaGetErrorString(status));
            d_ws_list_[i] = nullptr;
            continue;
        }

        printf("INFO: Workspace allocated on GPU %u at %p (size: %lu bytes)\n", i, d_ws_list_[i], max_buf_size);

        // 동기화
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            printf("ERROR: cudaDeviceSynchronize failed on GPU %u: %s\n", i, cudaGetErrorString(status));
        }
    }
}


void InitializeState() {
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus = multi_gpu_pointers.size();
    unsigned num_qubits = 29;
    size_t total_states = MinSize(num_qubits);
    size_t size_per_gpu = total_states / num_gpus;

    printf("KCJ: Initializing multi-GPU state |0⟩\n");

    /*
    for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        size_t global_index = gpu_id * size_per_gpu;
        printf("DEBUG: Initializing GPU %lu, Global index %lu\n", gpu_id, global_index);
        
        SetAmpl(global_index, 1.0, 0.0);
    }
    */
   // SetAmpl(0, 1.0, 0.0);

    printf("KCJ: Initialization complete.\n");
}




/*

// KCJ-메모리 해제 함수
static void ClearNonStateMemory(void* base_addr, size_t total_size, void* state_addr, size_t state_size) {
    printf("KCJ: ClearNonStateMemory called with base_addr=%p, total_size=%zu, state_addr=%p, state_size=%zu\n", 
           base_addr, total_size, state_addr, state_size);

    if (state_addr > base_addr) {
        printf("KCJ: Freeing memory before state: base_addr=%p\n", base_addr);
        cudaFree(base_addr);  // state 이전 메모리 블록 해제
        base_addr = nullptr;  // 포인터 무효화
    } else {
        printf("KCJ: No memory to free before state\n");
    }

    void* state_end = static_cast<char*>(state_addr) + state_size;
    if (state_end < static_cast<char*>(base_addr) + total_size) {
        void* clear_start = state_end;
        printf("KCJ: Freeing memory after state: clear_start=%p\n", clear_start);
        cudaFree(clear_start);  // state 이후 메모리 블록 해제
        clear_start = nullptr;  // 포인터 무효화
    } else {
        printf("KCJ: No memory to free after state\n");
    }

    printf("KCJ: ClearNonStateMemory completed\n");
}


// KCJ: Memory Free and allocation with Memory Tracking
void FreeAndAllocateMemory(State& state, unsigned num_qubits) {
    size_t free_mem, total_mem;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("KCJ DEBUG: Before Memory Management - Used: %llu, Free: %llu, Total: %llu\n",
           total_mem - free_mem, free_mem, total_mem);

    if (state.get() != nullptr) {
    printf("KCJ DEBUG: Freeing GPU memory. state.get(): %p\n", state.get());
    cudaError_t freeStatus = cudaFree(state.get());
    if (freeStatus == cudaSuccess) {
        state.set(nullptr, num_qubits); // 포인터를 null로 초기화하여 중복 해제 방지
        printf("KCJ DEBUG: Memory freed successfully.\n");
    } else {
        printf("KCJ DEBUG: CUDA free failed with error: %s\n", cudaGetErrorString(freeStatus));
    }
}

    // 메모리 해제 후 업데이트
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("KCJ DEBUG: After Memory Free - Used: %llu, Free: %llu, Total: %llu\n",
           total_mem - free_mem, free_mem, total_mem);

    // 메모리 재할당
    fp_type* device_ptr = nullptr;
    auto size_bytes = sizeof(fp_type) * MinSize(num_qubits);
    if (size_bytes > free_mem) {
        printf("KCJ DEBUG: Not enough memory available! Required: %llu bytes, Free: %llu bytes\n", size_bytes, free_mem);
        return;
    }
    
   cudaError_t mallocStatus = cudaMalloc(&device_ptr, size_bytes);
    cudaDeviceSynchronize();

    if (mallocStatus != cudaSuccess) {
        printf("KCJ DEBUG: CUDA malloc failed with error: %s\n", cudaGetErrorString(mallocStatus));
        return;
    }

    cudaMemset(device_ptr, 0, MinSize(state.num_qubits()) * sizeof(fp_type));
    state.set(device_ptr, num_qubits);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("KCJ DEBUG: New memory allocated successfully. Used: %llu, Free: %llu, Total: %llu\n",
           total_mem - free_mem, free_mem, total_mem);
           
    printf("KCJ DEBUG: Setting initial amplitude using SetAmpl.\n");
    SetAmpl(state, 0, 1.0, 0.0);  // 첫번째 진폭을 1로 초기화 (기본 상태)    
}
*/


//---------------------------------------------------------------



static uint64_t MinSize(unsigned num_qubits) {
    printf("KCJ: simulator_cuda-MinSize Start\n");
      //kcj
    num_qubits = 29;
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    printf("KCJ: simulator_cuda-MinSize Result: %llu\n", result);
    return result;
      
};


// KCJ Multiver.
void SetAllZeros() {
    printf("KCJ: simulator-SetAllZeros: cudaMemset for all GPUs\n");
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus = multi_gpu_pointers.size();
    unsigned num_qubits = 29;
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (num_gpus == 1) ? total_size : total_size / num_gpus;

    // ✅ 1. 모든 GPU의 상태 벡터를 `0`으로 초기화
    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        printf("KCJ: SetAllZeros on GPU %lu - state_part address: %p, size: %lu bytes\n", 
               i, multi_gpu_pointers[i], size_per_gpu);
        cudaMemset(multi_gpu_pointers[i], 0, size_per_gpu);
        cudaDeviceSynchronize();
    }

    
    fp_type one[1] = {1};
    for (size_t i = 0; i < num_gpus; ++i) {
        cudaSetDevice(0);
        cudaMemcpy(multi_gpu_pointers[0], one, sizeof(fp_type), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        printf("DEBUG: GPU %lu initialized with |0⟩ state.\n", 0);
   }

    cudaDeviceSynchronize();
    printf("KCJ: SetAllZeros completed for all GPUs\n");

    // ✅ 3. 디버깅: 모든 GPU의 상태 벡터 확인
    for (size_t i = 0; i < num_gpus; ++i) {
        fp_type* host_check = new fp_type[size_per_gpu / sizeof(fp_type)];
        cudaMemcpy(host_check, multi_gpu_pointers[i], size_per_gpu, cudaMemcpyDeviceToHost);
        printf("DEBUG: Simulator: SetAllZeros Initial state vector on GPU %lu:\n", i);
        for (size_t j = 0; j < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++j) {
            printf("GPU %lu - state[%lu]: %f\n", i, j, host_check[j]);
        }
        delete[] host_check;
    }
}



void SetAmpl(uint64_t i, fp_type re, fp_type im) {
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t num_gpus = multi_gpu_pointers.size();
    unsigned num_qubits = 29;
    size_t total_states = MinSize(num_qubits); // 단일 Statespace 크기
    size_t size_per_gpu = total_states / num_gpus;

    if (i >= total_states) {
        printf("ERROR: Index %lu exceeds total state size\n", i);
        return;
    }

    size_t gpu_id = (i / size_per_gpu) % num_gpus;  // GPU ID
    size_t local_index = i % size_per_gpu;          // GPU 내 로컬 인덱스

    printf("DEBUG: SetAmpl - global index: %lu -> GPU %lu, local index: %lu\n",
           i, gpu_id, local_index);

    fp_type* p = multi_gpu_pointers[gpu_id] + local_index;

    cudaSetDevice(gpu_id);
    
    // ✅ 모든 GPU의 첫 번째 로컬 인덱스를 명확하게 설정
    if (local_index == 0) {
        cudaMemcpy(p, &re, sizeof(fp_type), cudaMemcpyHostToDevice);
        cudaMemcpy(p + 1, &im, sizeof(fp_type), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        printf("DEBUG: GPU %lu - SetAmpl() first index updated.\n", gpu_id);
    }
}



//-----------------------------------------------------------------

/*
void ApplyGate(const std::vector<unsigned>& qs,
               const fp_type* matrix, State& state) const {
    size_t num_gpus = d_ws_list_.size();  // 사용 가능한 GPU 수
    bool is_entangling = (qs.size() > 1 && qs[0] > 4);

    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
    for (auto q : qs) {
        printf("%u ", q);
    }
    printf("\n");

    // 1. 비얽힘 게이트 처리
    if (!is_entangling) {
        for (size_t i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            switch (qs.size()) {
                case 1:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state);
                    break;
                case 2:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state);
                    break;
                case 3:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state);
                    break;
                case 4:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state);
                    break;
                case 5:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state);
                    break;
                case 6:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state);
                    break;
                default:
                    printf("KCJ: Unsupported non-entangling gate size.\n");
                    break;
            }
        }

        // NCCL AllReduce로 결과 병합
        if (num_gpus > 1) {
            size_t size_per_gpu = MinSize(state.num_qubits()) / num_gpus;
            ncclAllReduce(state.get(), state.get(), size_per_gpu, ncclFloat, ncclSum,
                          nccl_comms_[0], cudaStreamDefault);
            cudaDeviceSynchronize();
            printf("DEBUG: NCCL AllReduce completed for non-entangling gates.\n");
        }
    }

    // 2. 얽힘 게이트 처리
    if (is_entangling) {
        printf("KCJ: Applying entangling gates conditionally.\n");
        for (size_t i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            switch (qs.size()) {
                case 2:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state);
                    break;
                case 3:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state);
                    break;
                case 4:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state);
                    break;
                case 5:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state);
                    break;
                case 6:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state);
                    break;
                default:
                    printf("KCJ: Unsupported entangling gate size.\n");
                    break;
            }
        }

        // NCCL AllReduce로 결과 병합
        if (num_gpus > 1) {
            size_t size_per_gpu = MinSize(state.num_qubits()) / num_gpus;
            ncclAllReduce(state.get(), state.get(), size_per_gpu, ncclFloat, ncclSum,
                          nccl_comms_[0], cudaStreamDefault);
            cudaDeviceSynchronize();
            printf("DEBUG: NCCL AllReduce completed for entangling gates.\n");
        }
    }

    printf("KCJ: ApplyGate completed successfully.\n");
}
*/



/*
//Batch Test 
void ApplyGate(const std::vector<unsigned>& qs,
               const fp_type* matrix, State& state) const {
    static ThreadPool thread_pool(4);  // 병렬 비얽힘 연산을 위한 스레드 풀
    std::vector<std::vector<std::pair<std::vector<unsigned>, const fp_type*>>> non_entangling_batches;  
    std::vector<std::future<void>> tasks;  
    bool is_entangling = false;

    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
    for (auto q : qs) {
        printf("%u ", q);
    }
    printf("\n");
    
    // 비얽힘 게이트를 배치로 그룹화 (is_entangling 플래그로 구분)
    if (qs.size() == 1 || qs[0] <= 4) {  
        non_entangling_batches.push_back({{qs, matrix}});
    } else {
        is_entangling = true;
    }

    // 1. 비얽힘 게이트를 **배치 단위**로 병렬 처리
    if (!non_entangling_batches.empty()) {
        tasks.push_back(thread_pool.AddTask([this, &non_entangling_batches, &state]() {
            for (const auto& batch : non_entangling_batches) {
                for (const auto& gate : batch) {
                    const auto& qs_batch = gate.first;
                    const auto* matrix_batch = gate.second;

                    switch (qs_batch.size()) {
                        case 1:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs_batch, matrix_batch, state);
                            break;
                        case 2:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs_batch, matrix_batch, state);
                            break;
                        case 3:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs_batch, matrix_batch, state);
                            break;
                        case 4:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs_batch, matrix_batch, state);
                            break;
                        case 5:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs_batch, matrix_batch, state);
                            break;
                        case 6:
                            const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs_batch, matrix_batch, state);
                            break;
                        default:
                            printf("KCJ: Unsupported non-entangling gate size.\n");
                            break;
                    }
                }
            }
        }));
    }

    // 2. 병렬 처리 완료 대기
    for (auto& task : tasks) {
        task.get();
    }

    // 3. 얽힘 게이트 하이브리드 처리 (큐빗 수에 따라 병렬화 여부 결정)
    if (is_entangling) {
        printf("KCJ: Applying entangling gates conditionally.\n");
        
        // 병렬화 기준: 큐빗 수가 3 이하인 경우만 병렬 수행
        if (qs.size() <= 3) {
            std::vector<std::future<void>> entangling_tasks;
            entangling_tasks.push_back(thread_pool.AddTask([this, &qs, matrix, &state]() {
                switch (qs.size()) {
                    case 2:
                        const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state);
                        break;
                    case 3:
                        const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state);
                        break;
                    default:
                        break;
                }
            }));
            for (auto& task : entangling_tasks) {
                task.get();
            }
        } else {
            // 큐빗 수가 많을 경우 순차 처리
            switch (qs.size()) {
                case 4:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state);
                    break;
                case 5:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state);
                    break;
                case 6:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state);
                    break;
                default:
                    printf("KCJ: Unsupported entangling gate size.\n");
                    break;
            }
        }
    }

    printf("KCJ: ApplyGate completed successfully.\n");
}
*/
//KCJ: Testcase 1 
/*
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(task_mutex_);
                        cv_.wait(lock, [this]() { return !tasks_.empty() || stop_; });

                        if (tasks_.empty() && stop_) {
                            return;
                        }

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                    {
                        std::lock_guard<std::mutex> lock(task_mutex_);
                        --pending_tasks_;
                        if (pending_tasks_ == 0 && tasks_.empty()) {
                            cv_.notify_all();
                        }
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void AddTask(const std::function<void()>& task) {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            tasks_.push(task);
            ++pending_tasks_;
        }
        cv_.notify_one();
    }

    void WaitForAllTasks() {
        std::unique_lock<std::mutex> lock(task_mutex_);
        cv_.wait(lock, [this]() { return pending_tasks_ == 0 && tasks_.empty(); });
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex task_mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> pending_tasks_{0};
    bool stop_;
};


void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
    static ThreadPool thread_pool(1); 

    thread_pool.AddTask([this, qs, matrix, &state]() {
        unsigned num_qubits = state.num_qubits();
        const_cast<SimulatorCUDA*>(this)->FreeAndAllocateMemory(state, state.num_qubits());
    
        if (qs.size() == 0) {
            const_cast<SimulatorCUDA*>(this)->template ApplyGateH<0>(qs, matrix, state);
        } 
        
        else if (qs[0] > 4) {
            switch (qs.size()) {
            case 1: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<1>(qs, matrix, state); break;
            case 2: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<2>(qs, matrix, state); break;
            case 3: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<3>(qs, matrix, state); break;
            case 4: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<4>(qs, matrix, state); break;
            case 5: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<5>(qs, matrix, state); break;
            case 6: const_cast<SimulatorCUDA*>(this)->template ApplyGateH<6>(qs, matrix, state); break;
            default: printf("Unsupported qs.size() for ApplyGateH.\n"); break;
            }
        } else {
            switch (qs.size()) {
            case 1: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<1>(qs, matrix, state); break;
            case 2: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<2>(qs, matrix, state); break;
            case 3: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<3>(qs, matrix, state); break;
            case 4: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<4>(qs, matrix, state); break;
            case 5: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<5>(qs, matrix, state); break;
            case 6: const_cast<SimulatorCUDA*>(this)->template ApplyGateL<6>(qs, matrix, state); break;
            default: printf("Unsupported qs.size() for ApplyGateL.\n"); break;
            }
        }
    });

    thread_pool.WaitForAllTasks(); 
}

*/




class StorageManager {
public:
    // 데이터 오프로드
    static void OffloadData(const std::string& file_name, const void* data, size_t size, bool append = false) {
        std::ofstream ofs(file_name, append ? (std::ios::binary | std::ios::app) : std::ios::binary);
        if (!ofs) {
            std::cerr << "KCJ: Error - Failed to open file for writing: " << file_name << std::endl;
            throw std::runtime_error("KCJ: File open error");
        }
        ofs.write(reinterpret_cast<const char*>(data), size);
        if (!ofs.good()) {
            std::cerr << "KCJ: Error - Failed to write data to file: " << file_name << std::endl;
            throw std::runtime_error("KCJ: File write error");
        }
        ofs.close();
        printf("KCJ: Offloaded data to %s (%llu bytes).\n", file_name.c_str(), size);
    }

    // 데이터 리로드 (파일 크기 검사 추가)
    static void ReloadData(const std::string& file_name, void* data, size_t size, size_t offset = 0) {
        std::ifstream ifs(file_name, std::ios::binary | std::ios::ate);
        if (!ifs) {
            std::cerr << "KCJ: Error - Failed to open file for reading: " << file_name << std::endl;
            throw std::runtime_error("KCJ: File open error");
        }

        size_t file_size = ifs.tellg();
        if (file_size < size + offset) {
            std::cerr << "KCJ: Error - Insufficient data in file for reloading state. File size: " 
                      << file_size << " Requested: " << size + offset << std::endl;
            throw std::runtime_error("KCJ: Insufficient file size");
        }

        ifs.seekg(offset, std::ios::beg);
        ifs.read(reinterpret_cast<char*>(data), size);
        if (!ifs.good()) {
            std::cerr << "KCJ: Error - Failed to read data from file: " << file_name << std::endl;
            throw std::runtime_error("KCJ: File read error");
        }
        ifs.close();
        printf("KCJ: Reloaded data from %s (%llu bytes).\n", file_name.c_str(), size);
    }
};

// **State 오프로드 (CUDA 에러 체크 추가)**
// **State 오프로드 (CUDA 에러 체크 및 메모리 부족 방지 추가)**
void OffloadState(State& state) {
    printf("KCJ: Offloading state to storage...\n");
    size_t total_size = MinSize(state.num_qubits());
    std::string file_name = "/pscratch/sd/s/sgkim/kcj/qsim/storage_offload_data/state_backup.bin";

    // GPU 메모리 검사
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("KCJ DEBUG: Free Memory: %llu, Total Memory: %llu\n", free_mem, total_mem);

/*
    if (total_size * sizeof(fp_type) > free_mem) {
        printf("KCJ ERROR: Not enough GPU memory to offload state!\n");
        throw std::runtime_error("KCJ: GPU memory insufficient for offload.");
    }
*/

    // CPU 메모리 할당
    fp_type* host_buffer = nullptr;
    try {
        host_buffer = new fp_type[total_size];
    } catch (const std::bad_alloc&) {
        printf("KCJ ERROR: Host memory allocation failed.\n");
        throw std::runtime_error("KCJ: Host memory allocation failed.");
    }
    cudaDeviceSynchronize(); 
    
    // **CUDA 메모리 복사 및 동기화**
    cudaError_t cudaStatus = cudaMemcpy(host_buffer, state.get(), total_size * sizeof(fp_type), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
/*
    if (cudaStatus != cudaSuccess) {
        printf("KCJ ERROR: CUDA memcpy failed during offload: %s\n", cudaGetErrorString(cudaStatus));
        delete[] host_buffer;
        throw std::runtime_error("KCJ: CUDA memcpy error during offload");
    }
*/
    // **파일에 데이터 오프로드**
    try {
        StorageManager::OffloadData(file_name, host_buffer, total_size * sizeof(fp_type));
    } catch (const std::exception& e) {
        printf("KCJ ERROR: Error while writing to disk: %s\n", e.what());
        delete[] host_buffer;
        throw;
    }

    delete[] host_buffer;
    printf("KCJ: OffloadState complete. State data saved to: %s\n", file_name.c_str());
}


// **State 리로딩 (CUDA 에러 체크 추가)**
void ReloadState(State& state) {
    printf("KCJ: Reloading state from storage...\n");
    size_t total_size = MinSize(state.num_qubits());
    std::string file_name = "/pscratch/sd/s/sgkim/kcj/qsim/storage_offload_data/state_backup.bin";

    // CPU 메모리 할당
    fp_type* host_buffer = nullptr;
    try {
        host_buffer = new fp_type[total_size];
    } catch (const std::bad_alloc&) {
        printf("KCJ ERROR: Host memory allocation failed during reloading.\n");
        throw std::runtime_error("KCJ: Host memory allocation failed during reloading.");
    }

    // 파일에서 데이터 로딩
    try {
        StorageManager::ReloadData(file_name, host_buffer, total_size * sizeof(fp_type));
    } catch (const std::exception& e) {
        printf("KCJ ERROR: Error reading from disk: %s\n", e.what());
        delete[] host_buffer;
        throw;
    }

    // **데이터 GPU 메모리로 복사 및 동기화**
    cudaError_t cudaStatus = cudaMemcpy(state.get(), host_buffer, total_size * sizeof(fp_type), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
/*
    if (cudaStatus != cudaSuccess) {
        printf("KCJ ERROR: CUDA memcpy error during reloading: %s\n", cudaGetErrorString(cudaStatus));
        delete[] host_buffer;
        throw std::runtime_error("KCJ: CUDA memcpy error during reloading.");
    }
*/
    delete[] host_buffer;
    printf("KCJ: ReloadState complete. State reloaded from: %s\n", file_name.c_str());
}


/*

// Original
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .
       // auto state_ptr = std::make_shared<State>(state); // State를 스마트 포인터로 변환

    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
        for (auto q : qs) {
          printf("%u ", q);
        }
        printf("\n");
    
    if (qs.size() == 0) {
      const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state);
    } else if (qs[0] > 4) {
      printf("KCJ: ApplyGate is run !: qs: %u, matrix: %p\n", qs, matrix);
      switch (qs.size()) {
      case 1:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state);
        break;
      case 2:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state);
        break;
      case 3:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state);
        break;
      case 4:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state);
        break;
      case 5:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state);
        break;
      case 6:
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      } //KCJ: Running Else-case 2, 4 
    } else {
      switch (qs.size()) {
      case 1:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state);
        break;
      case 2:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state);
        break;
      case 3:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state);
        break;
      case 4:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state);
        break;
      case 5:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state);
        break;
      case 6:
        const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state);
        break;
      default:
        // Not implemented.
        break;
      }
    }
  }

*/

/*
void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
    for (auto q : qs) {
        printf("%u ", q);
    }
    printf("\n");

    size_t num_gpus = state_parts_.size();
    static size_t next_gpu = 0;

    if (qs.size() <= 3) {
        // 라운드로빈 방식으로 GPU 선택
        size_t assigned_gpu = next_gpu;
        next_gpu = (next_gpu + 1) % num_gpus;

        cudaSetDevice(assigned_gpu);
        printf("KCJ DEBUG: Assigned GPU %lu for this task.\n", assigned_gpu);

        if (qs.size() == 0) {
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state, assigned_gpu);
        } else if (qs[0] > 4) {
            switch (qs.size()) {
            case 1:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state, assigned_gpu);
                break;
            case 2:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state, assigned_gpu);
                break;
            case 3:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state, assigned_gpu);
                break;
            case 4:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state, assigned_gpu);
                break;
            case 5:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state, assigned_gpu);
                break;
            case 6:
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state, assigned_gpu);
                break;
            default:
                printf("KCJ WARNING: Unsupported qs.size in ApplyGateH.\n");
                break;
            }
        } else {
            switch (qs.size()) {
            case 1:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state, assigned_gpu);
                break;
            case 2:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state, assigned_gpu);
                break;
            case 3:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state, assigned_gpu);
                break;
            case 4:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state, assigned_gpu);
                break;
            case 5:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state, assigned_gpu);
                break;
            case 6:
                const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state, assigned_gpu);
                break;
            default:
                printf("KCJ WARNING: Unsupported qs.size in ApplyGateL.\n");
                break;
            }
        }
    } else {
        // P2P 접근을 활용하여 모든 GPU에서 연산 수행
        printf("KCJ DEBUG: Large qs.size detected, using P2P approach.\n");
        for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            cudaSetDevice(gpu_id);
            printf("KCJ DEBUG: Performing operation on GPU %lu.\n", gpu_id);

            if (qs.size() == 0) {
                const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state, gpu_id);
            } else if (qs[0] > 4) {
                switch (qs.size()) {
                case 1:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state, gpu_id);
                    break;
                case 2:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state, gpu_id);
                    break;
                case 3:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state, gpu_id);
                    break;
                case 4:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state, gpu_id);
                    break;
                case 5:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state, gpu_id);
                    break;
                case 6:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state, gpu_id);
                    break;
                default:
                    printf("KCJ WARNING: Unsupported qs.size in ApplyGateH.\n");
                    break;
                }
            } else {
                switch (qs.size()) {
                case 1:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state, gpu_id);
                    break;
                case 2:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state, gpu_id);
                    break;
                case 3:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state, gpu_id);
                    break;
                case 4:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state, gpu_id);
                    break;
                case 5:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state, gpu_id);
                    break;
                case 6:
                    const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state, gpu_id);
                    break;
                default:
                    printf("KCJ WARNING: Unsupported qs.size in ApplyGateL.\n");
                    break;
                }
            }
        }
    }
}
*/


/*
//TestCase 1 : MultiGPU 

void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
    for (auto q : qs) {
        printf("%u ", q);
    }
    printf("\n");

    size_t num_gpus = state_parts_.size();
    static size_t next_gpu = 0;

    // Round-robin 방식으로 GPU 선택 (모듈 연산으로 순환)
    size_t assigned_gpu = next_gpu;
    next_gpu = (next_gpu + 1) % num_gpus;
    
    cudaSetDevice(assigned_gpu);
    printf("KCJ DEBUG: Assigned GPU %lu for this task.\n", assigned_gpu);

    if (qs.size() == 0) {
        const_cast<SimulatorCUDA*>(this)->ApplyGateH<0>(qs, matrix, state, assigned_gpu);
    } else if (qs[0] > 4) {
        switch (qs.size()) {
        case 1:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<1>(qs, matrix, state, assigned_gpu);
            break;
        case 2:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<2>(qs, matrix, state, assigned_gpu);
            break;
        case 3:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<3>(qs, matrix, state, assigned_gpu);
            break;
        case 4:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<4>(qs, matrix, state, assigned_gpu);
            break;
        case 5:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<5>(qs, matrix, state, assigned_gpu);
            break;
        case 6:
            const_cast<SimulatorCUDA*>(this)->ApplyGateH<6>(qs, matrix, state, assigned_gpu);
            break;
        default:
            printf("KCJ WARNING: Unsupported qs.size in ApplyGateH.\n");
            break;
        }
    } else {
        switch (qs.size()) {
        case 1:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<1>(qs, matrix, state, assigned_gpu);
            break;
        case 2:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<2>(qs, matrix, state, assigned_gpu);
            break;
        case 3:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<3>(qs, matrix, state, assigned_gpu);
            break;
        case 4:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<4>(qs, matrix, state, assigned_gpu);
            break;
        case 5:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<5>(qs, matrix, state, assigned_gpu);
            break;
        case 6:
            const_cast<SimulatorCUDA*>(this)->ApplyGateL<6>(qs, matrix, state, assigned_gpu);
            break;
        default:
            printf("KCJ WARNING: Unsupported qs.size in ApplyGateL.\n");
            break;
        }
    }
}

*/


struct IndexMapping {
    size_t global_index; // 글로벌 인덱스
    size_t gpu_id;       // 해당 GPU ID
    size_t local_index;  // GPU 내 로컬 statespace에서의 위치
};

IndexMapping MapGlobalToLocal(size_t global_index, size_t total_size, size_t num_gpus) const {
    size_t size_per_gpu = total_size / num_gpus; // GPU당 Statespace 크기
    size_t gpu_id = (global_index / size_per_gpu) % num_gpus; // GPU ID
    size_t local_index = global_index % size_per_gpu; // GPU 내 로컬 인덱스

    printf("DEBUG: Mapping - global_index = %lu, gpu_id = %lu, local_index = %lu\n",
           global_index, gpu_id, local_index);

    return {global_index, gpu_id, local_index};
}

void PrintBinary(size_t value, size_t num_bits) const {
    for (int i = num_bits - 1; i >= 0; --i) {
        printf("%d", (value >> i) & 1);
    }
}



        

std::vector<size_t> CalculateAffectedIndices(
    size_t num_qubits, const std::vector<unsigned>& qs,
    size_t num_gpus, size_t size_per_gpu) const {
    size_t num_indices = 1ULL << qs.size(); // 영향을 받는 조합의 수

    std::vector<size_t> indices_global;
    indices_global.reserve(num_indices); // 메모리 미리 할당

    // 비트 위치 계산
    std::vector<size_t> bit_positions;
    for (unsigned q : qs) {
        bit_positions.push_back(1ULL << (q + 1)); // ✅ 글로벌 인덱스를 xss 기준과 동일하게 조정
    }

    // 글로벌 인덱스 생성
    for (size_t i = 0; i < num_indices; ++i) {
        size_t index = 0;
        for (size_t j = 0; j < qs.size(); ++j) {
            if ((i >> j) & 1) { // i의 j번째 비트가 1인지 확인
                index |= bit_positions[j]; // 해당 비트를 추가
            }
        }

        size_t gpu_id = index / size_per_gpu;  // ✅ xss 기준으로 GPU ID 계산
        if (gpu_id >= num_gpus) {
            printf("ERROR: Calculated GPU ID %lu out of range!\n", gpu_id);
            continue; // 잘못된 GPU ID면 무시
        }

        size_t local_index = index - (gpu_id * size_per_gpu);

        printf("DEBUG: Global index %lu (xss-based) -> GPU %lu, Local index %lu\n", index, gpu_id, local_index);

        indices_global.push_back(index); 
    }

    // 디버깅 출력
    printf("DEBUG: Affected Indices (qs.size = %lu, total = %lu): ", qs.size(), indices_global.size());
    for (size_t j = 0; j < std::min(size_t(10), indices_global.size()); ++j) {
        printf("%lu :", indices_global[j]);
    }
    if (indices_global.size() > 10) {
        printf("total %lu indices", indices_global.size());
    }
    printf("\n");

    return indices_global;
}

void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
    printf("KCJ: ApplyGate - qs.size: %llu\n", qs.size());
    printf("KCJ: ApplyGate - qs: ");
    for (auto q : qs) {
        printf("%u ", q);
    }
    printf("\n");

     auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();

    unsigned num_qubits = 29;
    size_t num_gpus = multi_gpu_pointers.size();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = total_size / num_gpus;

    printf("DEBUG: GPU statespace ranges:\n");
    for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        size_t start_index = gpu_id * size_per_gpu;
        size_t end_index = start_index + size_per_gpu - 1;
        printf("GPU %lu: [%lu, %lu]\n", gpu_id, start_index, end_index);
    }

    std::vector<size_t> affected_indices = CalculateAffectedIndices(num_qubits, qs, num_gpus, size_per_gpu);

    size_t gpu_task_count[num_gpus] = {0};
    std::map<size_t, std::vector<size_t>> gpu_indices_map;

    for (size_t global_index : affected_indices) {
        size_t gpu_id = global_index / size_per_gpu;  
        size_t local_index = global_index - (gpu_id * size_per_gpu); 

        gpu_task_count[gpu_id]++;
        gpu_indices_map[gpu_id].push_back(local_index);
    }

    for (size_t i = 0; i < num_gpus; ++i) {
        printf("GPU %lu: %lu tasks\n", i, gpu_task_count[i]);
    }

    for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        if (gpu_task_count[gpu_id] == 0) continue;

        printf("DEBUG: Processing GPU %lu with %lu tasks\n", gpu_id, gpu_task_count[gpu_id]);

        cudaSetDevice(gpu_id);
        //cudaDeviceSynchronize();

        for (const auto& local_index : gpu_indices_map[gpu_id]) {
            printf("DEBUG: Local index %lu on GPU %lu\n", local_index, gpu_id);
        }

        //cudaDeviceSynchronize();
            //std::vector<size_t> local_indices = gpu_indices_map[gpu_id];
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
       // cudaDeviceSynchronize();
    }
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
  


/* 
//Stream version

template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 29;
    printf("KCJ DEBUG: ApplyGateH called on GPU %lu - num_qubits: %u, qs.size: %lu\n", gpu_id, num_qubits, qs.size());

    cudaSetDevice(gpu_id);

    cudaStream_t stream;
    cudaStreamCreate(&stream);  //  GPU 별 독립적 CUDA 스트림 생성

    IndicesH<G> h_i(h_ws);
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (multi_gpu_pointers.size() == 1) ? total_size : total_size / multi_gpu_pointers.size();
    size_t global_offset = (multi_gpu_pointers.size() == 1) ? 0 : gpu_id * size_per_gpu;

    printf("KCJ: ApplyGateH: total_size=%lu bytes, size_per_gpu=%lu bytes, global_offset=%lu bytes, gpu_id=%lu, num_gpus=%lu\n", 
           total_size, size_per_gpu, global_offset, gpu_id, multi_gpu_pointers.size());

    GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);

    
    cudaMemset(d_ws_list_[gpu_id], 0, h_i.buf_size);

    
    //  비동기 메모리 복사 (스트림 사용)
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaError_t copy_err = cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        printf("ERROR: cudaMemcpyAsync failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(copy_err));
        return;
    }

    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    IndicesH<G> d_i(d_ws_list_[gpu_id]);

       // 메모리 정렬 확인
    if ((uintptr_t)d_ws_list_[gpu_id] % alignof(fp_type) != 0) {
        printf("ERROR: d_ws_list is not properly aligned for fp_type.\n");
    }
    if ((uintptr_t)multi_gpu_pointers[gpu_id] % alignof(fp_type) != 0) {
        printf("ERROR: multi_gpu_pointers[%lu] is not properly aligned for fp_type.\n", gpu_id);
    }

    cudaPointerAttributes attributes;
    cudaError_t attr_err = cudaPointerGetAttributes(&attributes, multi_gpu_pointers[gpu_id]);
    if (attr_err != cudaSuccess) {
        printf("ERROR: cudaPointerGetAttributes failed for multi_gpu_pointers[%lu]: %s\n", gpu_id, cudaGetErrorString(attr_err));
    } else if (attributes.type != cudaMemoryTypeDevice) {
        printf("ERROR: multi_gpu_pointers[%lu] is not pointing to device memory: %p\n", gpu_id, multi_gpu_pointers[gpu_id]);
    } else {
        printf("DEBUG: multi_gpu_pointers[%lu] is valid and points to device memory: %p\n", gpu_id, multi_gpu_pointers[gpu_id]);
    }
    
    if ((uintptr_t)multi_gpu_pointers[gpu_id] % alignof(fp_type) != 0) {
        printf("ERROR: multi_gpu_pointers[%lu] is not properly aligned for fp_type.\n", gpu_id);
    }
    //cudaPointerAttributes attributes;

    cudaPointerAttributes attributes1;
    cudaPointerGetAttributes(&attributes1, d_ws_list_[gpu_id]);
    printf("DEBUG: d_ws_list: %p, is %s memory\n", 
           d_ws_list_[gpu_id], 
           (attributes1.type == cudaMemoryTypeDevice) ? "Device" : "Host");
    
    cudaPointerAttributes attributes2;
    cudaPointerGetAttributes(&attributes2, d_i.xss);
    printf("DEBUG: d_i.xss: %p, is %s memory\n", 
           d_i.xss, 
           (attributes2.type == cudaMemoryTypeDevice) ? "Device" : "Host");
    
    cudaPointerAttributes attributes3;
    cudaPointerGetAttributes(&attributes3, d_i.ms);
    printf("DEBUG: d_i.ms: %p, is %s memory\n", 
           d_i.ms, 
           (attributes3.type == cudaMemoryTypeDevice) ? "Device" : "Host");

    

    printf("DEBUG: Launching kernel on GPU %lu with %u blocks, %u threads.\n", gpu_id, blocks, threads);


    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_id);
    printf("DEBUG: Device %d - Max threads per block: %d, Max grid size: (%d, %d, %d), Shared memory per block: %zu bytes\n",
           gpu_id, deviceProp.maxThreadsPerBlock,
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2],
           deviceProp.sharedMemPerBlock);

    
        // 모든 포인터 값 확인
    if (multi_gpu_pointers[gpu_id] == nullptr) {
        printf("ERROR: multi_gpu_pointers[%lu] is NULL!\n", gpu_id);
        return;
    }
    if (d_ws_list_[gpu_id] == nullptr) {
        printf("ERROR: d_ws_list_[%lu] is NULL!\n", gpu_id);
        return;
    }
    if (d_i.xss == nullptr) {
        printf("ERROR: d_i.xss is NULL!\n");
        return;
    }
    if (d_i.ms == nullptr) {
        printf("ERROR: d_i.ms is NULL!\n");
        return;
    }
    
        // cudaPointerGetAttributes()로 유효한 GPU 메모리인지 확인
    cudaPointerAttributes attributes4;
    cudaError_t attr_err4 = cudaPointerGetAttributes(&attributes4, multi_gpu_pointers[gpu_id]);
    if (attr_err != cudaSuccess) {
        printf("ERROR: cudaPointerGetAttributes failed for multi_gpu_pointers[%lu]: %s\n", gpu_id, cudaGetErrorString(attr_err4));
        return;
    } else if (attributes4.type != cudaMemoryTypeDevice) {
        printf("ERROR: multi_gpu_pointers[%lu] is not pointing to device memory!\n", gpu_id);
        return;
    }


    printf("DEBUG: Kernel arguments before launch:\n");
    printf("  v0: %p\n", (void*)d_ws_list_[gpu_id]);
    printf("  xss: %p\n", (void*)d_i.xss);
    printf("  ms: %p\n", (void*)d_i.ms);
    printf("  multi_gpu_pointers[%lu]: %p\n", gpu_id, (void*)multi_gpu_pointers[gpu_id]);
    printf("  global_offset: %lu\n", global_offset);
    printf("  size_per_gpu: %lu\n", size_per_gpu);
    
    //unsigned shared_memory_size = 49152;

    cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
        printf("ERROR: CUDA error before kernel launch on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err_before));
    }

    

    
 

    //  GPU 개별 CUDA 스트림을 사용한 비동기 실행
    ApplyGateH_Kernel<G><<<blocks, threads, 0, stream>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        multi_gpu_pointers[gpu_id],  
        global_offset,                    
        size_per_gpu                      
    );

    cudaStreamSynchronize(stream);  //  스트림 동기화 (GPU 개별 실행 보장)
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateH failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    //  GPU별 상태 벡터 확인
    fp_type* host_check = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpyAsync(host_check, multi_gpu_pointers[gpu_id], size_per_gpu, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  

    printf("DEBUG: State vector on GPU %lu:\n", gpu_id);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;

    cudaStreamDestroy(stream);  //  스트림 제거
    printf("KCJ DEBUG: ApplyGateH executed successfully on GPU %lu.\n", gpu_id);
}


template <unsigned G>
void ApplyGateL(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 29;
    printf("KCJ DEBUG: ApplyGateL called on GPU %lu - num_qubits: %u, qs.size: %lu\n", gpu_id, num_qubits, qs.size());

    cudaSetDevice(gpu_id);

    cudaStream_t stream;
    cudaStreamCreate(&stream);  //  GPU 별 독립적 CUDA 스트림 생성

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (multi_gpu_pointers.size() == 1) ? total_size : total_size / multi_gpu_pointers.size();
    size_t global_offset = (multi_gpu_pointers.size() == 1) ? 0 : gpu_id * size_per_gpu;

    printf("KCJ: ApplyGateL: total_size=%lu bytes, size_per_gpu=%lu bytes, global_offset=%lu bytes, gpu_id=%lu, num_gpus=%lu\n", 
           total_size, size_per_gpu, global_offset, gpu_id, multi_gpu_pointers.size());

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL_m(num_qubits, qs, h_i, global_offset);

    // ✅ 비동기 메모리 복사 (스트림 사용)
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice, stream);

    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;

    IndicesL<G> d_i(d_ws_list_[gpu_id]);

    printf("DEBUG: Launching kernel on GPU %lu with %u blocks, %u threads.\n", gpu_id, blocks, threads);

    // ✅ GPU 개별 CUDA 스트림을 사용한 비동기 실행
    ApplyGateL_Kernel<G><<<blocks, threads, 0, stream>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        d_i.qis,                         
        d_i.tis,                         
        1 << num_effective_qs,           
        (fp_type*) multi_gpu_pointers[gpu_id],  
        global_offset,                    
        size_per_gpu                      
    );

    cudaStreamSynchronize(stream);  // ✅ 스트림 동기화 (GPU 개별 실행 보장)
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateL failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    // ✅ GPU별 상태 벡터 확인
    fp_type* host_check = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpyAsync(host_check, multi_gpu_pointers[gpu_id], size_per_gpu, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  

    printf("DEBUG: State vector on GPU %lu:\n", gpu_id);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;

    cudaStreamDestroy(stream);  // ✅ 스트림 제거
    printf("KCJ DEBUG: ApplyGateL executed successfully on GPU %lu.\n", gpu_id);
}

*/


//KCJ: MultiGPU Ver1.- Enable P2P;
/* 잘되는 P2P
template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 29;
    printf("KCJ DEBUG: ApplyGateH called on GPU %lu - num_qubits: %u, qs.size: %lu\n", gpu_id, num_qubits, qs.size());

    cudaSetDevice(gpu_id);

    IndicesH<G> h_i(h_ws);

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (multi_gpu_pointers.size() == 1) ? total_size : total_size / multi_gpu_pointers.size();
    size_t global_offset = (multi_gpu_pointers.size() == 1) ? 0 : gpu_id * size_per_gpu;

    printf("KCJ: ApplyGateH: total_size=%lu bytes, size_per_gpu=%lu bytes, global_offset=%lu bytes, gpu_id=%lu, num_gpus=%lu\n", 
           total_size, size_per_gpu, global_offset, gpu_id, multi_gpu_pointers.size());

    GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);
    
    // ✅ 추가: 메모리 정합성 체크
    for (size_t i = 0; i < multi_gpu_pointers.size(); ++i) {
        printf("DEBUG: multi_gpu_pointers[%lu]: %p\n", i, multi_gpu_pointers[i]);
    }

    
    // ✅ 추가: P2P 데이터 공유 (GPU 0 → GPU 1)
    if (gpu_id > 0) {  
        printf("DEBUG: Copying state vector from GPU 0 to GPU %lu\n", gpu_id);
        cudaMemcpyPeer(multi_gpu_pointers[gpu_id], gpu_id, multi_gpu_pointers[0], 0, size_per_gpu);
        cudaDeviceSynchronize();
    }

        
    // ✅ 추가: 커널 실행 전 상태 벡터 확인
    fp_type* host_check1 = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check1, multi_gpu_pointers[gpu_id], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: State vector on GPU %lu before kernel execution:\n", gpu_id);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check1[i]);
    }
    delete[] host_check1;

    // ✅ 행렬 데이터를 호스트에서 디바이스로 복사
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    printf("DEBUG: Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",
           h_i.matrix_size, h_ws, d_ws_list_[gpu_id]);



    // ✅ 블록 및 쓰레드 설정
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);
 
    IndicesH<G> d_i(d_ws_list_[gpu_id]);

    printf("KCJ Before Kernel launch- GPU %lu - d_ws_list: %p, d_i.xss: %p, d_i.ms: %p, state_part: %p\n",
           gpu_id, d_ws_list_[gpu_id], d_i.xss, d_i.ms, multi_gpu_pointers[gpu_id]);

    printf("DEBUG: Kernel arguments before launch:\n");
    printf("  v0: %p\n", (void*)d_ws_list_[gpu_id]);
    printf("  xss: %p / %lu\n", (void*)d_i.xss, d_i.xss);
    printf("  ms: %p / %lu\n", (void*)d_i.ms, d_i.ms);
    printf("  multi_gpu_pointers[%lu]: %p\n", gpu_id, (void*)multi_gpu_pointers[gpu_id]);
    printf("  global_offset: %lu\n", global_offset);
    printf("  size_per_gpu: %lu\n", size_per_gpu);
    

    // ✅ Kernel 실행
    printf("DEBUG: Preparing to launch ApplyGateH_Kernel\n");
    ApplyGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        (fp_type*) multi_gpu_pointers[gpu_id],  
        global_offset,                    
        size_per_gpu                      
    );

    cudaDeviceSynchronize();

    // ✅ 커널 실행 후 CUDA 오류 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateH failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    // ✅ 커널 실행 후 상태 벡터 확인
    fp_type* host_check = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check, multi_gpu_pointers[gpu_id], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: State vector on GPU %lu after kernel execution:\n", gpu_id);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;


    if (gpu_id == 1) {  
        printf("DEBUG: Copying back state vector from GPU %lu to GPU 0\n", gpu_id);
        cudaMemcpyPeer(multi_gpu_pointers[0], 0, multi_gpu_pointers[gpu_id], gpu_id, size_per_gpu);
        cudaDeviceSynchronize();
    }


    printf("KCJ DEBUG: ApplyGateH executed successfully on GPU %lu.\n", gpu_id);
}
*/




template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 29;
    printf("KCJ DEBUG: ApplyGateH called on GPU %lu - num_qubits: %u, qs.size: %lu\n", gpu_id, num_qubits, qs.size());

    cudaSetDevice(gpu_id);
//    cudaDeviceSynchronize();

    IndicesH<G> h_i(h_ws);

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (multi_gpu_pointers.size() == 1) ? total_size : total_size / multi_gpu_pointers.size();
    size_t global_offset = (multi_gpu_pointers.size() == 1) ? 0 : gpu_id * size_per_gpu;


    printf("KCJ: ApplyGateH: total_size=%lu bytes, size_per_gpu=%lu bytes, global_offset=%lu bytes, gpu_id=%lu, num_gpus=%lu\n", 
           total_size, size_per_gpu, global_offset, gpu_id, multi_gpu_pointers.size());
    size_t free_mem = 0, total_mem = 0;
    
    //GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);
    GetIndicesH_m(num_qubits, qs, qs.size(), h_i, global_offset);
    //  추가: 메모리 정합성 체크
    for (size_t i = 0; i < multi_gpu_pointers.size(); ++i) {
        printf("DEBUG: multi_gpu_pointers[%lu]: %p\n", i, multi_gpu_pointers[i]);
    }
    

     /*
    //  추가: 커널 실행 전 상태 벡터 확인
    fp_type* host_check1 = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check1, multi_gpu_pointers[0], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: State vector on GPU %lu before kernel execution:\n", 0);
    for (int i = 0; i < std::min(size_t(5), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check1[i]);
    }
    delete[] host_check1;
*/
    
    //  행렬 데이터를 호스트에서 디바이스로 복사
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    printf("DEBUG: Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",
           h_i.matrix_size, h_ws, d_ws_list_[gpu_id]);


    
    //  블록 및 쓰레드 설정
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned long long size = 1ULL << n;
    unsigned threads = 64U;
    unsigned long long max_blocks = (1ULL << 30);
    unsigned long long blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    
    /*
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);
    */
    
    IndicesH<G> d_i(d_ws_list_[gpu_id]);

    printf("KCJ Before Kernel launch- GPU %lu - d_ws_list: %p, d_i.xss: %p, d_i.ms: %p, multi_gpu_pointers: %p\n",
           gpu_id, d_ws_list_[gpu_id], d_i.xss, d_i.ms, multi_gpu_pointers[gpu_id]);

    printf("DEBUG: Kernel arguments before launch:\n");
    printf("  v0: %p\n", (void*)d_ws_list_[gpu_id]);
    printf("  xss: %p / %lu\n", (void*)d_i.xss, d_i.xss);
    printf("  ms: %p / %lu\n", (void*)d_i.ms, d_i.ms);
    printf("  multi_gpu_pointers[%lu]: %p\n", gpu_id, (void*)multi_gpu_pointers[gpu_id]);
    printf("  global_offset: %lu\n", global_offset);
    printf("  size_per_gpu: %lu\n", size_per_gpu);
    

    //  Kernel 실행
    printf("DEBUG: Preparing to launch ApplyGateH_Kernel\n");
    ApplyGateH_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        (fp_type*) multi_gpu_pointers[gpu_id],  
        global_offset,                    
        size_per_gpu,
        gpu_id
    );

    printf("INFO: GPU %u - Free memory: %lu bytes, Total memory: %lu bytes\n", gpu_id, free_mem, total_mem);


   //cudaDeviceSynchronize();

    //  커널 실행 후 CUDA 오류 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateH failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

  /*
    //  커널 실행 후 상태 벡터 확인
    fp_type* host_check = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check, multi_gpu_pointers[0], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: State vector on GPU %lu after kernel execution:\n", 0);
    for (int i = 0; i < std::min(size_t(5), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;
*/
    printf("KCJ DEBUG: ApplyGateH executed successfully on GPU %lu.\n", gpu_id);
}



template <unsigned G>
void ApplyGateL(const std::vector<unsigned>& qs, const fp_type* matrix, State& state, size_t gpu_id) {
    unsigned num_qubits = 29;
    printf("KCJ DEBUG: ApplyGateL called on GPU %lu - num_qubits: %u, qs.size: %lu\n", gpu_id, num_qubits, qs.size());
    //gpu_id = 0;
    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    size_t total_size = MinSize(num_qubits);
    size_t size_per_gpu = (multi_gpu_pointers.size() == 1) ? total_size : total_size / multi_gpu_pointers.size();
    //size_per_gpu = total_size;
    size_t global_offset = (multi_gpu_pointers.size() == 1) ? 0 : gpu_id * size_per_gpu;

    
    cudaSetDevice(gpu_id);

    IndicesL<G> h_i(h_ws);
    size_t free_mem = 0, total_mem = 0;


    //auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);
    auto num_effective_qs = GetIndicesL_m(num_qubits, qs, h_i, global_offset);
    printf("KCJ DEBUG: Effective qubits for ApplyGateL on GPU %lu - num_effective_qs: %u\n", gpu_id, num_effective_qs);

    printf("KCJ: ApplyGateL: total_size=%lu bytes, size_per_gpu=%lu bytes, global_offset=%lu bytes, gpu_id=%lu, num_gpus=%lu\n", 
           total_size, size_per_gpu, global_offset, gpu_id, multi_gpu_pointers.size());


    /*
    //  커널 실행 전 상태 벡터 확인 (ApplyGateH와 동일)
    fp_type* host_check1 = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check1, multi_gpu_pointers[0], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: ApplyGateL State vector on GPU %lu before kernel execution:\n", 0);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check1[i]);
    }
    delete[] host_check1;
    */
    
    //  행렬 데이터를 호스트에서 디바이스로 복사
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_list_[gpu_id], h_ws, h_i.buf_size, cudaMemcpyHostToDevice);

    printf("DEBUG: ApplyGateL Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",
           h_i.matrix_size, h_ws, d_ws_list_[gpu_id]);

    //  블록 및 쓰레드 설정
    
    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned long long size = 1ULL << n;
    unsigned threads = 32;
    unsigned long long max_blocks = (1ULL << 30);
    unsigned long long blocks = std::min(max_blocks, std::max(1ULL, size / threads));
    

    /*
        unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;
    */
    IndicesL<G> d_i(d_ws_list_[gpu_id]);

    printf("KCJ DEBUG: GPU %lu - d_ws_list: %p, d_i.xss: %p, d_i.ms: %p, d_i.qis: %p, d_i.tis: %p, state_part: %p\n", gpu_id, (fp_type*) d_ws_list_[gpu_id], d_i.xss, d_i.ms, d_i.qis, d_i.tis, multi_gpu_pointers[gpu_id]);

    printf("DEBUG: Kernel arguments before launch:\n");
    printf("  v0: %p\n", (void*)d_ws_list_[gpu_id]);
    printf("  xss: %p / %lu\n", (void*)d_i.xss, d_i.xss);
    printf("  ms: %p / %lu\n", (void*)d_i.ms, d_i.ms);
    printf(" qis: %p / %lu\n", (void*)d_i.qis, d_i.qis);
    printf(" tis: %p / %lu\n", (void*)d_i.tis, d_i.tis);
    printf("  multi_gpu_pointers[%lu]: %p\n", gpu_id, (void*)multi_gpu_pointers[gpu_id]);
    printf("  global_offset: %lu\n", global_offset);
    printf("  size_per_gpu: %lu\n", size_per_gpu);

/*
    ApplyGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   
        d_i.xss,                         
        d_i.ms,                          
        d_i.qis,                         
        d_i.tis,                         
        1 << num_effective_qs,           
        (fp_type*) multi_gpu_pointers[gpu_id]            
    );
*/
    
    //  Kernel 실행
    printf("DEBUG: Preparing to launch ApplyGateL_Kernel\n");
    ApplyGateL_Kernel<G><<<blocks, threads>>>(
        (fp_type*) d_ws_list_[gpu_id],   // v0: 행렬 데이터
        d_i.xss,                         // xss: 인덱스 변환 테이블
        d_i.ms,                          // mss: 비트마스크 정보
        d_i.qis,                         // qis: 큐비트 정보
        d_i.tis,                         // tis: 큐비트 정보
        1 << num_effective_qs,           // esize: 활성 큐비트 수
        (fp_type*) multi_gpu_pointers[gpu_id],  // rstate: 현재 GPU의 상태 벡터
        global_offset,                    // global_offset: 현재 GPU의 메모리 시작점
        size_per_gpu,                      // size_per_gpu: 각 GPU가 담당하는 메모리 크기      
        gpu_id
        );
    

    //cudaDeviceSynchronize();
        printf("INFO: GPU %u - Free memory: %lu bytes, Total memory: %lu bytes\n", gpu_id, free_mem, total_mem);

    //  커널 실행 후 CUDA 오류 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KCJ ERROR: Kernel launch - ApplyGateL failed on GPU %lu: %s\n", gpu_id, cudaGetErrorString(err));
    }

    /* 
    //  커널 실행 후 상태 벡터 확인 (ApplyGateH와 동일)
    fp_type* host_check2 = new fp_type[size_per_gpu / sizeof(fp_type)];
    cudaMemcpy(host_check2, multi_gpu_pointers[0], size_per_gpu, cudaMemcpyDeviceToHost);
    printf("DEBUG: State vector on GPU %lu after kernel execution:\n", 0);
    for (int i = 0; i < std::min(size_t(10), size_per_gpu / sizeof(fp_type)); ++i) {
        printf("state[%d]: %f\n", i, host_check2[i]);
    }
    delete[] host_check2;
    */


    printf("KCJ DEBUG: ApplyGateL executed successfully on GPU %lu.\n", gpu_id);
}



 //KCJ: MultiGPU Ver1. 




//KCJ backup 

/*

//배치랑 같이 한거 
template <unsigned G>
void ApplyGateH(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) {
    unsigned num_qubits = state.num_qubits();
    printf("DEBUG: ApplyGateH called - num_qubits: %u, qs.size: %lu\n", num_qubits, qs.size());

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    // 1. 작업별 독립 메모리 할당
    fp_type* d_ws_local;
    cudaMalloc(&d_ws_local, h_i.buf_size);
    printf("DEBUG: Allocated device memory - size: %llu, address: %p\n", h_i.buf_size, d_ws_local);

    // 2. 메모리 병합 전송
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_local, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
    printf("DEBUG: Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",
           h_i.matrix_size, h_ws, d_ws_local);

    // 3. 커널 실행
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);
    printf("DEBUG: Kernel launch configuration - blocks: %u, threads: %u\n", blocks, threads);

    IndicesH<G> d_i(d_ws_local);
    printf("DEBUG: Kernel input indices - d_i.xss: %p, d_i.ms: %p, state.get(): %p\n", d_i.xss, d_i.ms, state.get());

    ApplyGateH_Kernel<G><<<blocks, threads>>>(reinterpret_cast<fp_type*>(d_ws_local), d_i.xss, d_i.ms, state.get());
    cudaDeviceSynchronize();
    printf("DEBUG: Kernel executed successfully for ApplyGateH.\n");

 // 4. 결과 검증 (선택 사항)
    fp_type* host_check = new fp_type[h_i.buf_size / sizeof(fp_type)];
    cudaMemcpy(host_check, d_ws_local, h_i.buf_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("DEBUG: Result validation [%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;


    // 5. 메모리 해제
    cudaFree(d_ws_local);
    printf("DEBUG: Device memory freed - address: %p\n", d_ws_local);
}


template <unsigned G>
void ApplyGateL(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) {
    unsigned num_qubits = state.num_qubits();
    printf("DEBUG: ApplyGateL called - num_qubits: %u, qs.size: %lu\n", num_qubits, qs.size());

    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);
    printf("DEBUG: Effective qubits for ApplyGateL - num_effective_qs: %u\n", num_effective_qs);

    // 1. 작업별 독립 메모리 할당
    fp_type* d_ws_local;
    cudaMalloc(&d_ws_local, h_i.buf_size);
    printf("DEBUG: Allocated device memory - size: %llu, address: %p\n", h_i.buf_size, d_ws_local);

    // 2. 메모리 병합 전송
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws_local, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
    printf("DEBUG: Copied matrix data to device memory - matrix size: %llu, h_ws: %p -> d_ws_local: %p\n",
           h_i.matrix_size, h_ws, d_ws_local);

    // 3. 커널 실행
    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;
    printf("DEBUG: Kernel launch configuration - blocks: %u, threads: %u\n", blocks, threads);

    IndicesL<G> d_i(d_ws_local);
    printf("DEBUG: Kernel input indices - d_i.xss: %p, d_i.ms: %p, d_i.qis: %p, d_i.tis: %p, state.get(): %p\n",
           d_i.xss, d_i.ms, d_i.qis, d_i.tis, state.get());

    ApplyGateL_Kernel<G><<<blocks, threads>>>(reinterpret_cast<fp_type*>(d_ws_local), d_i.xss, d_i.ms,
                                              d_i.qis, d_i.tis, 1 << num_effective_qs, state.get());
    cudaDeviceSynchronize();
    printf("DEBUG: Kernel executed successfully for ApplyGateL.\n");

 // 4. 결과 검증 (선택 사항)
    fp_type* host_check = new fp_type[h_i.buf_size / sizeof(fp_type)];
    cudaMemcpy(host_check, d_ws_local, h_i.buf_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("DEBUG: Result validation [%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;


    // 5. 메모리 해제
    cudaFree(d_ws_local);
    printf("DEBUG: Device memory freed - address: %p\n", d_ws_local);
}


*/

/* 오프로딩 로직. 
  template <unsigned G>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) {
    unsigned num_qubits = state.num_qubits();

    IndicesH<G> h_i(h_ws);
    GetIndicesH(num_qubits, qs, qs.size(), h_i);

    
    //KCJ
    // 1. 처음에 오프로딩 수행 (초기화)
    
    if (!is_initialized) {
        printf("KCJ: Initializing state and offloading first time...\n");
        const_cast<SimulatorCUDA*>(this)->OffloadState(state);
        is_initialized = true;
    }
    
    cudaDeviceSynchronize();
    
    
    // 2. 연산 전에 리로딩 수행
    printf("KCJ DEBUG: Reloading state for next cycle. state.get(): %p\n", state.get());
    const_cast<SimulatorCUDA*>(this)->ReloadState(state);
    cudaDeviceSynchronize();
    printf("KCJ DEBUG: After ReloadState. state.get(): %p\n", state.get());
    
    
    // 데이터 검증
    fp_type* host_check = new fp_type[sizeof(fp_type) * MinSize(state.num_qubits())];
    cudaMemcpy(host_check, state.get(), sizeof(fp_type) * MinSize(state.num_qubits()), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) { 
        printf("KCJ DEBUG: Data Check [%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;
    
    
    printf("KCJ DEBUG: ApplyGateH - Number of qubits: %u\n", num_qubits);
    printf("KCJ DEBUG: ApplyGateH - matrix pointer: %p, matrix size: %llu\n", matrix, h_i.matrix_size);
    
    // CPU에서 GPU로 메모리 복사
    std::memcpy((fp_type*)h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    printf("KCJ DEBUG: Matrix data copied from host to device. d_ws: %p\n", d_ws);
    
    unsigned k = 5 + G;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 64U;
    unsigned blocks = std::max(1U, size / 2);

    printf("KCJ: ApplyGateH: before Kernel execution? k = %u, n = %u, size = %u, threads = %u, blocks = %u\n", 
           5 + G, num_qubits > (5 + G) ? num_qubits - (5 + G) : 0, 
           unsigned{1} << (num_qubits > (5 + G) ? num_qubits - (5 + G) : 0), 
           64U, std::max(1U, (unsigned{1} << (num_qubits > (5 + G) ? num_qubits - (5 + G) : 0)) / 2));

    // IndicesH 객체에 새로운 d_ws 메모리를 사용하여 작업
    IndicesH<G> d_i(d_ws);
    printf("KCJ DEBUG: IndicesH initialized with device memory. d_ws: %p\n", d_ws);

    // 커널 실행
    printf("KCJ DEBUG: Launching Kernel. state.get(): %p\n", state.get());
    ApplyGateH_Kernel<G><<<blocks, threads>>>(
        reinterpret_cast<fp_type*>(d_ws), d_i.xss, d_i.ms, state.get());
      
    cudaDeviceSynchronize();
    printf("KCJ DEBUG<applyGateH>: Kernel launch details: d_ws: %p, d_i.xss: %p, d_i.ms: %p, state.get(): %p\n", 
       d_ws, d_i.xss, d_i.ms, state.get());
    printf("KCJ DEBUG: Kernel execution completed. state.get(): %p\n", state.get());
    
    
    // **8. 메모리 오프로딩 (결과 저장)**
    printf("KCJ DEBUG: Offloading state after gate application...\n");
    const_cast<SimulatorCUDA*>(this)->OffloadState(state);
    printf("KCJ DEBUG: State offloaded successfully. state.get(): %p\n", state.get());

    

}



  template <unsigned G>
  void ApplyGateL(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) {
   unsigned num_qubits = state.num_qubits();
    IndicesL<G> h_i(h_ws);
    auto num_effective_qs = GetIndicesL(num_qubits, qs, h_i);


    printf("KCJ: ApplyGateL - Using existing state memory: %p\n", state.get());    
    
    if (!is_initialized) {
        printf("KCJ: Initializing state and offloading first time...\n");
        const_cast<SimulatorCUDA*>(this)->OffloadState(state);
        is_initialized = true;
    }      
    

    cudaDeviceSynchronize();
    
    
    // 데이터 검증 
    fp_type* host_check = new fp_type[sizeof(fp_type) * MinSize(state.num_qubits())];
    cudaMemcpy(host_check, state.get(), sizeof(fp_type) * MinSize(state.num_qubits()), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) { 
        printf("KCJ DEBUG: Data Check [%d]: %f\n", i, host_check[i]);
    }
    delete[] host_check;

    // **5. 데이터 복사 (CPU → GPU)**
    printf("KCJ DEBUG: Reloading state for next cycle. state.get(): %p\n", state.get());
    const_cast<SimulatorCUDA*>(this)->ReloadState(state);
    cudaDeviceSynchronize();
    printf("KCJ DEBUG: After ReloadState. state.get(): %p\n", state.get());
    //cudaDeviceSynchronize();


    // CPU에서 GPU로 메모리 복사
    std::memcpy((fp_type*) h_ws, matrix, h_i.matrix_size);
    cudaMemcpyAsync(d_ws, h_ws, h_i.buf_size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    printf("KCJ DEBUG: Matrix data copied from host to device. d_ws: %p\n", d_ws);


    // **7. 커널 실행**
    unsigned k = 5 + num_effective_qs;
    unsigned n = num_qubits > k ? num_qubits - k : 0;
    unsigned size = unsigned{1} << n;
    unsigned threads = 32;
    unsigned blocks = size;
    
    
    printf("KCJ DEBUG: Preparing Kernel Execution | k=%u, n=%u, size=%u, threads=%u, blocks=%u\n", 
           k, n, size, threads, blocks);

   // **커널 실행**
    IndicesL<G> d_i(d_ws);
    printf("KCJ DEBUG: Launching Kernel. state.get(): %p\n", state.get());
    ApplyGateL_Kernel<G><<<blocks, threads>>>(
        reinterpret_cast<fp_type*>(d_ws), d_i.xss, d_i.ms, d_i.qis, d_i.tis,
        1 << num_effective_qs, state.get());

    cudaDeviceSynchronize();
    printf("KCJ DEBUG:<applyGateH> Kernel launch details: d_ws: %p, d_i.xss: %p, d_i.ms: %p, d_i.qis: %p, d_i.tis: %p, state.get(): %p\n", 
       d_ws, d_i.xss, d_i.ms, d_i.qis, d_i.tis, state.get());
    printf("KCJ DEBUG: Kernel execution completed. state.get(): %p\n", state.get());

    // **8. 메모리 오프로딩 (결과 저장)**
    printf("KCJ DEBUG: Offloading state after gate application...\n");
    const_cast<SimulatorCUDA*>(this)->OffloadState(state);
    printf("KCJ DEBUG: State offloaded successfully. state.get(): %p\n", state.get());


}
*/
/*
void CheckPointerValidity(const void* ptr, const char* ptr_name, size_t gpu_id) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err != cudaSuccess) {
        printf("KCJ ERROR: %s on GPU %lu is invalid: %s\n", ptr_name, gpu_id, cudaGetErrorString(err));
    } else {
        printf("KCJ DEBUG: %s on GPU %lu is valid.\n", ptr_name, gpu_id);
        if (attributes.type == cudaMemoryTypeDevice) {
            printf("KCJ DEBUG: %s is device memory.\n", ptr_name);
        } else if (attributes.type == cudaMemoryTypeHost) {
            printf("KCJ DEBUG: %s is host memory.\n", ptr_name);
        }
    }
}
*/


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
    num_qubits = 29;
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
                       
    num_qubits = 29;
      
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

    printf("KCJ: GetIndicesH_M start\n");
    num_qubits = 29;
    
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

            printf("DEBUG: GPU offset=%lu, xs[0]=%lu, xs[1]=%lu, xs[2]=%lu\n", global_offset, xs[0], xs[1], xs[2]);  
            printf("DEBUG: indices.ms[%u]: %u\n", i, indices.ms[i]);
            //  로그 추가
            printf("DEBUG: i=%u, a=%lu, global_offset=%lu, xss[%u] = %lu (global_index)\n", 
                    i, a, global_offset, i, indices.xss[i]);
        }
    }

    for (unsigned i = 0; i < std::min(size_t(10), static_cast<size_t>(indices.gsize)); ++i) {
        printf("DEBUG: GetIndicesH_m - GPU offset=%lu, indices.xss[%u] = %lu (global_index), indices.ms[%u] = %lu\n", 
               global_offset, i, indices.xss[i], i, indices.ms[i]);
    }
}


template <unsigned G>
void GetIndicesL_m(unsigned num_effective_qs, unsigned qmask,
                    IndicesL<G>& indices, size_t global_offset) {
    printf("KCJ: GetIndicesL_M start\n");

    auto& multi_gpu_pointers = VectorSpaceCUDA<StateSpaceCUDA<FP>, FP>::MultiGPUPointers();
    unsigned num_qubits = 29;
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
    
    //  디버깅 로그 추가 (GPU별 `qis`, `tis`, `xss` 값 검증)
    for (unsigned i = 0; i < std::min(size_t(10), static_cast<size_t>(indices.gsize)); ++i) {
        printf("DEBUG: GetIndicesL_m - indices.xss[%u] = %lu, indices.qis[%u] = %lu, indices.tis[%u] = %lu\n",
               i, indices.xss[i], i, indices.qis[i], i, indices.tis[i]);
    }
}



template <unsigned G>
unsigned GetIndicesL_m(unsigned num_qubits, const std::vector<unsigned>& qs,
                      IndicesL<G>& indices, size_t global_offset)  {
    num_qubits = 29;

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

    printf("DEBUG: qmaskh = %u, qmaskl = %u, qmaskh | qmaskl = %u\n", qmaskh, qmaskl, qmaskh | qmaskl);

    
    cudaDeviceSynchronize(); 
    GetIndicesH_m(num_qubits, eqs, num_effective_qs, indices, global_offset);
    printf("KCJ: Call GetIndicesH_M ! \n");
    cudaDeviceSynchronize(); 
    GetIndicesL_m(num_effective_qs, qmaskh | qmaskl, indices, global_offset);
    printf("KCJ: Call GetIndicesL_M ! \n");

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
    



}  // namespace qsim

#endif  // SIMULATOR_CUDA_H_

