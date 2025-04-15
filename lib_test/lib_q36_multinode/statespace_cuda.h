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

#ifndef STATESPACE_CUDA_H_
#define STATESPACE_CUDA_H_

#ifdef __NVCC__
  #include <cuda.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //KCJ
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

#include <algorithm>
#include <complex>
#include <cstdint>

#include "statespace.h"
#include "statespace_cuda_kernels.h"
#include "vectorspace_cuda.h"
#include "util_cuda.h"

namespace qsim {

/**
 * Object containing context and routines for CUDA state-vector manipulations.
 * State is a vectorized sequence of 32 real components followed by 32
 * imaginary components. 32 floating numbers can be proccessed in parallel by
 * a single warp. It is not recommended to use `GetAmpl` and `SetAmpl`.
 */
    
//KCJ: NCCL Class. 
/*  
class NCCLContext {
 public:
  NCCLContext() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    num_gpus_ = nDevices;
    comms_.resize(num_gpus_);

    // Initialize NCCL communicators
    ncclCommInitAll(comms_.data(), num_gpus_, nullptr);
  }

  ~NCCLContext() {
    for (auto& comm : comms_) {
      ncclCommDestroy(comm);
    }
  }

  ncclComm_t GetComm(int gpu_id) const { return comms_[gpu_id]; }
  int GetNumGPUs() const { return num_gpus_; }

 private:
  int num_gpus_;
  std::vector<ncclComm_t> comms_;
};

template <typename FP = float>
class StateSpaceCUDA : public StateSpace<StateSpaceCUDA<FP>, VectorSpaceCUDA, FP> {
 public:
  using Base = StateSpace<StateSpaceCUDA<FP>, qsim::VectorSpaceCUDA, FP>;
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;

  struct Grid {
    unsigned threads;
    unsigned dblocks;
    unsigned blocks;
  };

  struct Parameter {
    unsigned num_threads = 512;
    unsigned num_dblocks = 16;

    // 인자를 받는 생성자
    Parameter(unsigned threads, unsigned dblocks) 
        : num_threads(threads), num_dblocks(dblocks) {}

    // 기본 생성자 추가
    Parameter() = default;
  };

  NCCLContext nccl_context_;  // NCCL context for multi-GPU communication
  void* scratch_;
  uint64_t scratch_size_;

  explicit StateSpaceCUDA(const Parameter& param)
      : param_(param), scratch_(nullptr), scratch_size_(0), nccl_context_() {
    printf("KCJ: StateSpaceCUDA initialized with num_threads=%u, num_dblocks=%u\n", param.num_threads, param.num_dblocks);
  }

  virtual ~StateSpaceCUDA() {
    if (scratch_ != nullptr) {
      cudaFree(scratch_);
      printf("KCJ: Freed scratch memory at address %p\n", scratch_);
    }
  }

  // KCJ: GPU별로 상태 벡터를 나눠 파티션을 반환하는 함수
  std::vector<fp_type*> GetPartitions(State& state, int num_gpus) const {
    uint64_t total_size = MinSize(state.num_qubits());
    uint64_t size_per_gpu = total_size / num_gpus;

    printf("KCJ: GetPartitions - total_size: %llu, size_per_gpu: %llu, num_gpus: %d\n", total_size, size_per_gpu, num_gpus);

    std::vector<fp_type*> partitions(num_gpus);
    fp_type* base_ptr = state.get();

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      partitions[gpu_id] = base_ptr + gpu_id * size_per_gpu;
      printf("KCJ: Partition for GPU %d: ptr = %p\n", gpu_id, partitions[gpu_id]);
    }

    return partitions;
  }

  // KCJ: Multi GPU에서 AllReduceSum 수행
  void AllReduceSum(State& state) const {
    int num_gpus = nccl_context_.GetNumGPUs();
    auto partitions = GetPartitions(state, num_gpus);
    uint64_t size_per_gpu = MinSize(state.num_qubits()) / num_gpus;

    printf("KCJ: AllReduceSum - num_gpus: %d, size_per_gpu: %llu\n", num_gpus, size_per_gpu);

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      cudaSetDevice(gpu_id);
      ncclComm_t comm = nccl_context_.GetComm(gpu_id);
      
      // Debugging NCCL call
      printf("KCJ: Calling ncclAllReduce on GPU %d, comm = %p\n", gpu_id, comm);

      ncclAllReduce(partitions[gpu_id], partitions[gpu_id], size_per_gpu, ncclFloat, ncclSum, comm, cudaStreamDefault);
    }
    
    cudaDeviceSynchronize();
    printf("KCJ: AllReduceSum completed and all devices synchronized\n");
  }
  Parameter param_;
  NCCLContext nccl_context_;
  void* scratch_;
uint64_t scratch_size_;

*/

    
template <typename FP = float>
class StateSpaceCUDA :
    public StateSpace<StateSpaceCUDA<FP>, VectorSpaceCUDA, FP> {
 private:
  using Base = StateSpace<StateSpaceCUDA<FP>, qsim::VectorSpaceCUDA, FP>;

 protected:
  struct Grid {
    unsigned threads;
    unsigned dblocks;
    unsigned blocks;
  };

 public:
  using State = typename Base::State;
  using fp_type = typename Base::fp_type;
  void* scratch_;
  uint64_t scratch_size_;

    struct Parameter {
        unsigned num_threads = 512;
        unsigned num_dblocks = 16;

        Parameter() = default;
        Parameter(unsigned threads, unsigned dblocks)
            : num_threads(threads), num_dblocks(dblocks) {}
    };

    //KCJ
    // 다중 GPU용 생성자
    explicit StateSpaceCUDA(const std::vector<StateSpaceCUDA>& state_spaces,
                            const std::vector<ncclComm_t>& nccl_comms)
        : scratch_(nullptr), scratch_size_(0) {
  //    printf("KCJ: StateSpaceCUDA multi-GPU constructor called. Number of GPUs: %lu\n", state_spaces.size());

      // 필요한 초기화 작업 수행
      for (size_t i = 0; i < state_spaces.size(); ++i) {
       // printf("KCJ: Initializing state space for GPU %lu\n", i);
        // 각 GPU의 상태 공간이나 NCCL communicator에 대한 추가 설정 가능
      }
    }


    // 단일 GPU용 생성자
    explicit StateSpaceCUDA(const Parameter& param, int gpu_id)
        : param_(param), scratch_(nullptr), scratch_size_(0) {
    
    cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
        //printf("ERROR: Statespace cuda - CUDA error before kernel launch on GPU %lu: %s\n", 0, cudaGetErrorString(err_before));
    }
            
      //cudaSetDevice(gpu_id);
  //    printf("StateSpaceCUDA initialized on GPU %d with num_threads=%u, num_dblocks=%u\n",gpu_id, param.num_threads, param.num_dblocks);
   }
   

  virtual ~StateSpaceCUDA() {
    if (scratch_ != nullptr) {
      cudaFree(scratch_);
    }
  }



  static uint64_t MinSize(unsigned num_qubits) {
//    printf("KCJ: statespace_cuda-MinSize Start\n");
      //kcj
    //num_qubits = 1;
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
      
    //printf("KCJ: statespace_cuda-MinSize Result: %llu\n", result);
    return result;
      
  };
        
        /* 32 큐비트 기준
        size = 4294967296 bytes (4GB)로, 상태 벡터를 반으로 나누어 처리.
        threads = 512는 CUDA 커널에서 한 블록에서 실행되는 스레드 수.
        blocks = 8388608은 전체 데이터를 처리하기 위해 필요한 블록 수.
        bytes = 4096는 각 스레드가 처리할 데이터의 크기(바이트).
        */

  void InternalToNormalOrder(State& state) const {
//    printf("KCJ: statespace_cuda-InternalToNormalOrder: start\n");
    //unsigned num_qubits = 1;
    uint64_t size = MinSize(state.num_qubits()) / 2;
    //uint64_t size = MinSize(num_qubits) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads}); // 32큐비트 512 thread 
    unsigned blocks = size / threads; // 32큐비트 8388608 blocks
    unsigned bytes = 2 * threads * sizeof(fp_type); //4byte씩 

    InternalToNormalOrderKernel<<<blocks, threads, bytes>>>(state.get());
    printf("KCJ: statespace_cuda-InternalToNormalOrder:size: %llu, threads: %u, blocks: %u, bytes: %u\n", size, threads, blocks, bytes);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  void NormalToInternalOrder(State& state) const {
    printf("KCJ: statespace_cuda-NormalToInternalOrder: start\n");

    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;
    unsigned bytes = 2 * threads * sizeof(fp_type);
      
    NormalToInternalOrderKernel<<<blocks, threads, bytes>>>(state.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  // Uniform superposition.
  void SetStateUniform(State& state) const {
      //KCJ: 호출하는 곳 없는듯 
    uint64_t size = MinSize(state.num_qubits()) / 2;
    uint64_t hsize = uint64_t{1} << state.num_qubits();

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    fp_type v = double{1} / std::sqrt(hsize);

    SetStateUniformKernel<<<blocks, threads>>>(v, hsize, state.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    printf("KCJ: statespace_cuda-SetStateUniform \n");
  }



  void SetAllZeros(State& state) const {
      /*
    printf("KCJ: statespace_cuda-SetAllZeros: cudaMemset\n");
    cudaMemset(state.get(), 0,
               MinSize(state.num_qubits()) * sizeof(fp_type));
               */
  }




  // |0> state.
  void SetStateZero(State& state) const {
    /*
    SetAllZeros(state);
    fp_type one[1] = {1};
    cudaMemcpy(state.get(), one, sizeof(fp_type), cudaMemcpyHostToDevice);
    printf("KCJ: statespace_cuda-SetStateZero: cudamemcpy initial state vector transfer GPU?\n");
    */
  }



  // It is not recommended to use this function.
  static std::complex<fp_type> GetAmpl(const State& state, uint64_t i) {
    fp_type re, im;
    auto p = state.get() + 64 * (i / 32) + i % 32;
    printf("KCJ: statespace_cuda: complex : memcpy(D2H)\n");
    cudaMemcpy(&re, p, sizeof(fp_type), cudaMemcpyDeviceToHost);
    
        cudaMemcpy(&im, p + 32, sizeof(fp_type), cudaMemcpyDeviceToHost);
    return std::complex<fp_type>(re, im);
  }

  // It is not recommended to use this function.
  static void SetAmpl(
      State& state, uint64_t i, const std::complex<fp_type>& ampl) {
    fp_type re = std::real(ampl);
    fp_type im = std::imag(ampl);
    auto p = state.get() + 64 * (i / 32) + i % 32;
    printf("KCJ: statespace_cuda: SetAmpl : memcpy(H2D)\n");

    cudaMemcpy(p, &re, sizeof(fp_type), cudaMemcpyHostToDevice);
    
        cudaMemcpy(p + 32, &im, sizeof(fp_type), cudaMemcpyHostToDevice);
  }

  // It is not recommended to use this function.
  static void SetAmpl(State& state, uint64_t i, fp_type re, fp_type im) {
     cudaError_t err_before = cudaGetLastError();
    if (err_before != cudaSuccess) {
        printf("ERROR: Statespace 2 - CUDA error before kernel launch on GPU %lu: %s\n", 0, cudaGetErrorString(err_before));
    }
      return;
     /* 
    auto p = state.get() + 64 * (i / 32) + i % 32;
    printf("KCJ: statespace_cuda: SetAmpl2 : memcpy(H2D)\n");

   cudaMemcpy(p, &re, sizeof(fp_type), cudaMemcpyHostToDevice);
   
        cudaMemcpy(p + 32, &im, sizeof(fp_type), cudaMemcpyHostToDevice);
      */
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits,
                   const std::complex<fp_type>& val,
                   bool exclude = false) const {
    BulkSetAmpl(state, mask, bits, std::real(val), std::imag(val), exclude);
  }

  // Sets state[i] = complex(re, im) where (i & mask) == bits.
  // if `exclude` is true then the criteria becomes (i & mask) != bits.
  void BulkSetAmpl(State& state, uint64_t mask, uint64_t bits, fp_type re,
                   fp_type im, bool exclude = false) const {
    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    BulkSetAmplKernel<<<blocks, threads>>>(
        mask, bits, re, im, exclude, state.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  // Does the equivalent of dest += src elementwise.
  bool Add(const State& src, State& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }
    printf("KCJ: statespace_cuda-Add start()\n");
    uint64_t size = MinSize(src.num_qubits());

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    AddKernel<<<blocks, threads>>>(src.get(), dest.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    return true;
  }

  // Does the equivalent of state *= a elementwise.
  void Multiply(fp_type a, State& state) const {
    uint64_t size = MinSize(state.num_qubits());
    printf("KCJ: statespace_cuda-Multiply start()\n");
    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    MultiplyKernel<<<blocks, threads>>>(a, state.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  std::complex<double> InnerProduct(
      const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    using C = Complex<double>;
    auto r = Reduce<C, C, Product<fp_type>>(state1, state2);

    return {r.re, r.im};
  }

  double RealInnerProduct(const State& state1, const State& state2) const {
    if (state1.num_qubits() != state2.num_qubits()) {
      return std::nan("");
    }

    return Reduce<double, double, RealProduct<fp_type>>(state1, state2);
  }

  double Norm(const State& state) const {
    return Reduce<double, double, RealProduct<fp_type>>(state, state);
  }

  template <typename DistrRealType = double>
  std::vector<uint64_t> Sample(
      const State& state, uint64_t num_samples, unsigned seed) const {
    std::vector<uint64_t> bitstrings;
    printf("KCJ: statespace_cuda-Sample Start-Why D2H?\n");

    if (num_samples > 0) {
      Grid g1 = GetGrid1(MinSize(state.num_qubits()) / 2);
      unsigned bytes = g1.threads * sizeof(double);

      unsigned scratch_size = (g1.blocks + 1) * sizeof(double)
          + num_samples * (sizeof(uint64_t) + sizeof(DistrRealType));

      void* scratch = AllocScratch(scratch_size);

      double* d_res2 = (double*) scratch;
      double* d_res1 = d_res2 + 1;
      uint64_t* d_bitstrings = (uint64_t*) (d_res1 + g1.blocks);
      DistrRealType* d_rs = (DistrRealType *) (d_bitstrings + num_samples);

      auto op1 = RealProduct<fp_type>();
      auto op2 = Plus<double>();

      Reduce1Kernel<double><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, op1, op2, op2, state.get(), state.get(), d_res1);
      cudaPeekAtLastError();
      cudaDeviceSynchronize();

      double norm;

      if (g1.blocks == 1) {
            printf("KCJ: statespace_cuda-sample-d_res1 d2h?\n");
            cudaMemcpy(&norm, d_res1, sizeof(double), cudaMemcpyDeviceToHost);
      } else {
        Grid g2 = GetGrid2(g1.blocks);
        unsigned bytes = g2.threads * sizeof(double);

        auto op3 = Plus<double>();

        Reduce2Kernel<double><<<g2.blocks, g2.threads, bytes>>>(
            g2.dblocks, g1.blocks, op3, op3, d_res1, d_res2);
        cudaPeekAtLastError();
        cudaDeviceSynchronize();

        printf("KCJ: statespace_cuda-sample-d_res2 d2h?\n");

        cudaMemcpy(&norm, d_res2, sizeof(double), cudaMemcpyDeviceToHost);
      }

      // TODO: generate random values on the device.
      auto rs = GenerateRandomValues<DistrRealType>(num_samples, seed, norm);

      cudaMemcpy(d_rs, rs.data(),
                            num_samples * sizeof(DistrRealType),
                            cudaMemcpyHostToDevice);

      SampleKernel<<<1, g1.threads>>>(g1.blocks, g1.dblocks, num_samples,
                                      d_rs, d_res1, state.get(), d_bitstrings);
      cudaPeekAtLastError();
      cudaDeviceSynchronize();

      bitstrings.resize(num_samples, 0);
        
      printf("KCJ: statespace_cuda-sample-bitstring d2h?\n");

      cudaMemcpy(bitstrings.data(), d_bitstrings,
                            num_samples * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost);
    }

    return bitstrings;
  }

  using MeasurementResult = typename Base::MeasurementResult;

  void Collapse(const MeasurementResult& mr, State& state) const {
      
    printf("KCJ: statespace_cuda-Collapse start()\n");
    using Op = RealProduct<fp_type>;
    double r = Reduce<double, double, Op>(mr.mask, mr.bits, state, state);
    fp_type renorm = 1 / std::sqrt(r);

    uint64_t size = MinSize(state.num_qubits()) / 2;

    unsigned threads = std::min(size, uint64_t{param_.num_threads});
    unsigned blocks = size / threads;

    CollapseKernel<<<blocks, threads>>>(mr.mask, mr.bits, renorm, state.get());
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  std::vector<double> PartialNorms(const State& state) const {
      
    printf("KCJ: statespace_cuda-PartialNorms start()\n");
    Grid g = GetGrid1(MinSize(state.num_qubits()) / 2);

    unsigned scratch_size = g.blocks * sizeof(double);
    unsigned bytes = g.threads * sizeof(double);

    double* d_res = (double*) AllocScratch(scratch_size);

    auto op1 = RealProduct<fp_type>();
    auto op2 = Plus<double>();

    Reduce1Kernel<double><<<g.blocks, g.threads, bytes>>>(
        g.dblocks, op1, op2, op2, state.get(), state.get(), d_res);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    std::vector<double> norms(g.blocks);

        printf("KCJ: statespace_cuda:Collapse: cudaMemcpy(D2H)\n");
        cudaMemcpy(norms.data(), d_res, scratch_size, cudaMemcpyDeviceToHost);

    return norms;
  }

  uint64_t FindMeasuredBits(
      unsigned m, double r, uint64_t mask, const State& state) const {
    printf("KCJ: statespace_cuda-FindMeasuredBits start()\n");
    Grid g = GetGrid1(MinSize(state.num_qubits()) / 2);

    uint64_t res;
    uint64_t* d_res = (uint64_t*) AllocScratch(sizeof(uint64_t));

    FindMeasuredBitsKernel<<<1, g.threads>>>(
        m, g.dblocks, r, state.get(), d_res);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

        printf("KCJ: statespace_cuda:FindMeasuredBits: cudaMemcpy(D2H)\n");
        cudaMemcpy(&res, d_res, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    return res & mask;
  }

 protected:
  Parameter param_;

  void* AllocScratch(uint64_t size) const {
      //KCJ: Not used...? 
    if (size > scratch_size_) {
      if (scratch_ != nullptr) {
        cudaFree(scratch_);
      }
      printf("KCJ: statespace_cuda1 > size: %llu\n", size);
      cudaMalloc(const_cast<void**>(&scratch_), size);
      printf("KCJ: statespace_cuda1 > cudaMalloc(), scratch: %p, size: %llu\n", &scratch_, size);
      const_cast<uint64_t&>(scratch_size_) = size;
    }
    printf("KCJ: statespace_cuda2 > size: %llu\n", size);
    printf("KCJ: statespace_cuda2 > cudaMalloc(), scratch: %p, size: %llu\n", &scratch_, size);

    return scratch_;
  }

  Grid GetGrid1(uint64_t size) const {
    Grid grid;

    grid.threads = std::min(size, uint64_t{param_.num_threads});
    grid.dblocks = std::min(size / grid.threads, uint64_t{param_.num_dblocks});
    grid.blocks = size / (grid.threads * grid.dblocks);

    return grid;
  }

  Grid GetGrid2(unsigned size) const {
    Grid grid;

    grid.threads = std::min(param_.num_threads, std::max(32U, size));
    grid.dblocks = std::max(1U, size / grid.threads);
    grid.blocks = 1;

    return grid;
  }

  template <typename FP1, typename FP2, typename Op>
  FP2 Reduce(const State& state1, const State& state2) const {
    return Reduce<FP1, FP2, Op>(0, 0, state1, state2);
  }

  template <typename FP1, typename FP2, typename Op>
  FP2 Reduce(uint64_t mask, uint64_t bits,
             const State& state1, const State& state2) const {
    uint64_t size = MinSize(state1.num_qubits()) / 2;
      
    printf("KCJ: statespace_cuda.h-FP2 reduce: when call. ?\n");

    Grid g1 = GetGrid1(size);
    unsigned bytes = g1.threads * sizeof(FP1);

    FP2* d_res2 = (FP2*) AllocScratch((g1.blocks + 1) * sizeof(FP2));
    FP2* d_res1 = d_res2 + 1;

    auto op1 = Op();
    auto op2 = Plus<FP1>();
    auto op3 = Plus<typename Scalar<FP1>::type>();

    if (mask == 0) {
      printf("KCJ: statespace_cuda-reduce1Kernerl gogo\n");
      Reduce1Kernel<FP1><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, op1, op2, op3, state1.get(), state2.get(), d_res1);
    } else {
            printf("KCJ: statespace_cuda-reduce1MaskedKernerl gogo\n");  
      Reduce1MaskedKernel<FP1><<<g1.blocks, g1.threads, bytes>>>(
          g1.dblocks, mask, bits, op1, op2, op3, state1.get(), state2.get(),
          d_res1);
    }
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    FP2 result;

    printf("KCJ: statespace_cuda-result?\n", &result);
      
    if (g1.blocks == 1) {
          printf("KCJ: statespace_cuda: cudaMemcpy(D2H)\n");
          cudaMemcpy(&result, d_res1, sizeof(FP2), cudaMemcpyDeviceToHost);
    } else {
      Grid g2 = GetGrid2(g1.blocks);
      unsigned bytes = g2.threads * sizeof(FP2);

      auto op2 = Plus<FP2>();
      auto op3 = Plus<typename Scalar<FP2>::type>();

      Reduce2Kernel<FP2><<<g2.blocks, g2.threads, bytes>>>(
          g2.dblocks, g1.blocks, op2, op3, d_res1, d_res2);
      cudaPeekAtLastError();
      cudaDeviceSynchronize();

     // ErrorCheck(
        //KCJ: Not used;;;
          cudaMemcpy(&result, d_res2, sizeof(FP2), cudaMemcpyDeviceToHost);
          printf("KCJ: statespace_cuda: cudaMemcpy(D2H) : result: %p, d_res2: %p, size: %llu\n", &result, d_res2, sizeof(FP2));
    }

    return result;
  }


};

}  // namespace qsim

#endif  // STATESPACE_CUDA_H_
