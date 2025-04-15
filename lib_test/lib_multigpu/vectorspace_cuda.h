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



#ifndef VECTORSPACE_CUDA_H_
#define VECTORSPACE_CUDA_H_

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //KCJ
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

#include <memory>
#include <utility>
#include <vector>
#include </opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include/mpi.h>

namespace qsim {

namespace detail {

inline void do_not_free(void*) {}

inline void free(void* ptr) {
  //cudaFree(ptr);
}

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename FP>
class VectorSpaceCUDA {
 public:
  using fp_type = FP;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;
  // CreateMultiGPU 전용 포인터 관리 배열
  static std::vector<fp_type*> multi_gpu_ptrs_;


 public:
  class Vector {
   public:
    Vector() : ptr_(nullptr, &detail::free), num_qubits_(0) {}

    Vector(Pointer&& ptr, unsigned num_qubits)
        : ptr_(std::move(ptr)), num_qubits_(num_qubits) {}

    Vector(const Vector& other)
        : ptr_(nullptr, &detail::free), num_qubits_(other.num_qubits_) {
      if (other.ptr_) {
        fp_type* p;
        //auto size = sizeof(fp_type) * Impl::MinSize(other.num_qubits_);
        auto size = sizeof(fp_type) * 64;
        cudaMalloc(&p, size);
        cudaMemcpy(p, other.ptr_.get(), size, cudaMemcpyDeviceToDevice);
        ptr_.reset(p);
      }
    }

    Vector& operator=(const Vector& other) {
      if (this != &other) {
        num_qubits_ = other.num_qubits_;
        if (other.ptr_) {
          fp_type* p;
          //auto size = sizeof(fp_type) * Impl::MinSize(other.num_qubits_);
          auto size = sizeof(fp_type) * 64;
          cudaMalloc(&p, size);
          cudaMemcpy(p, other.ptr_.get(), size, cudaMemcpyDeviceToDevice);
          ptr_.reset(p);
        } else {
          ptr_.reset(nullptr);
        }
      }
      return *this;
    }

    void set(fp_type* ptr, unsigned num_qubits) {
      ptr_.reset(ptr);
      num_qubits_ = num_qubits;
    }

    fp_type* get() {
      printf("KCJ DEBUG: Vector.get() called. Returning address: %p\n", ptr_.get());
      return ptr_.get();
    }

    const fp_type* get() const {
      printf("KCJ DEBUG: Vector.get() const ver. called. Returning address: %p\n", ptr_.get());
      return ptr_.get();
    }


    fp_type* release() {
      num_qubits_ = 0;
      return ptr_.release();
    }

    unsigned num_qubits() const {
      return num_qubits_;
    }

   bool requires_copy_to_host() const {
        return true;  // 항상 true를 반환
      }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
  };

  static std::vector<fp_type*>& MultiGPUPointers() {
    return multi_gpu_ptrs_;
  }

  static void ClearMultiGPUPointers() {
    for (auto ptr : multi_gpu_ptrs_) {
      cudaFree(ptr);
    }
    multi_gpu_ptrs_.clear();
  }

  static std::vector<Vector>& GlobalStateParts() {
    static std::vector<Vector> global_state_parts;
    return global_state_parts;
  }

  static Vector Create(unsigned num_qubits) {
    if (num_qubits == 0) {
      printf("KCJ WARNING: num_qubits is 0. Skipping initialization.\n");
      return Null();
    }

    if (!GlobalStateParts().empty() && GlobalStateParts()[0].num_qubits() == num_qubits) {
      printf("KCJ DEBUG: GlobalStateParts already initialized.\n");
      return GlobalStateParts()[0];
    }

    fp_type* p;
    auto size = sizeof(fp_type) * 64;
    auto rc = cudaMalloc(&p, size);
    printf("KCJ: vectorspace: Create Dummy Vector -> size : %u, rc : %d, p : %p\n", size, rc, p);

    if (rc == cudaSuccess) {
      int num_gpus = 0;
      cudaError_t err = cudaGetDeviceCount(&num_gpus);

      if (err != cudaSuccess || num_gpus == 0) {
        printf("KCJ WARNING: No CUDA devices detected. Defaulting to single GPU.\n");
        num_gpus = 1;
      }

      printf("KCJ DEBUG: Initializing GlobalStateParts with %d GPUs.\n", num_gpus);
      GlobalStateParts() = CreateMultiGPU(num_qubits, num_gpus);
      cudaFree(p);
      //cudaDeviceSynchronize();
      return GlobalStateParts()[0];
    } else {
      printf("KCJ ERROR: Failed to allocate dummy vector.\n");
      return Null();
    }
  }


static std::vector<Vector> CreateMultiGPU(unsigned num_qubits, int num_gpus) {
    if (num_qubits == 0) {
        printf("KCJ WARNING: num_qubits is 0. Skipping multi-GPU initialization.\n");
        return GlobalStateParts();
    }


    if (!GlobalStateParts().empty()) {
        printf("KCJ DEBUG: GlobalStateParts already initialized. Skipping CreateMultiGPU.\n");
        return GlobalStateParts();
    }

    // 기존 포인터 초기화
    printf("DEBUG: Clearing MultiGPUPointers. Current size: %lu\n", MultiGPUPointers().size());
    ClearMultiGPUPointers(); // GPU 메모리 해제 후 초기화
    printf("DEBUG: MultiGPUPointers cleared. Current size: %lu\n", MultiGPUPointers().size());

    if (!GlobalStateParts().empty()) {
        printf("DEBUG: Clearing GlobalStateParts. Current size: %lu\n", GlobalStateParts().size());
        GlobalStateParts().clear();
        printf("DEBUG: GlobalStateParts cleared. Current size: %lu\n", GlobalStateParts().size());
    }

    size_t total_size = sizeof(fp_type) * MinSize(num_qubits);
    size_t size_per_gpu = total_size / num_gpus;

    std::vector<Vector> state_parts;

    // P2P 초기화 호출
    //InitializeP2P(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        fp_type* gpu_ptr = nullptr;
        size_t free_mem = 0, total_mem = 0;
        //KCJ: 이부분 때는 문제가 없음. (Invalid CUDA)
        cudaError_t err_before = cudaGetLastError();
        if (err_before != cudaSuccess) {
            printf("ERROR: Vector_cuda -> CUDA error before kernel launch on GPU %lu: %s\n", i, cudaGetErrorString(err_before));
        }

        
        cudaError_t rc = cudaMalloc(&gpu_ptr, size_per_gpu);
        if (rc != cudaSuccess) {
            printf("ERROR: cudaMalloc failed on GPU %d: %s\n", i, cudaGetErrorString(rc));
            return {};
        }

        MultiGPUPointers().push_back(gpu_ptr);
        printf("DEBUG: Allocated GPU memory: multi_gpu_pointers[%d] = %p\n", i, gpu_ptr);

        Vector vector{Pointer{gpu_ptr, &detail::do_not_free}, num_qubits};
        printf("DEBUG: Vector created - vector.get() = %p\n", vector.get());
        state_parts.emplace_back(std::move(vector));

        
        printf("INFO: GPU %u - Free memory: %lu bytes, Total memory: %lu bytes\n", i, free_mem, total_mem);

    
    }

    GlobalStateParts() = state_parts;

    return state_parts;
}


  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &detail::free}, 0};
  }

  static bool IsNull(const Vector& vector) {
    return vector.get() == nullptr;
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }
    printf("KCJ: vectorspace: Cuda memcpy D2H Copy 1 \n");

    cudaMemcpy(dest.get(), src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToDevice);

    return true;
  }

  bool Copy(const Vector& src, fp_type* dest) const {
    printf("KCJ: vectorspace: Cuda memcpy D2H Copy 2 \n");
    cudaMemcpy(dest, src.get(),
               sizeof(fp_type) * Impl::MinSize(src.num_qubits()),
               cudaMemcpyDeviceToHost);
    return true;
  }

  bool Copy(const fp_type* src, Vector& dest) const {
    printf("KCJ: vectorspace: Cuda memcpy D2H Copy3 \n");
    cudaMemcpy(dest.get(), src,
               sizeof(fp_type) * Impl::MinSize(dest.num_qubits()),
               cudaMemcpyHostToDevice);
    return true;
  }

  bool Copy(const fp_type* src, uint64_t size, Vector& dest) const {
    size = std::min(size, Impl::MinSize(dest.num_qubits()));
    printf("KCJ: vectorspace: Cuda memcpy D2H Copy 4 \n");
    cudaMemcpy(dest.get(), src,
               sizeof(fp_type) * size,
               cudaMemcpyHostToDevice);
    return true;
  }
    
   static void InitializeP2P(int num_gpus) {
        printf("KCJ: Initializing P2P for %d GPUs.\n", num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                        if (err == cudaErrorPeerAccessAlreadyEnabled) {
                            printf("KCJ DEBUG: P2P already enabled between GPU %d and GPU %d\n", i, j);
                        } else if (err != cudaSuccess) {
                            printf("KCJ ERROR: Failed to enable P2P between GPU %d and GPU %d: %s\n", 
                                   i, j, cudaGetErrorString(err));
                        } else {
                            printf("KCJ: P2P enabled between GPU %d and GPU %d\n", i, j);
                        }
                    } else {
                        printf("KCJ: P2P not supported between GPU %d and GPU %d\n", i, j);
                    }
                }
            }
        }
    }




static uint64_t MinSize(unsigned num_qubits) {
    printf("KCJ: vectorCuda_cuda-MinSize Start\n");
      //kcj
    num_qubits = 29;
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    printf("KCJ: vectorCuda_cuda-MinSize Result: %llu\n", result);
    return result;
      
};


  void DeviceSync() {
    //cudaDeviceSynchronize();
  }

 protected:
};

// 정적 멤버 변수 정의
template <typename Impl, typename FP>
std::vector<typename VectorSpaceCUDA<Impl, FP>::fp_type*> 
VectorSpaceCUDA<Impl, FP>::multi_gpu_ptrs_;

}  // namespace qsim

#endif  // VECTORSPACE_CUDA_H_
