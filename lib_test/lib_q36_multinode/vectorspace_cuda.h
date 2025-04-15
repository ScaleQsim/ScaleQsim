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
#include </opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include/mpi.h>
#include <chrono> 

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
  static double total_cuda_malloc_time;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;
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
  //    printf("KCJ DEBUG: Vector.get() called. Returning address: %p\n", ptr_.get());
      return ptr_.get();
    }

    const fp_type* get() const {
   //   printf("KCJ DEBUG: Vector.get() const ver. called. Returning address: %p\n", ptr_.get());
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
   //   printf("KCJ WARNING: num_qubits is 0. Skipping initialization.\n");
      return Null();
    }

    if (!GlobalStateParts().empty() && GlobalStateParts()[0].num_qubits() == num_qubits) {
   //   printf("KCJ DEBUG: GlobalStateParts already initialized.\n");
      return GlobalStateParts()[0];
    }

    fp_type* p;
    auto size = sizeof(fp_type) * 64;
    auto rc = cudaMalloc(&p, size);
  //  printf("KCJ: vectorspace: Create Dummy Vector -> size : %u, rc : %d, p : %p\n", size, rc, p);

    if (rc == cudaSuccess) {
      int num_gpus = 0;
      cudaError_t err = cudaGetDeviceCount(&num_gpus);

      if (err != cudaSuccess || num_gpus == 0) {
    //    printf("KCJ WARNING: No CUDA devices detected. Defaulting to single GPU.\n");
        num_gpus = 1;
      }

    //  printf("KCJ DEBUG: Initializing GlobalStateParts with %d GPUs.\n", num_gpus);
      GlobalStateParts() = CreateMultiGPU(num_qubits, num_gpus);
      cudaFree(p);
      //cudaDeviceSynchronize();
      return GlobalStateParts()[0];
    } else {
   //   printf("KCJ ERROR: Failed to allocate dummy vector.\n");
      return Null();
    }
  }

/*
 * CreateMultiGPU:
 * Initializes distributed state vector memory across multiple GPUs and nodes.
 * Computes partition size per node and GPU, then allocates aligned memory on each GPU.
 * Ensures consistent MPI initialization and synchronizes global state after allocation.
*/

static std::vector<Vector> CreateMultiGPU(unsigned num_qubits, int num_gpus) {
    if (num_qubits == 0) {
        printf("KCJ WARNING: num_qubits is 0. Skipping multi-GPU initialization.\n");
        return GlobalStateParts();
    }


    int flag;
    MPI_Initialized(&flag);
        if (!flag) {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
        }

    // KCJ: Enable Multi node. 
    int node_id, total_nodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    //MPI_Comm_rank(MPI_COMM_WORLD, &total_nodes);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);
    printf("KCJ: MPI Comm rank=%d, total_nodes=%d\n", node_id, total_nodes);
    
    if (!GlobalStateParts().empty()) {
        printf("KCJ DEBUG: GlobalStateParts already initialized. Skipping CreateMultiGPU.\n");
        return GlobalStateParts();
    }

 //   printf("DEBUG: Clearing MultiGPUPointers. Current size: %lu\n", MultiGPUPointers().size());
    ClearMultiGPUPointers(); 
  //  printf("DEBUG: MultiGPUPointers cleared. Current size: %lu\n", MultiGPUPointers().size());

    if (!GlobalStateParts().empty()) {
      //  printf("DEBUG: Clearing GlobalStateParts. Current size: %lu\n", GlobalStateParts().size());
        GlobalStateParts().clear();
       // printf("DEBUG: GlobalStateParts cleared. Current size: %lu\n", GlobalStateParts().size());
    }

    size_t total_size = MinSize(num_qubits);
    size_t size_per_node, size_per_gpu, Real_size_per_gpu;

    if (total_nodes > 1) {
        size_per_node = total_size / total_nodes;  
        size_per_gpu = size_per_node / num_gpus;   
        Real_size_per_gpu = ((size_per_gpu + 128) / 128) * 128;

        //printf("KCJ: Vector_cuda---->Multi-Node Mode (NODE %d) - size_per_node: %lu, size_per_gpu: %lu\n",  node_id, size_per_node, Real_size_per_gpu);
    } else {
        size_per_node = total_size;                
        size_per_gpu = total_size / num_gpus;      
       // printf("KCJ: Vector_cuda---->Single Node Mode - size_per_node: %lu, size_per_gpu: %lu\n",  size_per_node, size_per_gpu);
    }

    std::vector<Vector> state_parts;

    // InitializeP2P(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        fp_type* gpu_ptr = nullptr;
        size_t free_mem = 0, total_mem = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        cudaError_t err_before = cudaGetLastError();
        if (err_before != cudaSuccess) {
            printf("ERROR: Vector_cuda -> CUDA error before kernel launch on GPU %d: %s\n", 
                   i, cudaGetErrorString(err_before));
        }

        cudaError_t rc = cudaMalloc(&gpu_ptr, Real_size_per_gpu);
        if (rc != cudaSuccess) {
            printf("ERROR: cudaMalloc failed on NODE %d, GPU %d: %s\n", 
                   node_id, i, cudaGetErrorString(rc));
            return {};
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_cuda_malloc_time += elapsed_time;  //  실행 시간 누적

        MultiGPUPointers().push_back(gpu_ptr);

        Vector vector{Pointer{gpu_ptr, &detail::do_not_free}, num_qubits};
        state_parts.emplace_back(std::move(vector));


    }

    GlobalStateParts() = state_parts;
    MPI_Barrier(MPI_COMM_WORLD);
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


/*
 * InitializeP2P:
 * Enables CUDA Peer-to-Peer (P2P) access between all GPU pairs within a node.
 * Checks P2P capability and activates bidirectional access if supported.
 * This allows direct memory reads across GPUs to reduce intra-node communication overhead.
*/

   static void InitializeP2P(int num_gpus) {
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {
                        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                        if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        } else if (err != cudaSuccess) {
                                   i, j, cudaGetErrorString(err);
                        } else {
                        }
                    } else {
                    }
                }
            }
        }
    }




static uint64_t MinSize(unsigned num_qubits) {
    num_qubits = 36;  
    uint64_t result = std::max(uint64_t{64}, 2 * (uint64_t{1} << num_qubits));
    return result;
}



  void DeviceSync() {
    //cudaDeviceSynchronize();
  }

 protected:
};

template <typename Impl, typename FP>
std::vector<typename VectorSpaceCUDA<Impl, FP>::fp_type*> 
VectorSpaceCUDA<Impl, FP>::multi_gpu_ptrs_;

template <typename Impl, typename FP>
double VectorSpaceCUDA<Impl, FP>::total_cuda_malloc_time = 0.0;
}  // namespace qsim

#endif  // VECTORSPACE_CUDA_H_
