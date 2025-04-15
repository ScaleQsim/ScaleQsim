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

#ifndef SIMULATOR_CUDA_KERNELS_H_
#define SIMULATOR_CUDA_KERNELS_H_

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>

  #include "util_cuda.h"
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif


namespace qsim {


template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, fp_type* multi_gpu_rstate,
    const size_t global_offset, const size_t size_per_gpu, const size_t gpu_id, const size_t world_size, const size_t global_gpu_id) {

  static_assert(G < 7, "Gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                                             (G < 6 ? gsize : 32) : (G < 5 ? 8 : 16));

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];
    
  __shared__ idx_type shared_global_offset;
  __shared__ idx_type shared_size_per_gpu;
  __shared__ idx_type shared_gpu_id;
  __shared__ idx_type shared_global_gpu_id; //global 

 if (threadIdx.x == 0) {
    shared_size_per_gpu = (idx_type)size_per_gpu;
    shared_gpu_id = (idx_type)gpu_id;
    shared_global_gpu_id = (idx_type)global_gpu_id; 
    shared_global_offset = (idx_type)global_offset;
     //shared_global_offset = (idx_type)global_offset;
  }
__syncthreads();

    
 if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }
    
__syncthreads();
    

 int64_t i = (64 * idx_type{blockIdx.x} + threadIdx.x) & 0xffffffffffe0;
 int64_t ii = (int64_t)(i & mss[0]) | shared_global_offset;

ii = (uint64_t)(ii) - (uint64_t)(shared_gpu_id * shared_size_per_gpu);

    for (unsigned j = 1; j <= G; ++j) {
        i *= 2;
        ii |= (i & mss[j]);
    }
  
    
    fp_type* rstate = multi_gpu_rstate; 
    auto p0 = rstate + 2 * ii + threadIdx.x % 32; 
    



for (unsigned k = 0; k < gsize; ++k) {
    int64_t temp_xss = 0;
      temp_xss = (int64_t)xss[k] - (int64_t)(shared_global_offset);
        
    
    if (temp_xss < 0) {
        temp_xss = (int64_t)-temp_xss;    
    }

    if ((uint64_t)temp_xss >= shared_size_per_gpu) {
        temp_xss = shared_size_per_gpu - 1;
    }

    temp_xss &= ~3;
    xss[k] = (idx_type)temp_xss;



    size_t node_start = (size_t)rstate;
    size_t node_end = node_start + shared_size_per_gpu;

    if ((size_t)(p0) < (size_t)multi_gpu_rstate || 
        (size_t)(p0) > (size_t)(multi_gpu_rstate + shared_size_per_gpu)) {
        return; }
    
    
    if ((size_t)(p0 + xss[k]) < node_start ||
        (size_t)(p0 + xss[k]) >= node_end) {
        return;
    } 

    if (xss[k] < 0 ||  ((uint64_t)(xss[k]) < shared_global_offset || (uint64_t)(xss[k]) >= (shared_global_offset + shared_size_per_gpu))) {
        rs[k] = rs[k];  
        is[k] = is[k];  
        return;
    } 

    rs[k] = *(p0 + xss[k]);
    is[k] = *(p0 + xss[k] + 32);
     
}



  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      __syncthreads();

      for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }

      __syncthreads();
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

        *(p0 + xss[k]) = rn;
        *(p0 + xss[k] + 32) = in;


      
    } 
  }
    __syncthreads();

    }




template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateL_Kernel(
    const fp_type* v0, idx_type*  xss,
    const idx_type*  mss, const unsigned* qis,
    const unsigned*  tis, const unsigned esize,
    fp_type*  multi_gpu_rstate, const size_t global_offset, const size_t size_per_gpu, const size_t gpu_id, const size_t world_size, const size_t global_gpu_id) {

  static_assert(G < 7, "Gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                                             (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type v[2 * gsize * rows];
  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
    
  __shared__ size_t shared_global_offset;
  __shared__ size_t shared_global_gpu_id;

  shared_global_offset = global_offset;
  shared_global_gpu_id = global_gpu_id;
    
  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

    
  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = (i & mss[0]) | shared_global_offset;
  ii = (uint64_t)(ii) - (uint64_t)(shared_global_gpu_id * size_per_gpu);
  
  for (unsigned j = 1; j <= G; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }
    
    fp_type* rstate = multi_gpu_rstate;  
    auto p0 = rstate + 2 * ii + threadIdx.x;


  for (unsigned k = 0; k < gsize; ++k) {
      int64_t temp_xss = (int64_t)xss[k] - (int64_t)(shared_global_offset);

    if (temp_xss < 0) {
        temp_xss = -temp_xss;

    }
      
    if ((uint64_t)temp_xss >= size_per_gpu) {
       temp_xss = size_per_gpu - 1;
    }

  
    xss[k] = (idx_type)temp_xss; 


    size_t node_start = (size_t)rstate;
    size_t node_end = node_start + size_per_gpu;

    if ((size_t)(p0) < (size_t)multi_gpu_rstate || 
        (size_t)(p0) > (size_t)(multi_gpu_rstate + size_per_gpu)) {

        return; 
    }
    

    if ((size_t)(p0 + xss[k]) < node_start ||
        (size_t)(p0 + xss[k]) >= node_end) {
        return;
    }

    if (xss[k] < 0 ||  ((uint64_t)(xss[k]) < shared_global_offset || (uint64_t)(xss[k]) >= (shared_global_offset + size_per_gpu))) {
   
        rs[k] = rs[k]; 
        is[k] = is[k];  
        return;
    } 

      
    rs0[threadIdx.x][k] = *(p0 + xss[k]);
    is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
  }



    
    
 for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }


  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  for (unsigned k = 0; k < esize; ++k) {
    *(p0 + xss[k]) = rs0[threadIdx.x][k];
    *(p0 + xss[k] + 32) = is0[threadIdx.x][k];

  }
         
}

//--------------------------





// Original
template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 64.
          //printf("KCJ: Simulator_cuda_kernel-ApplyGateH_Kernel start\n");
//if (threadIdx.x == 0) {
  //printf("KCJ: ApplyGateH_Kernel start. blockIdx.x: %d, threadIdx.x: %d\n",blockIdx.x, threadIdx.x); }

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                       (G < 6 ? gsize : 32) : (G < 5 ? 8 : 16));

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  idx_type i = (64 * idx_type{blockIdx.x} + threadIdx.x) & 0xffffffffffe0;
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j <= G; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }
    //if (threadIdx.x == 0) {
  //printf("KCJ: ApplyGateH_Kernel Intermediate values - i: %llu, ii: %llu\n", i, ii); }


  auto p0 = rstate + 2 * ii + threadIdx.x % 32;

  for (unsigned k = 0; k < gsize; ++k) {
    rs[k] = *(p0 + xss[k]);
    is[k] = *(p0 + xss[k] + 32);
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      __syncthreads();

      for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }

      __syncthreads();
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      *(p0 + xss[k]) = rn;
      *(p0 + xss[k] + 32) = in;
    }
  }
    //if (threadIdx.x == 0) {
      //printf("KCJ: ApplyGateH_Kernel end.\n");}
}

    


template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyGateL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned esize,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.
    // KCJ: blockIdx.x와 threadIdx.x는 각각 커널 블록과 스레드를 식별 - 어떤 스레드가 실행 중인지 추적.
    //if (threadIdx.x == 0) {
   //printf("KCJ: ApplyGateL_Kernel start. blockIdx.x: %p, threadIdx.x: %u\n",blockIdx.x, threadIdx.x); //}

    //printf("KCJ: Simulator_cuda_kernel-ApplyGateL_Kernel start\n");
 //KCJ: G는 큐빗 수를 나타내는 변수-이 값이 7 이상이면 6큐빗까지의 게이트만 지원.
  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type v[2 * gsize * rows];
  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }
    
    //printf("KCJ: ApplyGateL_Kernel - v0[threadIdx.x]: %u, v[%d]: %u\n", v0[threadIdx.x], threadIdx.x, v[threadIdx.x]);
    // v0에서 v로 데이터를 복사한 후 각 thread에서 v[threadIdx.x]의 값 확인
    
    // Step 2: mss, xss 계산을 위한 초기 인덱스 계산
  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j <= G; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  // KCJ: p0 -> rstate 포인트임 (ii: Index 값(rstate 배열 첨자 - 실수 + 허수 (2) )
  auto p0 = rstate + 2 * ii + threadIdx.x;

      //  printf("KCJ: ApplyGateL_Kernel - threadIdx.x: %u: - initial p0: %p, ii: %u\n", threadIdx.x, p0, ii);
    // p0는 rstate 배열 내의 인덱스 포인터로 계산되며, 이 값을 확인.

 // Step 3: rs0, is0 배열에 데이터를 로드
  for (unsigned k = 0; k < gsize; ++k) {
    rs0[threadIdx.x][k] = *(p0 + xss[k]);
    is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
  }
    
    //printf("KCJ: ApplyGateL_Kernel - rs0[threadIdx.x][0]: %u, is0[threadIdx.x][0]: %u\n", rs0[threadIdx.x][0], is0[threadIdx.x][0]);
    // 각 thread에 대해 rs0, is0 배열을 초기화한 후 첫 번째 값 출력

    // Step 4: rs, is 배열에 데이터를 로드
  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
      
    //printf("KCJ: ApplyGateL_Kernel - k: %u, rs[%u]: %f, is[%u]: %f\n", k, k, rs[k], k, is[k]);

  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
          
      //  printf("KCJ: ApplyGateL_Kernel - s: %u, k: %u, l: %u, rm: %f, im: %f, rn: %f, in: %f\n", s, k, l, rm, im, rn, in);
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }
  // 행렬 곱셈을 통해 결과를 계산하고 rs0, is0 배열에 결과를 저장.

    // Step 6: 계산된 결과를 rstate에 저장 (p0 가 rstate)
  for (unsigned k = 0; k < esize; ++k) {
    *(p0 + xss[k]) = rs0[threadIdx.x][k];
    *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
        //printf("KCJ: ApplyGateL_Kernel - Saving to p0 + xss[%u]: %f, %f\n", k, rs0[threadIdx.x][k], is0[threadIdx.x][k]);


  }
    // 계산된 실수 및 허수 결과를 rstate에 저장하면서 각 값을 출력.

}
    

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, unsigned num_mss, idx_type cvalsh,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 64.
  printf("KCJ: ApplyControlledGateH_Kernel\n");

    
      //printf("KCJ: Simulator_cuda_kernel-ApplyControlledGateH_Kernel start\n");

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                           (G < 6 ? gsize : 32) : (G < 5 ? 8 : 16));

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  idx_type i = (64 * idx_type{blockIdx.x} + threadIdx.x) & 0xffffffffffe0;
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + threadIdx.x % 32;

  for (unsigned k = 0; k < gsize; ++k) {
    rs[k] = *(p0 + xss[k]);
    is[k] = *(p0 + xss[k] + 32);
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      __syncthreads();

      for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }

      __syncthreads();
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      *(p0 + xss[k]) = rn;
      *(p0 + xss[k] + 32) = in;
    }
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateLH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned num_mss, idx_type cvalsh,
    unsigned esize, fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.
  printf("KCJ: ApplyControlledGateLH_Kernel\n");

 //printf("KCJ: Simulator_cuda_kernel-ApplyControlledGateLH_Kernel start\n");
  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + threadIdx.x;

  for (unsigned k = 0; k < gsize; ++k) {
    rs0[threadIdx.x][k] = *(p0 + xss[k]);
    is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
  }

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  for (unsigned k = 0; k < esize; ++k) {
    *(p0 + xss[k]) = rs0[threadIdx.x][k];
    *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
  }
}

template <unsigned G, typename fp_type, typename idx_type>
__global__ void ApplyControlledGateL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, const idx_type* __restrict__ cis,
    unsigned num_mss, idx_type cvalsh, unsigned esize, unsigned rwthreads,
    fp_type* __restrict__ rstate) {
  // blockDim.x must be equal to 32.
  printf("KCJ: ApplyControlledGateL_Kernel\n");
    
          //printf("KCJ: Simulator_cuda_kernel-ApplyControlledGateL_Kernel start\n");


  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned
      rows = G < 4 ? gsize : (sizeof(fp_type) == 4 ?
                              (G < 5 ? gsize : 8) : (G < 6 ? 8 : 4));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  idx_type i = 32 * idx_type{blockIdx.x};
  idx_type ii = i & mss[0];
  for (unsigned j = 1; j < num_mss; ++j) {
    i *= 2;
    ii |= i & mss[j];
  }

  ii |= cvalsh;

  auto p0 = rstate + 2 * ii + cis[threadIdx.x];

  if (threadIdx.x < rwthreads) {
    for (unsigned k = 0; k < gsize; ++k) {
      rs0[threadIdx.x][k] = *(p0 + xss[k]);
      is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
    }
  }

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  for (unsigned k = 0; k < gsize; ++k) {
    unsigned i = tis[threadIdx.x] | qis[k];
    unsigned m = i & 0x1f;
    unsigned n = i / 32;

    rs[k] = rs0[m][n];
    is[k] = is0[m][n];
  }

  for (unsigned s = 0; s < gsize / rows; ++s) {
    if (s > 0) {
      for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
        v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
      }
    }

    unsigned j = 0;

    for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
      fp_type rn = 0;
      fp_type in = 0;

      for (unsigned l = 0; l < gsize; ++l) {
        fp_type rm = v[j++];
        fp_type im = v[j++];
        rn += rs[l] * rm;
        rn -= is[l] * im;
        in += rs[l] * im;
        in += is[l] * rm;
      }

      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs0[m][n] = rn;
      is0[m][n] = in;
    }
  }

  if (threadIdx.x < rwthreads) {
    for (unsigned k = 0; k < esize; ++k) {
      *(p0 + xss[k]) = rs0[threadIdx.x][k];
      *(p0 + xss[k] + 32) = is0[threadIdx.x][k];
    }
  }
}

template <unsigned G, typename fp_type, typename idx_type, typename Op,
          typename cfp_type>
__global__ void ExpectationValueH_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss0,
    const idx_type* __restrict__ mss, unsigned num_iterations_per_block,
    const fp_type* __restrict__ rstate, Op op, cfp_type* __restrict__ result) {
  // blockDim.x must be equal to 64.
          //printf("KCJ: Simulator_cuda_kernel-ExpectationValueH_Kernel start\n");
  printf("KCJ: ExpectationValueH_Kernel\n");

  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows =
      G < 5 ? gsize : (sizeof(fp_type) == 4 ? (G < 6 ? 4 : 8) : 8);

  fp_type rs[gsize], is[gsize];

  __shared__ idx_type xss[64];
  __shared__ fp_type v[2 * gsize * rows];

  if (threadIdx.x < gsize) {
    xss[threadIdx.x] = xss0[threadIdx.x];
  }

  if (G <= 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  __syncthreads();

  double re = 0;
  double im = 0;

  for (unsigned iter = 0; iter < num_iterations_per_block; ++iter) {
    idx_type b = num_iterations_per_block * idx_type{blockIdx.x} + iter;

    idx_type i = (64 * b + threadIdx.x) & 0xffffffffffe0;
    idx_type ii = i & mss[0];
    for (unsigned j = 1; j <= G; ++j) {
      i *= 2;
      ii |= i & mss[j];
    }

    auto p0 = rstate + 2 * ii + threadIdx.x % 32;

    for (unsigned k = 0; k < gsize; ++k) {
      rs[k] = *(p0 + xss[k]);
      is[k] = *(p0 + xss[k] + 32);
    }

    for (unsigned s = 0; s < gsize / rows; ++s) {
      if (s > 0 || iter > 0) {
        __syncthreads();

        for (unsigned m = 0; m < 2 * gsize * rows; m += 64) {
          v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
        }

        __syncthreads();
      }

      unsigned j = 0;

      for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
        fp_type rn = 0;
        fp_type in = 0;

        for (unsigned l = 0; l < gsize; ++l) {
          fp_type rm = v[j++];
          fp_type im = v[j++];
          rn += rs[l] * rm;
          rn -= is[l] * im;
          in += rs[l] * im;
          in += is[l] * rm;
        }

        re += rs[k] * rn;
        re += is[k] * in;
        im += rs[k] * in;
        im -= is[k] * rn;
      }
    }
  }

  __shared__ cfp_type partial1[64];
  __shared__ cfp_type partial2[2];

  partial1[threadIdx.x].re = re;
  partial1[threadIdx.x].im = im;

  auto val = WarpReduce(partial1[threadIdx.x], op);

  if (threadIdx.x % 32 == 0) {
    partial2[threadIdx.x / 32] = val;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    result[blockIdx.x].re = partial2[0].re + partial2[1].re;
    result[blockIdx.x].im = partial2[0].im + partial2[1].im;
  }
}

template <unsigned G, typename fp_type, typename idx_type,
          typename Op, typename cfp_type>
__global__ void ExpectationValueL_Kernel(
    const fp_type* __restrict__ v0, const idx_type* __restrict__ xss,
    const idx_type* __restrict__ mss, const unsigned* __restrict__ qis,
    const unsigned* __restrict__ tis, unsigned num_iterations_per_block,
    const fp_type* __restrict__ rstate, Op op, cfp_type* __restrict__ result) {
  // blockDim.x must be equal to 32.
          //printf("KCJ: Simulator_cuda_kernel-ExpectationValueL_Kernel start\n");
  printf("KCJ: ExpectationValueL_Kernel\n");
  static_assert(G < 7, "gates acting on more than 6 qubits are not supported.");

  constexpr unsigned gsize = 1 << G;
  constexpr unsigned rows = G < 5 ? gsize : (sizeof(fp_type) == 4 ?
                                             (G < 6 ? 4 : 2) : (G < 6 ? 2 : 1));

  fp_type rs[gsize], is[gsize];

  __shared__ fp_type rs0[32][gsize + 1], is0[32][gsize + 1];
  __shared__ fp_type v[2 * gsize * rows];

  if (G < 2) {
    if (threadIdx.x < 2 * gsize * gsize) {
      v[threadIdx.x] = v0[threadIdx.x];
    }
  } else {
    for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
      v[m + threadIdx.x] = v0[m + threadIdx.x];
    }
  }

  double re = 0;
  double im = 0;

  for (idx_type iter = 0; iter < num_iterations_per_block; ++iter) {
    idx_type i = 32 * (num_iterations_per_block * idx_type{blockIdx.x} + iter);
    idx_type ii = i & mss[0];
    for (unsigned j = 1; j <= G; ++j) {
      i *= 2;
      ii |= i & mss[j];
    }

    auto p0 = rstate + 2 * ii + threadIdx.x;

    for (unsigned k = 0; k < gsize; ++k) {
      rs0[threadIdx.x][k] = *(p0 + xss[k]);
      is0[threadIdx.x][k] = *(p0 + xss[k] + 32);
    }

    for (unsigned k = 0; k < gsize; ++k) {
      unsigned i = tis[threadIdx.x] | qis[k];
      unsigned m = i & 0x1f;
      unsigned n = i / 32;

      rs[k] = rs0[m][n];
      is[k] = is0[m][n];
    }

    for (unsigned s = 0; s < gsize / rows; ++s) {
      if (s > 0 || iter > 0) {
        for (unsigned m = 0; m < 2 * gsize * rows; m += 32) {
          v[m + threadIdx.x] = v0[m + 2 * gsize * rows * s + threadIdx.x];
        }
      }

      unsigned j = 0;

      for (unsigned k = rows * s; k < rows * (s + 1); ++k) {
        fp_type rn = 0;
        fp_type in = 0;

        for (unsigned l = 0; l < gsize; ++l) {
          fp_type rm = v[j++];
          fp_type im = v[j++];
          rn += rs[l] * rm;
          rn -= is[l] * im;
          in += rs[l] * im;
          in += is[l] * rm;
        }

        re += rs[k] * rn;
        re += is[k] * in;
        im += rs[k] * in;
        im -= is[k] * rn;
      }
    }
  }

  __shared__ cfp_type partial[32];

  partial[threadIdx.x].re = re;
  partial[threadIdx.x].im = im;

  auto val = WarpReduce(partial[threadIdx.x], op);

  if (threadIdx.x == 0) {
    result[blockIdx.x].re = val.re;
    result[blockIdx.x].im = val.im;
  }
}

}  // namespace qsim

#endif  // SIMULATOR_CUDA_KERNELS_H_
