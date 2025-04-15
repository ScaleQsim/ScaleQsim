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

#include "pybind_main_cuda.h"

#ifdef __NVCC__
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include </global/common/software/nersc9/nccl/2.21.5/include/nccl.h> //KCJ
#elif __HIP__
  #include <hip/hip_runtime.h>
  #include "cuda2hip.h"
#endif

#include "../../lib/simulator_cuda.h"
#include </opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include/mpi.h>

/*
namespace qsim {
  using Simulator = SimulatorCUDA<float>;

  struct Factory {
    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    Factory(
      unsigned num_sim_threads,
      unsigned num_state_threads,
      unsigned num_dblocks
    ) {
      ss_params.num_threads = num_state_threads;
      ss_params.num_dblocks = num_dblocks;
    }
    StateSpace CreateStateSpace() const {
     // printf("KCJ: CreateStateSpace");
      return StateSpace(ss_params);
    }

    Simulator CreateSimulator() const {
      //printf("KCJ: CreateSimultor");
      return Simulator();
    }

    StateSpace::Parameter ss_params;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}
*/

namespace qsim {
  using Simulator = SimulatorCUDA<float>;

  struct Factory {
    using Simulator = qsim::Simulator;
    using StateSpace = Simulator::StateSpace;

    Factory(unsigned num_sim_threads, unsigned num_state_threads, unsigned num_dblocks) {
      ss_params.num_threads = num_state_threads;
      ss_params.num_dblocks = num_dblocks;


      // GPU 수 초기화
        cudaError_t err_before = cudaGetLastError();
        if (err_before != cudaSuccess) {
            printf("ERROR: pybind 1 - CUDA error before kernel launch on GPU %lu: %s\n", 0, cudaGetErrorString(err_before));
    }
     
      cudaGetDeviceCount(&num_gpus);
    }

    StateSpace CreateStateSpace() const {
      std::vector<StateSpace> state_spaces;
      std::vector<ncclComm_t> nccl_comms(num_gpus);

      // NCCL 초기화
      ncclCommInitAll(nccl_comms.data(), num_gpus, nullptr);

      for (int i = 0; i < num_gpus; ++i) {
        state_spaces.emplace_back(ss_params, i);
      }

      return StateSpace(state_spaces, nccl_comms);
    }

    Simulator CreateSimulator() const {
      std::vector<StateSpace> state_spaces;
      std::vector<ncclComm_t> nccl_comms(num_gpus);

      // NCCL 초기화
      ncclCommInitAll(nccl_comms.data(), num_gpus, nullptr);

      for (int i = 0; i < num_gpus; ++i) {
        state_spaces.emplace_back(ss_params, i);
      }

      return Simulator(state_spaces, nccl_comms);
    }

    StateSpace::Parameter ss_params;
    int num_gpus;
  };

  inline void SetFlushToZeroAndDenormalsAreZeros() {}
  inline void ClearFlushToZeroAndDenormalsAreZeros() {}
}


#include "../pybind_main.cpp"
