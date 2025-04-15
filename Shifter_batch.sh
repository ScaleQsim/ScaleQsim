#!/bin/bash
#SBATCH -N 32            
#SBATCH -C gpu&hbm80g                     
#SBATCH --gpus-per-node=4         
#SBATCH -q regular                  
#SBATCH -J qsim_KCJ_q38                
#SBATCH --mail-user=changjong5238@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 00:05:00
#SBATCH -A m1248


# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

conda activate kcj_qsim_mpi

module  load  PrgEnv-nvidia  cudatoolkit/12.2  craype-accel-nvidia80  python/3.9
module load cray-mpich/8.1.27
module unload craype-accel-nvidia80
module load craype-accel-nvidia80
module load nccl/2.21.5

module unload craype-accel-nvidia80
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

export LD_PRELOAD=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi/lib/libmpi_gtl_cuda.so

export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export CONDA_PREFIX=/global/homes/s/sgkim/.conda/envs/kcj_qsim_mpi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH:/opt/cray/pe/mpich/8.1.27/ofi/nvidia/20.7/lib-abi-mpich:/opt/cray/pe/mpich/8.1.27/gtl/lib


export CUDA_VISIBLE_DEVICES=0,1,2,3



srun --ntasks=32 --ntasks-per-node=32 --mpi=pmi2 python qft.py > q40_n16_test.txt 2>&1
