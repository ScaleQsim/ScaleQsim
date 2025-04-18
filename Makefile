EIGEN_PREFIX = "3bb6a48d8c171cf20b5f8e48bfb4e424fbd4f79e"
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/"

TARGETS = qsim
TESTS = run-cxx-tests

CXX = g++
NVCC = nvcc
HIPCC = hipcc

# KCJ
# KCJ - Conda 환경의 MPI 사용
MPICXX = $(MPI_HOME)/bin/mpicxx
MPICC = $(MPI_HOME)/bin/mpicc
CC = $(MPI_HOME)/bin/mpicc
#CXX = $(MPI_HOME)/bin/mpicxx

export MPICC=$(MPI_HOME)/bin/mpicc
export MPICXX=$(MPI_HOME)/bin/mpicxx
export CC=$(MPI_HOME)/bin/mpicc
#export CXX=$(MPI_HOME)/bin/mpicxx


CXXFLAGS = -O3 -fopenmp
ARCHFLAGS = -march=native
NVCCFLAGS = -O3 -I/global/common/software/nersc9/nccl/2.21.5/include \
            -L/global/common/software/nersc9/nccl/2.21.5/lib -lnccl \
            -L$(MPI_HOME)/lib -lmpi

HIPCCFLAGS = -O3

# CUQUANTUM_ROOT should be set.
CUSTATEVECFLAGS = -I$(CUQUANTUM_ROOT)/include -L${CUQUANTUM_ROOT}/lib -L$(CUQUANTUM_ROOT)/lib64 -lcustatevec -lcublas

PYBIND11 = true

export CXX
export CXXFLAGS
export ARCHFLAGS
export NVCC
export NVCCFLAGS
export CUSTATEVECFLAGS
export HIPCC
export HIPCCFLAGS

ifeq ($(PYBIND11), true)
  TARGETS += pybind
  TESTS += run-py-tests
endif

.PHONY: all
all: $(TARGETS)

.PHONY: qsim
qsim:
	$(MAKE) -C apps/ qsim

.PHONY: qsim-cuda
qsim-cuda:
	$(MAKE) -C apps/ qsim-cuda NVCCFLAGS="$(NVCCFLAGS)"

.PHONY: qsim-custatevec
qsim-custatevec:
	$(MAKE) -C apps/ qsim-custatevec

.PHONY: qsim-hip
qsim-hip:
	$(MAKE) -C apps/ qsim-hip

.PHONY: pybind
pybind:
	$(MAKE) -C pybind_interface/ pybind

.PHONY: cxx-tests
cxx-tests: eigen
	$(MAKE) -C tests/ cxx-tests

.PHONY: cuda-tests
cuda-tests:
	$(MAKE) -C tests/ cuda-tests NVCCFLAGS="$(NVCCFLAGS)"

.PHONY: custatevec-tests
custatevec-tests:
	$(MAKE) -C tests/ custatevec-tests

.PHONY: hip-tests
hip-tests:
	$(MAKE) -C tests/ hip-tests

.PHONY: run-cxx-tests
run-cxx-tests: cxx-tests
	$(MAKE) -C tests/ run-cxx-tests

.PHONY: run-cuda-tests
run-cuda-tests: cuda-tests
	$(MAKE) -C tests/ run-cuda-tests NVCCFLAGS="$(NVCCFLAGS)"

.PHONY: run-custatevec-tests
run-custatevec-tests: custatevec-tests
	$(MAKE) -C tests/ run-custatevec-tests

.PHONY: run-hip-tests
run-hip-tests: hip-tests
	$(MAKE) -C tests/ run-hip-tests

PYTESTS = $(shell find qsimcirq_tests/ -name '*_test.py')

.PHONY: run-py-tests
run-py-tests: pybind
	for exe in $(PYTESTS); do if ! python3 -m pytest $$exe; then exit 1; fi; done

.PHONY: run-tests
run-tests: $(TESTS)

eigen:
	$(shell\
		rm -rf eigen;\
		wget $(EIGEN_URL)/$(EIGEN_PREFIX)/eigen-$(EIGEN_PREFIX).tar.gz;\
		tar -xf eigen-$(EIGEN_PREFIX).tar.gz && mv eigen-$(EIGEN_PREFIX) eigen;\
		rm eigen-$(EIGEN_PREFIX).tar.gz;)

.PHONY: clean
clean:
	rm -rf eigen;
	-$(MAKE) -C apps/ clean
	-$(MAKE) -C tests/ clean
	-$(MAKE) -C pybind_interface/ clean
