#!/bin/sh

#flags=-DNGB
flags=""

CUDAINC="-I/usr/local/cuda/include -I/usr/local/cuda-sdk/common/inc"
CUDALIB="-L/usr/local/cuda/lib64 -lcuda"
# #  -L/usr/local/cuda-sdk/lib -lcutil"

# # nvcc -O0 -g  --device-emulation $flags -maxrregcount=32  -o host_evaluate_gravity.cu_o -c host_evaluate_gravity.cu -I/home/nvidia/NVIDIA_CUDA_SDK/common/inc/

# nvcc -O0 -g -D_DEBUG $flags -maxrregcount=32  -o host_evaluate_gravity.cu_o -c host_evaluate_gravity.cu $CUDAINC

# g++ -O3 $flags -g -c GPUWorker.cc $CUDAINC
# g++ -O3 $flags -g -c sapporo.cpp $CUDAINC
# g++ -O3 $flags -g -c send_fetch_data.cpp $CUDAINC
# g++ -O3 $flags -g -c sapporoG6lib.cpp $CUDAINC
# /bin/rm -rf libsapporo.a
# ar qv ./libsapporo.a sapporo.o send_fetch_data.o sapporoG6lib.o host_evaluate_gravity.cu_o GPUWorker.o
# ranlib ./libsapporo.a 


make clean; make;

g++ -O3 $flags -g -o test_gravity_block_cuda test_gravity_block.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_performance_cuda test_performance.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_performance_rangeN_cuda test_performance_rangeN.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_gravity_block_6th_cuda test_gravity_block_6th.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_performance_rangeN_cudaDP test_performance_rangeN_DP.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL

g++ -O3 $flags -g -o test_gravity_block_cudaDP test_gravity_blockDP.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL

rm  *.o

make -f Makefile_ocl
flags=" -D _OCL_ "
g++ -O3 $flags -g -o test_gravity_block_ocl test_gravity_block.cpp -lsapporo_ocl -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_performance_ocl test_performance.cpp -lsapporo_ocl -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_performance_rangeN_ocl test_performance_rangeN.cpp -lsapporo_ocl -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
g++ -O3 $flags -g -o test_gravity_block_6th_ocl test_gravity_block_6th.cpp -lsapporo_ocl -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL



