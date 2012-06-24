#!/bin/bash


#This script is used to compile using different block sizes for performance testing
#Its in a non-workable state atm

#flags=-DNGB
#flags=""

CUDAINC="-I/usr/local/cuda/include -I/usr/local/cuda-sdk/common/inc"
CUDALIB="-L/usr/local/cuda/lib64 -lcuda"


rm blockSizeTiming_ocl_128k.txt

for j in {32..1024..32}
#for j in 32
do
	echo $j
	for k in {1,2,4,6,8,10}
	#for k in 1
	do
	
	i=$j

	make clean; 
	curNTHREADS=$j curNPIPES=$i curNMULTI=$k make -f Makefile_ocl_bs;
	flags=" -D NTHREADS=$i -D NPIPES=$j -D NBLOCKS_PER_MULTI=$k"

	#g++ -O3 $flags -g -o test_performance_rangeN_cuda test_performance_rangeN.cpp -lsapporo -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
	#CUDA_VISIBLE_DEVICES=1	./test_performance_rangeN_cuda 131072 2>> blockSizeTiming_cuda_128k.txt

	g++ -O3 $flags -g -o test_performance_rangeN_ocl test_performance_rangeN.cpp -lsapporo_ocl -L.  $CUDAINC $CUDALIB -fopenmp  -lOpenCL
	#CUDA_VISIBLE_DEVICES=1	./test_performance_rangeN_ocl 131072 2>> blockSizeTiming_ocl_128k.txt
	CUDA_VISIBLE_DEVICES=0	./test_performance_rangeN_ocl 131072 2>> blockSizeTiming_ocl_128k.txt

	done
done



