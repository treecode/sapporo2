#!/bin/bash


#This script is used to compile using different block sizes for performance testing

CUDAINC="-I/usr/local/cuda/include"
CUDALIB="-L/usr/local/cuda/lib64 -lcuda"


LOGFILE1=blockSizeTiming_cuda_128k.txt
LOGFILE2=blockSizeTiming_ocl_128k.txt
rm $LOGFILE1 $LOGFILE2

for j in {32..1024..32}
#for j in 32
do
	echo $j
	for k in {1,2,4,6,8,10}
	#for k in 1
	do
	
		echo "Testing nblocks: " $k " and nthreads: " $j
		#remove the library
		rm ../../lib/libsapporo.a
		rm ../../lib/sapporo_ocl.a	

		#Try to make the code, will fail, but needed to get the compiler to the right rebuild
		cd ../
		rm test_performance_rangeN_cuda test_performance_rangeN_ocl 
		make -f Makefile     test_performance_rangeN_cuda
		make -f Makefile_ocl test_performance_rangeN_ocl
		
		#Rebuild the library using the settings
		cd ../lib/
		make clean
		NTHREADS=$j NBLOCKS_PER_MULTI=$k make -f Makefile;
		NTHREADS=$j NBLOCKS_PER_MULTI=$k make -f Makefile_ocl;

		cd ../testCodes/
		make clean; 
		flags=" -D NTHREADS=$i -D NPIPES=$j -D NBLOCKS_PER_MULTI=$k"

		make -f Makefile_ocl test_performance_rangeN_ocl;
		make -f Makefile test_performance_rangeN_cuda;

				
		CUDA_VISIBLE_DEVICES=0	./test_performance_rangeN_cuda 131072 2>> $LOGFILE1
		CUDA_VISIBLE_DEVICES=0	./test_performance_rangeN_ocl  131072 2>> $LOGFILE2
    
                cd runScripts

	done
done



