
for i in  512 1024 2048 4092
#for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 
  do
	  N=$(($i * 1024))
	echo $N
	OPENCL_PROFILE_CSV=1 OPENCL_PROFILE=1 OPENCL_PROFILE_CONFIG=./profiler.conf OPENCL_PROFILE_LOG=rangeN_${i}_ocl_580.csv ./test_performance_rangeN_ocl $N

	CUDA_PROFILE_CSV=1 CUDA_PROFILE=1 CUDA_PROFILE_CONFIG=./profiler.conf CUDA_PROFILE_LOG=rangeN_${i}_cuda_580.csv ./test_performance_rangeN_cuda $N
  done


