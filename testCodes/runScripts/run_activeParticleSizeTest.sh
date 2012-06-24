OUTNAME="perf_5.0"

CUDA_VISIBLE_DEVICES=1,0 OPENCL_PROFILE_CSV=1 OPENCL_PROFILE=1 OPENCL_PROFILE_CONFIG=./profiler.conf OPENCL_PROFILE_LOG=$OUTNAME"_ocl_128k.tsv" ../test_performance_blockStep_ocl 131072 "../OpenCL/kernels4th.cl"

sleep 30

CUDA_VISIBLE_DEVICES=1,0 OPENCL_PROFILE_CSV=1 OPENCL_PROFILE=1 OPENCL_PROFILE_CONFIG=./profiler.conf OPENCL_PROFILE_LOG=$OUTNAME"_ocl_256k.tsv" ../test_performance_blockStep_ocl 262144 "../OpenCL/kernels4th.cl"

sleep 30

CUDA_VISIBLE_DEVICES=1,0 CUDA_PROFILE_CSV=1 CUDA_PROFILE=1 CUDA_PROFILE_CONFIG=./profiler.conf CUDA_PROFILE_LOG=$OUTNAME"_cuda_128k.tsv" ../test_performance_blockStep_cuda 131072 "../CUDA/kernels4th.ptx"

sleep 30

CUDA_VISIBLE_DEVICES=1,0 CUDA_PROFILE_CSV=1 CUDA_PROFILE=1 CUDA_PROFILE_CONFIG=./profiler.conf CUDA_PROFILE_LOG=$OUTNAME"_cuda_256k.tsv" ../test_performance_blockStep_cuda 262144 "../CUDA/kernels4th.ptx"
