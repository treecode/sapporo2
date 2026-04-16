#########################################################
# 
#   If you want to get the timings for 1D blocks make sure to fix 'q' to '1' in sapporohostclass.cpp 
#   this ensures 1D thread blocks. When using 2D thread-blocks the timings will be 
#   much better since more threads are used. Also disable the adjustment of 'ni' to multiples of warp sizes
#   to see the effect of half-filled warps/wavefronts. 
#   see the lines with 'BLOCK_TIMING' in sapporohostclass.cpp. In test_performance_blockStep.cpp
#   set 'NMAXTHREADS' to the maximum number of thread block size you like to test (256 or 512
#   is commonly used). 
#       
#       NVIDIA hardware
#
#       Use the profiler to gather the timings, no change needed
#
#       ATI hardware
#       
#       Use the in OpenCL built in event timers. Uncomment '#define DEBUG_PRINT2' 
#       line in 'include/ocldev.h' to execute these timings.
#
#       DONT FORGET TO UNDO THE CODE CHANGES AFTER TIMING RUNS
#
##############################################################

NVIDIA=1

if [ ${NVIDIA} -eq 1 ] 
then 
  echo  "NVIDIA"

  OUTNAME="perf_5.5_k20c"

  CUDA_VISIBLE_DEVICES=0 OPENCL_PROFILE_CSV=1 OPENCL_PROFILE=1 OPENCL_PROFILE_CONFIG=./profiler.conf OPENCL_PROFILE_LOG=/tmp/$OUTNAME".tsv" ../test_performance_blockStep_ocl 131072 "../OpenCL/kernels4th.cl"

  cat /tmp/$OUTNAME".tsv" | grep eval | awk -F "," '{print "Executing on command queue: Kernel: dev_evaluate_gravity\t Took: " 1000*$3 " \tx Threads: " $7}' > "$OUTNAME"_ocl_128k.txt

  sleep 30

#   CUDA_VISIBLE_DEVICES=0 OPENCL_PROFILE_CSV=1 OPENCL_PROFILE=1 OPENCL_PROFILE_CONFIG=./profiler.conf OPENCL_PROFILE_LOG=/tmp/$OUTNAME".tsv" ../test_performance_blockStep_ocl 262144 "../OpenCL/kernels4th.cl"
# 
#   cat /tmp/$OUTNAME".tsv" | grep eval | awk -F "," '{print "Executing on command queue: Kernel: dev_evaluate_gravity\t Took: " 1000*$3 " \tx Threads: " $7}' > "$OUTNAME"_ocl_256k.txt

  sleep 30

  CUDA_VISIBLE_DEVICES=0 CUDA_PROFILE_CSV=1 CUDA_PROFILE=1 CUDA_PROFILE_CONFIG=./profiler.conf CUDA_PROFILE_LOG=/tmp/$OUTNAME".tsv" ../test_performance_blockStep_cuda 131072 "../CUDA/kernels.ptx"

  cat /tmp/$OUTNAME".tsv" | grep eval | awk -F "," '{print "Executing on command queue: Kernel: dev_evaluate_gravity\t Took: " 1000*$3 " \tx Threads: " $7}' > "$OUTNAME"_cuda_128k.txt

  sleep 30

#   CUDA_VISIBLE_DEVICES=0 CUDA_PROFILE_CSV=1 CUDA_PROFILE=1 CUDA_PROFILE_CONFIG=./profiler.conf CUDA_PROFILE_LOG=/tmp/$OUTNAME".tsv" ../test_performance_blockStep_cuda 262144 "../CUDA/kernels.ptx"
# 
#   cat /tmp/$OUTNAME".tsv" | grep eval | awk -F "," '{print "Executing on command queue: Kernel: dev_evaluate_gravity\t Took: " 1000*$3 " \tx Threads: " $7}' > "$OUTNAME"_cuda_256k.txt

else
  echo "AMD/Other"

#For the ATI/Other card we have to do it slightly different
  ../test_performance_blockStep_ocl 131072 2>&1 | tee /tmp/log.txt > /dev/null
  more /tmp/log.txt | grep Took | grep eval > sum_ati_7970_128k.txt

#   ../test_performance_blockStep_ocl 262144 2>&1 | tee /tmp/log.txt > /dev/null
#   more /tmp/log.txt | grep Took | grep eval > sum_ati_7970_256k.txt
  

fi