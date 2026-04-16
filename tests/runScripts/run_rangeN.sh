FILE=cuda_rangeN_K20c.txt
#FILE=ocl_rangeN_GTX680.txt
rm ${FILE}

#for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4092
for i in 1 2 4 8 16 32 
  do
          #for j in 1 2 3 4 5 6
          for j in 1
            do
                N=$(($i * 1024))
                echo $N $j


         #Single precision CUDA 4th order
         CUDA_VISIBLE_DEVICES=0 ../test_performance_rangeN_cuda $N ../CUDA/kernels.ptx 1 1 $j 2>&1 |  grep TIMING >> ${FILE}

         #Double precision CUDA 4th order
         #CUDA_VISIBLE_DEVICES=1 ../test_performance_rangeN_cuda $N ../CUDA/kernels4thDP.ptx 1 1 $j 2>&1 |  grep TIMING >> ${FILE}

         #Single precision OpenCL 4th order
         #CUDA_VISIBLE_DEVICES=1 ../test_performance_rangeN_ocl $N ../OpenCL/kernels4th.cl 1 0 $j 2>&1 |  grep TIMING >> ${FILE}    

         #Double precision OpenCL 4th order
         #CUDA_VISIBLE_DEVICES=1 ../test_performance_rangeN_ocl $N ../OpenCL/kernels4th.cl 1 1 $j 2>&1 |  grep TIMING >> ${FILE}    

         #sleep 10
         done
done


