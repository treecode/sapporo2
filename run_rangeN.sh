rm cuda_multiGPUTiming_GTX580_41.txt

#for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4092 8192 
#for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4092
for i in 1 2 4 8 16 32 64 128
  do
          #for j in 1 2 3 4 5 6
          for j in 1
          #for j in 4
          do
                 N=$(($i * 1024))
                echo $N $j
         #./test_performance_rangeN_cuda $N $j 2>&1 | tee custlog.txt |  grep TIMING >> cuda_multiGPUTiming_GTX480.txt
         ./test_performance_rangeN_cuda $N $j 2>&1 |  grep TIMING >> cuda_multiGPUTiming_GTX580_41.txt

         sleep 10
 done
  done


