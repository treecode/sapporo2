#Process the logdata of the specified files
#Bin and average the timing data
#and each file becomes one line

import matplotlib.pyplot as plt
import numpy
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


#fileNames =  ["profLog_128k.csv", "profLog_512k.csv",  "/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_128k.csv",  "/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_512k.csv" ]


NRange = [1, 2, 4, 8, 16, 32, 64, 128 ,256, 512, 1024] 




#FileTypeRange = ["_ocl", "_cuda", "_sap16"]
FileTypeRange = ["ocl", "cuda", "ocl_gpu3", "cuda_gpu3", "cuda_580", "ocl_580"]
#FileTypeRange = ["ocl", "cuda", "cuda_580", "ocl_580"]


xAxisResults = []

gravityEval = []

for kind in FileTypeRange:
  xValues = []
  plotResultsD2H  = []
  plotResultsH2D  = []
  plotResultsEVAL = []
  plotResultsRED = []
  
  for N in NRange:
  
    fileNameIn = "meetData/rangeN_" + str(N) + "_" + kind + ".csv"
    
    gputime_IDX = -1
    gridsizeX_IDX = -1
    threadblocksizeX_IDX = -1
    threadblocksizeY_IDX = -1


    fileIn = open(fileNameIn, "r")


    #First skip the comments
    while True:
      line = fileIn.readline()
      if(line[0] == "#"):
        print line
      else:
        #First non comment line specifies column headers
        items = line.split(',')
        idx = 0
        for item in items:
          if(item == "gputime"):
            gputime_IDX = idx
          if(item == "gridsizeX"):
            gridsizeX_IDX = idx
          if(item == "threadblocksizeX"):
            threadblocksizeX_IDX = idx
          if(item == "workgroupsizeX"):
            threadblocksizeX_IDX = idx          
          if(item == "threadblocksizeY"):
            threadblocksizeY_IDX = idx
          if(item == "workgroupsizeY"):
            threadblocksizeY_IDX = idx              
          idx +=1

        break
        
    sumD2H  = 0
    sumH2D  = 0
    sumEVAL = 0
    sumRED  = 0
    for line in fileIn.readlines():
      items = line.split(',')
      time     = float(items[gputime_IDX])

      if "evaluate" in line:
          sumEVAL += time
      if "reduce" in line:
          sumRED  += time      
      if "memcpyHtoD" in line:
          sumH2D  += time      
      if "memcpyDtoH" in line:
          sumD2H  += time         

    plotResultsD2H.append(sumD2H)
    plotResultsH2D.append(sumH2D)
    plotResultsEVAL.append(sumEVAL)
    plotResultsRED.append(sumRED)
    xValues.append(N*1024)
    
  """
  print xValues
  print plotResultsEVAL
  plt.figure()
  ax = plt.subplot(111)
  plt.plot(xValues, plotResultsEVAL, label="Gravity")
  plt.plot(xValues, plotResultsRED,  label="Reduce")
  plt.plot(xValues, plotResultsD2H,  label="d2h")
  plt.plot(xValues, plotResultsH2D,  label="h2d")
  
  ax.set_yscale('log')
  ax.set_xscale('log')
  
  plt.xlabel('N')
  plt.ylabel('Time\n[in msec]', multialignment='center')
  #ylabel('this is another!')
  plt.legend(loc=2)
  plt.show()
  """    
  
  #Store
  gravityEval.append(plotResultsEVAL)
  xAxisResults.append(xValues)
  

#xdata =  numpy.arange(startIdx,npipes+1)
#ydata =  numpy.arange(startIdx,npipes+1)

#for x in xdata:
  #ydata[x-1] = (timing[x] / timing_count[x])
  ##ydata[x-1] = (timing[x] / timing_count[x]) / x
  ##ydata[startIdx-x-1] = (timing[x] / timing_count[x]) / x

print gravityEval


plt.figure()
ax = plt.subplot(111)

for i in range(0, len(gravityEval)):
  plt.plot(xAxisResults[i], gravityEval[i], label=FileTypeRange[i])
  #plt.plot(xAxisResults[i], gravityEval[i], label=str(i))
  print FileTypeRange[i]
  print xAxisResults[i]
  print gravityEval[i]
  

ax.set_yscale('log')
ax.set_xscale('log')

plt.xlabel('N')
plt.ylabel('Time\n[in msec]', multialignment='center')
#plt.legend(loc=2)
plt.legend()
plt.show()



    