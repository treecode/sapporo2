#Process the logdata of the specified files
#Bin and average the timing data
#and each file becomes one line

import matplotlib.pyplot as plt
import numpy
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


#fileNames =  ["profLog_128k.csv", "profLog_512k.csv",  "/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_128k.csv",  "/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_512k.csv" ]

fileNames =  ["profLog_128k.csv",   "/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_128k.csv", 
"profLog_128k_ocl.csv"]

legendTitles = ["128k CUDA", "128k SAP1.6", "128k OCL"]

#fileNames =  ["/home/jbedorf/papers/BGPZ2012Sap2/data/performancePerThread/sapporo16/profLog_128k.csv" ]


npipes = 256

startIdx = 1

plotResults = []
xAxisResults = []

for fileNameIn in fileNames:


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
      
      
  timing = {}
  timing_count = {}



  for i in range(npipes+1):
    timing[i] = 0
    timing_count[i] = 0

  for line in fileIn.readlines():
    if "eval" in line:
      
      items = line.split(',')
      
      nthreads = int(items[threadblocksizeX_IDX])
      time     = float(items[gputime_IDX])
      
      timing[nthreads] += time
      timing_count[nthreads] += 1
      

  xdata =  numpy.arange(startIdx,npipes+1)
  ydata =  numpy.arange(startIdx,npipes+1)

  for x in xdata:
    ydata[x-1] = (timing[x] / timing_count[x])
    #ydata[x-1] = (timing[x] / timing_count[x]) / x
    #ydata[startIdx-x-1] = (timing[x] / timing_count[x]) / x

  plotResults.append(ydata)
  xAxisResults.append(xdata)

plt.figure()
ax = plt.subplot(111)
ax.xaxis.grid(True, 'major')
#ax.set_aspect(1)
for i in range(0, len(plotResults)):
  plt.plot(xAxisResults[i], plotResults[i], label=legendTitles[i])


ax.xaxis.set_major_locator(MultipleLocator(32))
#majloc = ticker.IndexLocator( 8, 0 ),
plt.xlabel('Number of active particles\n(with newlines!)')
plt.ylabel('Duration of gravity computation\n[in msec]', multialignment='center')
#ylabel('this is another!')
plt.legend()
plt.show()

    