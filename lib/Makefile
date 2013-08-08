CXX = g++
CC  = gcc
LD  = g++ 
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

CUDA_TK  = /usr/local/cuda

#OFLAGS = -O3 -g -Wall -fopenmp  -D_OCL_
#OFLAGS = -O3 -g -Wall -fopenmp
#OFLAGS = -g -Wall -fopenmp -D_OCL_

#OFLAGS = -g -Wall -fopenmp  DEFAULT

OFLAGS = -g -O3 -Wall -Wextra -Wstrict-aliasing=2 -fopenmp

CXXFLAGS =  -fPIC $(OFLAGS) -I$(CUDA_TK)/include -msse4

#Below line is used when compiling for testing
#CXXFLAGS =  -fPIC $(OFLAGS) -I$(CUDA_TK)/include -D NTHREADS=${curNTHREADS} -D NPIPES=${curNPIPES} -D NBLOCKS_PER_MULTI=${curNMULTI} 


# NVCCFLAGS = -D_DEBUG -O0 -g -I$(CUDA_SDK)/common/inc -arch=sm_12 --maxrregcount=64  --opencc-options -OPT:Olimit=0 -I$(CUDPP)/cudpp/include
NVCC      = $(CUDA_TK)/bin/nvcc  


NVCCFLAGS = -arch sm_35
#NVCCFLAGS = -arch sm_20 
#NVCCFLAGS = -arch sm_30
#NVCCFLAGS = -arch sm_30 -D NTHREADS=${curNTHREADS} -D NPIPES=${curNPIPES} -D NBLOCKS_PER_MULTI=${curNMULTI}

# Use with Mac OS X
# NVCCFLAGS = -arch sm_12 -Xcompiler="-Duint=unsigned\ int"

LDFLAGS = -lcuda -fopenmp 


INCLUDEPATH = ./include
CXXFLAGS  += -I$(INCLUDEPATH) -I./
NVCCFLAGS += -I$(INCLUDEPATH) -I./

INTERFACEPATH =./interfaces

CUDAKERNELSPATH = ./CUDAKernels
CUDAKERNELS = kernels4th.cu  kernels4thDP.cu kernels6thDP.cu kernelsG5DS.cu kernelsG5SP.cu

CUDAPTX = $(CUDAKERNELS:%.cu=$(CUDAKERNELSPATH)/%.ptx)

SRCPATH = src
SRC = sapporohostclass.cpp sapporoG6lib.cpp sapporoYeblib.cpp sapporoG5lib.cpp sapporo6thlib.cpp
OBJ = $(SRC:%.cpp=%.o)

LIBOBJ = sapporohostclass.o $(INTERFACEPATH)/sapporoG6lib.o $(INTERFACEPATH)/sapporoYeblib.o $(INTERFACEPATH)/sapporoG5lib.o
TARGET = libsapporo.a


all:	  $(OBJ) $(CUDAPTX) $(TARGET)
kernels:  $(CUDAPTX)


$(TARGET): $(LIBOBJ)
	ar qv $@ $^        

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDAKERNELSPATH)/%.ptx: $(CUDAKERNELSPATH)/%.cu
	$(NVCC) $(NVCCFLAGS) -ptx $< -o $@

clean:
	/bin/rm -rf *.o *.ptx *.a 
	cd $(INTERFACEPATH); /bin/rm -rf *.o; cd ..
	cd $(CUDAKERNELSPATH); /bin/rm -rf *.ptx; cd ..

$(OBJ): $(INCLUDEPATH)/*.h


sapporohostclass.o : $(INCLUDEPATH)/sapporohostclass.h $(INCLUDEPATH)/sapdevclass.h $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernels4th.ptx : $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernels4thDP.ptx : $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernels6thDP.ptx : $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernelsG5DS.ptx : $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernelsG5SP.ptx : $(INCLUDEPATH)/defines.h

libsapporo.a : sapporohostclass.o







