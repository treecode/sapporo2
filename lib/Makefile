CXX = g++
CC  = gcc
LD  = g++ 
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

CUDA_TK  ?= /usr/local/cuda


testRunFlags1=
testRunFlags2=
testRunFlags3=

#Check for the defines 

ifdef NTHREADS
	testRunFlags1="-D NTHREADS=$(NTHREADS)"
	testRunFlags3="-D TIMING_STATS=1"
endif

ifdef NBLOCKS_PER_MULTI
	testRunFlags2="-D NBLOCKS_PER_MULTI=$(NBLOCKS_PER_MULTI)"
	testRunFlags3="-D TIMING_STATS=1"
endif

OFLAGS = -g -O3 -Wall -Wextra -Wstrict-aliasing=2 -fopenmp

#Use below if compiling with CPU_SUPPORT (SSE)
#CXXFLAGS += ${testRunFlags} -fPIC $(OFLAGS) -I$(CUDA_TK)/include -msse4
CXXFLAGS += ${testRunFlags} -fPIC $(OFLAGS) -I$(CUDA_TK)/include

testRunFlags= $(testRunFlags1) $(testRunFlags2) $(testRunFlags3)
$(info $(testRunFlags))

NVCC      = $(CUDA_TK)/bin/nvcc  


#NVCCFLAGS = -arch sm_35
NVCCFLAGS ?= -arch sm_30
#NVCCFLAGS = -arch sm_20
NVCCFLAGS +=  ${testRunFlags}

# Use with Mac OS X
# NVCCFLAGS = -arch sm_12 -Xcompiler="-Duint=unsigned\ int"

LDFLAGS = -lcuda -fopenmp 


INCLUDEPATH = ./include
CXXFLAGS  += -I$(INCLUDEPATH) -I./
NVCCFLAGS += -I$(INCLUDEPATH) -I./

INTERFACEPATH =./interfaces

CUDAKERNELSPATH = ./CUDAKernels
CUDAKERNELS = kernels.cu

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

$(INCLUDEPATH)/%.ptxh: $(CUDAKERNELSPATH)/%.ptx
	xxd -i $< $@

clean:
	/bin/rm -rf *.o *.ptx *.a 
	cd $(INTERFACEPATH); /bin/rm -rf *.o; cd ..
	cd $(CUDAKERNELSPATH); /bin/rm -rf *.ptx; cd ..
	rm -f *.ptxh $(INCLUDEPATH)/*.ptxh

$(OBJ): $(INCLUDEPATH)/*.h


sapporohostclass.o : $(INCLUDEPATH)/kernels.ptxh $(INCLUDEPATH)/sapporohostclass.h $(INCLUDEPATH)/sapdevclass.h $(INCLUDEPATH)/defines.h
$(CUDAKERNELSPATH)/kernels.ptx : $(INCLUDEPATH)/defines.h

libsapporo.a : sapporohostclass.o








