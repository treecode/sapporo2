CXX = g++
CC  = gcc
LD  = g++ 
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

SAPPOROPATH=../lib/
SAPLIB2 = sapporo
SAPLIB = lib$(SAPLIB2).a


CUDA_TK  = /usr/local/cuda

OFLAGS = -g -Wall -fopenmp 
CXXFLAGS =  -fPIC $(OFLAGS) -I$(CUDA_TK)/include 

LDFLAGS = -lcuda -fopenmp  

INCLUDEPATH = $(SAPPOROPATH)/include 
CXXFLAGS  += -I$(INCLUDEPATH) -I./ -I $(SAPPOROPATH)


SRCPATH = src
SRC = test_gravity_block.cpp  test_gravity_block_6th.cpp test_performance_rangeN.cpp test_performance_blockStep.cpp test_performance_rangeN_6th.cpp
OBJ = $(SRC:%.cpp=%.o)

PROG = test_gravity_block_cuda test_gravity_block_6th_cuda test_performance_rangeN_cuda test_performance_blockStep_cuda test_performance_rangeN_6th_cuda

all: $(OBJ) $(PROG) kernels


kernels:
	ln -s $(SAPPOROPATH)/CUDAKernels/ CUDA/

#$(PROG): $(OBJ)
#	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -lsapporo

test_gravity_block_cuda : test_gravity_block.o
	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2)

test_gravity_block_6th_cuda : test_gravity_block_6th.o
	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2)    

test_performance_rangeN_cuda : test_performance_rangeN.o
	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2)  

test_performance_blockStep_cuda : test_performance_blockStep.o
	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) 

test_performance_rangeN_6th_cuda : test_performance_rangeN_6th.o
	$(LD) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) 

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o *.ptx *.a $(PROG)


$(OBJ): $(SAPPOROPATH)/$(SAPLIB)








