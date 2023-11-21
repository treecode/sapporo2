CXX ?= g++
CC  ?= gcc
LD  ?= g++ 
F90  ?= ifort

.SUFFIXES: .o .cpp .ptx .cu

SAPPOROPATH=../
SAPLIB2 = sapporo
SAPLIB = lib$(SAPLIB2).a
SAPLIBG6 = sapporoG6


CUDA_TK  ?= /usr/local/cuda

OFLAGS = -g  -O3 -Wall -Wextra -Wstrict-aliasing=2 -fopenmp
CXXFLAGS +=  -fPIC -fopenmp $(OFLAGS) -I$(CUDA_TK)/include 

LDFLAGS += -lcuda -L$(CUDA_TK)/lib64

INCLUDEPATH = $(SAPPOROPATH)/include 
CXXFLAGS  += -I$(INCLUDEPATH) -I./ -I $(SAPPOROPATH)/src


SRCPATH = src
SRC = test_gravity_block.cpp test_gravity_block_g5.cpp test_gravity_block_6th.cpp test_performance_rangeN.cpp test_performance_blockStep.cpp test_performance_rangeN_6th.cpp test_performance_rangeN_g5.cpp test_integrator.cpp
OBJ = $(SRC:%.cpp=%.o)

PROG = test_gravity_block_cuda test_gravity_block_g5_cuda test_gravity_block_6th_cuda test_performance_rangeN_cuda test_performance_blockStep_cuda test_performance_rangeN_6th_cuda test_performance_rangeN_g5_cuda test_integrator_cuda

all: $(OBJ) $(PROG) kernels


kernels:
	rm -f CUDA && ln -s $(SAPPOROPATH)/CUDA CUDA

#$(PROG): $(OBJ)
#	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -lsapporo

test_gravity_block_cuda : test_gravity_block.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS)

test_gravity_block_g5_cuda: test_gravity_block_g5.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS)

test_gravity_block_6th_cuda : test_gravity_block_6th.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2)  $(LDFLAGS)   

test_performance_rangeN_cuda : test_performance_rangeN.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS)  

test_performance_blockStep_cuda : test_performance_blockStep.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS) 

test_performance_rangeN_6th_cuda : test_performance_rangeN_6th.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS) 

test_performance_rangeN_g5_cuda : test_performance_rangeN_g5.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIB2) $(LDFLAGS) 

test_integrator_cuda : test_integrator.o
	$(CXX) $(LDFLAGS) $^ -o $@ -L $(SAPPOROPATH) -l$(SAPLIBG6) $(LDFLAGS) 


%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o *.ptx *.a $(PROG) CUDA


$(OBJ): $(SAPPOROPATH)/$(SAPLIB)








