CXX ?= g++
CC ?= gcc
PREFIX ?= $(PWD)

ifdef CUDA_HOME
CUDA_TK ?= $(CUDA_HOME)
endif 
ifdef CUDA_PATH
CUDA_TK ?= $(CUDA_PATH)
endif

.PHONY: all
all: libsapporo2.a libsapporo2.so emulated_interfaces


# Detect CUDA
ifndef CUDA_TK
    NVCC := $(shell which nvcc || echo NOTFOUND)
    ifeq ($(NVCC), NOTFOUND)
        $(info The nvcc command is not available in your shell.)
        $(info To compile with CUDA, please install it, set up your environment)
        $(info according to the CUDA installation instructions, and try again.)
        $(info )
    else
        CUDA_TK := $(dir $(NVCC))..
        CUDA_AVAILABLE := 1
    endif
else
    NVCC ?= $(CUDA_TK)/bin/nvcc
    CUDA_AVAILABLE := 1
endif


# Detect OpenCL
OPENCL_LDFLAGS := -lOpenCL
ifdef OPENCL
    OPENCL_LDFLAGS := -L$(OPENCL)/lib -lOpenCL
endif

OPENCL_STATUS := $(shell echo 'int main() {}' | $(CXX) -x c++ $(OPENCL_LDFLAGS) - && rm a.out || echo NOTFOUND)

ifeq ($(OPENCL_STATUS), NOTFOUND)
    $(info OpenCL support was not detected on the system.)
    $(info If it is installed in a non-standard location, then set OPENCL to)
    $(info the installation prefix and try again.)
    $(info )
else
    OPENCL_AVAILABLE := 1
endif


# Select backend
ifeq ($(filter clean,$(MAKECMDGOALS)),)
ifndef BACKEND
    ifdef CUDA_AVAILABLE
        $(info BACKEND not set and CUDA was detected, using CUDA)
        BACKEND := CUDA
    else
        ifdef OPENCL_AVAILABLE
            $(info BACKEND not set and OpenCL was detected, using OpenCL)
            BACKEND := OpenCL
        else
            $(error BACKEND not set and neither CUDA nor OpenCL was detected.)
        endif
    endif
else
    ifeq ($(BACKEND), CUDA)
        ifndef CUDA_AVAILABLE
            $(error BACKEND set to CUDA but it was not found.)
        endif
        $(info Using selected backend CUDA)
    else
        ifeq ($(BACKEND), OpenCL)
            ifndef OPENCL_AVAILABLE
                $(error BACKEND set to OpenCL but it was not found.)
            endif
        else
            $(error BACKEND set to unknown value "$(BACKEND)", please use CUDA or OpenCL)
        endif
        $(info Using selected backend OpenCL)
    endif
endif
endif
$(info )

# Testing/optimisation support
ifdef NTHREADS
    CXXFLAGS += -DNTHREADS=$(NTHREADS) -DTIMING_STATS=1
endif

ifdef NBLOCKS_PER_MULTI
    CXXFLAGS += -DNBLOCKS_PER_MULTI=$(NBLOCKS_PER_MULTI) -DTIMING_STATS=1
endif


# CUDA kernels
ifeq ($(BACKEND), CUDA)

INCLUDES = -I$(CUDA_TK)/include
CXXFLAGS += -D__INCLUDE_KERNELS__
LDFLAGS += -lcuda -fopenmp

CUDA_SRC = $(wildcard src/CUDA/*.cu)
PTX = $(CUDA_SRC:src/CUDA/%.cu=src/CUDA/%.ptx)
PTXH = $(CUDA_SRC:src/CUDA/%.cu=src/CUDA/%.ptxh)
NVCCFLAGS += -Isrc

KERNELS = $(PTX) $(PTXH)

%.ptx: %.cu
	$(NVCC) --forward-unknown-to-host-compiler $(CXXFLAGS) $(NVCCFLAGS) -ptx $< -o $@

src/CUDA/%.ptxh: src/CUDA/%.ptx
	xxd -i $< | sed 's/src_CUDA_/CUDAKernels_/g' > $@

endif


# OpenCL kernels
ifeq ($(BACKEND), OpenCL)

ifdef OPENCL
    CXXFLAGS += -I$(OPENCL)/include
    LDFLAGS += -L$(OPENCL)/lib
endif

INCLUDES =
CXXFLAGS += -D_OCL_ -D__INCLUDE_KERNELS__
LDFLAGS += -lOpenCL -fopenmp

OPENCL_SRC = $(wildcard src/OpenCL/*.cl)
CLE = $(OPENCL_SRC:src/OpenCL/%.cl=src/OpenCL/%.cle)
CLH = $(OPENCL_SRC:src/OpenCL/%.cl=src/OpenCL/%.clh)

KERNELS = $(CLE) $(CLH)

%.cle: %.cl
	$(CC) -E -Isrc -o $@ - <$<

# xxd names the variable after the file name argument, and we expect
# the variable to not have a src_ prefix, so we have to remove it.
src/OpenCL/%.clh: src/OpenCL/%.cle
	cd src && xxd -i $(<:src/%=%) $(@:src/%=%)

endif


# Main implementation
CXX_SRC := $(wildcard src/*.cpp src/SSE_AVX/*.cpp)
OBJS := $(CXX_SRC:%.cpp=%.o)
INCLUDES += -Isrc
CXXFLAGS += $(INCLUDES) -fPIC -g -O3 -Wall -Wextra -Wstrict-aliasing=2 -fopenmp

# The objects are backend-specific: the CUDA backend compiles cudadev.h while
# the OpenCL backend compiles ocldev.h (selected by -D_OCL_). Make only tracks
# file timestamps, not the value of BACKEND, so after building one backend a
# subsequent build with the other backend would silently reuse the existing
# objects and library. That produced the reported failure: an OpenCL test
# linked against a CUDA-compiled libsapporo2 tried to cuModuleLoad() a .cl
# source file, aborting with CUDA_ERROR_INVALID_IMAGE in cudadev.h.
#
# Record the active backend in a stamp file and depend on it so that switching
# BACKEND forces the objects to be recompiled. The stamp is only rewritten when
# the backend actually changes, so unchanged rebuilds stay incremental.
BACKEND_STAMP := .backend.stamp

.PHONY: FORCE
$(BACKEND_STAMP): FORCE
	@[ "$$(cat $@ 2>/dev/null)" = "$(BACKEND)" ] || \
	    { echo "Backend changed to $(BACKEND), rebuilding objects"; echo "$(BACKEND)" > $@; }

$(OBJS): $(BACKEND_STAMP)

src/sapporohostclass.o: $(KERNELS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

libsapporo2.a: $(OBJS)
	ar qv $@ $^

libsapporo2.so: $(OBJS)
	$(CXX) -o $@ -shared $^ $(LDFLAGS)


# API compatibility libraries
EMU_SRC := $(wildcard src/interfaces/*lib.cpp)
EMU_STATIC_LIBS := $(EMU_SRC:src/interfaces/%lib.cpp=lib%.a)
EMU_SHARED_LIBS := $(EMU_SRC:src/interfaces/%lib.cpp=lib%.so)

.PHONY: emulated_interfaces
emulated_interfaces: $(EMU_STATIC_LIBS) $(EMU_SHARED_LIBS)

$(EMU_STATIC_LIBS): libsapporo2.a

$(EMU_SHARED_LIBS): libsapporo2.so


lib%.a: src/interfaces/%lib.o
	ar qv $@ $^

lib%.so: src/interfaces/%lib.o
	$(CXX) -o $@ -shared $^ -L. -lsapporo2 $(LDFLAGS)


# Installation
INSTALLED_LIBS := $(PREFIX)/lib/libsapporo2.a $(PREFIX)/lib/libsapporo2.so
INSTALLED_LIBS += $(EMU_STATIC_LIBS:%.a=$(PREFIX)/lib/%.a)
INSTALLED_LIBS += $(EMU_SHARED_LIBS:%.so=$(PREFIX)/lib/%.so)

INSTALLED_LIBS: $(PREFIX)/lib

HEADERS := $(wildcard include/*)
INSTALLED_HEADERS := $(HEADERS:include/%=$(PREFIX)/include/%)

INSTALLED_HEADERS: $(PREFIX)/include

$(PREFIX)/include:
	mkdir -p $(PREFIX)/include

$(PREFIX)/include/%: include/% $(PREFIX)/include
	install -m 644 $< $@

$(PREFIX)/lib:
	mkdir -p $(PREFIX)/lib

$(PREFIX)/lib/%: % $(PREFIX)/lib
	install -m 644 $< $@

.PHONY: install
install: $(INSTALLED_LIBS) $(INSTALLED_HEADERS)

.PHONY: uninstall
uninstall:
	rm -rf $(INSTALLED_LIBS) $(INSTALLED_HEADERS)


# Tests
# Build the test programs against the freshly built libraries and run the
# GPU-vs-CPU correctness tests for each supported integration order. The
# backend selected above determines which test Makefile and binaries are used.
ifeq ($(BACKEND), CUDA)
    TEST_MAKEFILE := Makefile
    TEST_SUFFIX := cuda
else
    TEST_MAKEFILE := Makefile_ocl
    TEST_SUFFIX := ocl
endif

CORRECTNESS_TESTS := test_gravity_block_$(TEST_SUFFIX) \
                     test_gravity_block_g5_$(TEST_SUFFIX) \
                     test_gravity_block_6th_$(TEST_SUFFIX)

.PHONY: build-tests
build-tests: all
	$(MAKE) -C tests -f $(TEST_MAKEFILE) CXX="$(CXX)" CC="$(CC)" \
	    $(if $(CUDA_TK),CUDA_TK="$(CUDA_TK)")

.PHONY: test
test: build-tests
	@for t in $(CORRECTNESS_TESTS); do \
	    echo "=== Running $$t ==="; \
	    ( cd tests && LD_LIBRARY_PATH="$(CURDIR):$$LD_LIBRARY_PATH" ./$$t ) || exit 1; \
	done


# Clean-up
.PHONY: clean
clean:
	rm -f *.a *.so src/*.o src/SSE_AVX/SSE/*.o src/SSE_AVX/AVX/*.o
	rm -f src/CUDA/*.ptx src/CUDA/*.ptxh src/OpenCL/*.cle src/OpenCL/*.clh
	rm -f $(BACKEND_STAMP)
	$(MAKE) -C tests -f Makefile clean
	$(MAKE) -C tests -f Makefile_ocl clean

