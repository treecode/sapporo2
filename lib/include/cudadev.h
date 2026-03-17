#ifndef __CUDADEV_H__
#define __CUDADEV_H__

#define __CUDA_DEVICE__

#include <vector>
#include <cassert>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <memory.h>
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>
#include <builtin_types.h>

#ifdef __INCLUDE_KERNELS__
extern unsigned char CUDAKernels_kernels_ptx[];
extern unsigned int CUDAKernels_kernels_ptx_len;
#endif

namespace dev {

#  define CUT_CHECK_ERROR(errorMessage) {				\
  cudaError_t err = cudaGetLastError();					\
  if( cudaSuccess != err) {						\
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",	\
	    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) ); \
    exit(EXIT_FAILURE);							\
  }									\
}

  inline const char* cuPrintError(const int err) {
    switch (err) {
    case CUDA_SUCCESS : return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE : return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY : return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED : return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED : return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_NO_DEVICE : return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE : return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE : return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT : return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT : return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED : return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED : return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED : return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED : return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU : return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED : return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED : return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_INVALID_SOURCE : return "CUDA_ERROR_INVALID SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND : return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_INVALID_HANDLE : return "CASE_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_NOT_FOUND : return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY : return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_LAUNCH_FAILED : return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : return "CUDA_ERROR_LAUNCH_OUT_OF_RESOUCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT : return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING : return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_UNKNOWN : return "CUDA_ERROR_UNKNOWN";
    default : return "Unknown CUresult.";
    }
  }

// fprintf(stderr, "CUDA_ERROR_DEINITIALIZED error, continue for testing! \n"); 
#define CU_SAFE_CALL_NO_SYNC( call ) {					\
    CUresult err = call;						\
    if(err == CUDA_ERROR_DEINITIALIZED){ \
    } \
    else if( CUDA_SUCCESS != err) {						\
      fprintf(stderr, "Cuda driver error <%s> in file '%s' in line %i.\n", \
	      cuPrintError(err), __FILE__, __LINE__ );			\
      assert(false);\
    } }

#define cuSafeCall( call )       CU_SAFE_CALL_NO_SYNC(call);

  class context {
  protected:
    size_t devId;
    CUcontext Context;
    CUdevice  Device;

    int DeviceCount;
    bool ContextFlag;
    bool InitFlag;

    //Compute capabilty, important for default compilation mode
    int ccMajor;
    int ccMinor;
    int defaultComputeMode;

    int multiProcessorCount;
    int workGroupSizeMultiple;
    std::string                deviceName;    

  public:
    context() {
      ContextFlag = false;

      cuSafeCall(cuInit(0));     // Initialise driver API;
      InitFlag    = true;
    }
    ~context() {
      cerr << "Delete context! \n";
      if (ContextFlag) cuCtxDetach(Context);
    }

    void setDefaultComputeMode()
    {
      #if CUDA_VERSION >= 6000
        //From 6.0 onwards we can just compute the required mode
        defaultComputeMode = ccMajor*10 + ccMinor;      
      #else
        switch(ccMajor)
        {
          case 1:
            switch(ccMinor)
            {
              case 0:
                defaultComputeMode = CU_TARGET_COMPUTE_10;
                break;
              case 1:
                defaultComputeMode = CU_TARGET_COMPUTE_11;
                break;
              case 2:
                defaultComputeMode = CU_TARGET_COMPUTE_12;
                break;
              case 3:
                defaultComputeMode = CU_TARGET_COMPUTE_13;
                break;
            }
            break;
        case 2:
            switch(ccMinor)
            {
              case 0:
                defaultComputeMode = CU_TARGET_COMPUTE_20;
                break;
              case 1:
                  defaultComputeMode = CU_TARGET_COMPUTE_21;
                break;
            }
            break;
            
        #if CUDA_VERSION >= 5000
          case 3:        
            switch(ccMinor)
              {
                case 0:
                  defaultComputeMode = CU_TARGET_COMPUTE_30;
                  break;
                case 5:
                  defaultComputeMode = CU_TARGET_COMPUTE_35;
                break;
              }        
        #endif
      } //switch
     #endif
     
     fprintf(stderr, "Compute mode: %d.%d Target: %d \tCUDA version: %d \n", ccMajor, ccMinor, defaultComputeMode, CUDA_VERSION);
   }

    int getDeviceCount() {
      assert(InitFlag);
      std::cerr << "Getting list of CUDA devices ...\n";

      DeviceCount = 0;
      cuSafeCall(cuDeviceGetCount(&DeviceCount));

      std::cerr << "Found " << DeviceCount << " suitable devices: \n";
      for (int dev = 0; dev < DeviceCount; dev++) {
        char device_string[1024];
        cuSafeCall(cuDeviceGetName(device_string, 1024, dev));
	std::cerr << dev << ": " << device_string << "\n";
      }

      return  DeviceCount;
    }

    void createQueue(const int dev = 0, const int ctxCreateFlags = 0) {
      //use CU_CTX_MAP_HOST as flag for zero-copy memory

      assert(!ContextFlag);
      assert(InitFlag);
      devId = dev;
      assert((int)devId < DeviceCount);
      if (dev >= 0) {
	fprintf(stderr, "Using device %d\n", (int)dev);
	//Get the device handle for dev
	cuSafeCall(cuDeviceGet(&Device, devId));

	//   result status = cuCtxCreate(&g_oContext, CU_CTX_LMEM_RESIZE_TO_MAX, g_oDevice);
	//  Faster and async kernel launches when using large size arrays of local memory
	// ctxCreateFlags |= CU_CTX_LMEM_RESIZE_TO_MAX;

	//Create the context for this device handle
	cuSafeCall(cuCtxCreate(&Context, ctxCreateFlags, Device));
      } else {
	int dev = 0;
	while(1) {
	  fprintf(stderr, "Trying device %d \n", (int)dev);
	  cuSafeCall(cuDeviceGet(&Device, dev));
	  if(cuCtxCreate(&Context, ctxCreateFlags, Device) != CUDA_SUCCESS) {
	    dev = (dev + 1)  % DeviceCount;
	  } else {
	    devId = dev;
	    break;
	  }
	}
      }
      
      //Not import on CUDA, we use Compute Capability
      deviceName.assign("");

      //Retrieve CC of the selected device
      cuDeviceComputeCapability(&ccMajor, &ccMinor, Device);
      fprintf(stderr, "Compute mode: %d.%d \n", ccMajor, ccMinor);
      setDefaultComputeMode();

      //Get the number of multiprocessors of the device
      cuSafeCall(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, Device ));

      //Get work size
      cuSafeCall(cuDeviceGetAttribute(&workGroupSizeMultiple,CU_DEVICE_ATTRIBUTE_WARP_SIZE, Device ));

      ContextFlag = true;
    }

    int getComputeCapabilityMajor() const {return ccMajor;} 
    int getComputeCapabilityMinor() const {return ccMajor;}
    std::string getDeviceName() const {return deviceName;}
    
    const CUcontext& get_context() const {return Context;}
    int        get_command_queue() const {return -1;}
    const CUdevice&  get_device()        const {return Device;}
//     const CUdevice&  get_device(const int dev) const {return Device;}

    const int&  getDefaultComputeMode() const {return defaultComputeMode;}

    int        get_numberOfMultiProcessors() const { return multiProcessorCount;}
    int        get_workGroupSizeMultiple()   const { return workGroupSizeMultiple;}

  };

  template<class T>
  class memory {
  protected:
    CUcontext Context;
    bool ContextFlag;
    bool pinned;

    size_t n;
    CUdeviceptr DeviceMem;
    int         DeviceMemFlags;
    std::vector<T> HostMem;
    
    T           *HostMemPinned;
    
    bool *memSet;       //Trick to prevent double crashes if memory is shared

    void cuda_free() {
      if (n > 0 && memSet[0]) {
	assert(ContextFlag);

	cuSafeCall(cuMemFree(DeviceMem));
	HostMem.clear();
        HostMem.resize(0);
        if(pinned)
        {
          cuSafeCall(cuMemFreeHost((void*)HostMemPinned));
          HostMemPinned = NULL;
        }
	n         = 0;
        memSet[0] = false;
      }
    }


    void setContext(const CUcontext &context, const int &null) {
      assert(!ContextFlag);
      Context      = context;
      ContextFlag  = true;
            
      if(null) ContextFlag = true; //Make compiler happy
      
    }

  public:
    memory() : ContextFlag(false), n(0) {}
    memory(class context &c) : ContextFlag(false), n(0) { setContext(c); }
    memory(class context &c, const int _n, const int flags = 0) : ContextFlag(false), n(0) {
      setContext(c);
      allocate(_n, flags);
    }
    memory(class memory &x) : ContextFlag(false), n(0) {   setContext(x.get_context(), x.get_command_queue());  }
    memory(class memory &x, const int _n, const int flags = 0) : ContextFlag(false), n(0) {
      setContext(x.get_context(), x.get_command_queue());
      allocate(_n, flags);
    }
    ~memory() {
      cuda_free();
    }

    void setContext(const context &c) { setContext(c.get_context(), c.get_command_queue());  }

    const std::vector<T> to_vector() const {return HostMem;}

    void allocate(const int _n, const int flags = 0, bool pinned = false) {
      assert(ContextFlag);
      
      memSet = new bool[1];
      memSet[0] = true;

      if (n > 0) cuda_free();
      n = _n;
      DeviceMemFlags = flags;
      if(!pinned)
      {
        HostMem.resize(n);
      }
      else
      { 
        cuSafeCall(cuMemAllocHost((void**)&HostMemPinned, n*sizeof(T)));
      }
      cuSafeCall(cuMemAlloc(&DeviceMem, n*sizeof(T)));
      
      this->pinned = pinned;
    }

    void realloc(const unsigned int _n, const int flags = 0, bool copyBack = true)
    {
      //Reallocate the array
      assert(ContextFlag);
      
      DeviceMemFlags = flags; //make compiler happy
      
      if(_n != n  && _n > 0)
      {
        //We want more memory, increase size on host, copy
        //data to host, free and allocate mem on device
        //And finally copy the data back to the device
        if(copyBack)
          d2h();

        if(pinned)
        {
          //No realloc function so do it by hand
          T *tmp_ptr;            
          cuSafeCall(cuMemAllocHost((void**)&tmp_ptr, _n*sizeof(T)));        
          //Copy old content to newly allocated mem
          int tmpSize = std::min((int)n, (int)_n);
                
          //Copy the old data to the new pointer and free the old location
          memcpy (((void*) tmp_ptr), ((void*) HostMemPinned), tmpSize*sizeof(T)); 
          cuSafeCall(cuMemFreeHost((void*)HostMemPinned));
          HostMemPinned = tmp_ptr;     
        }
        else
        {
          HostMem.resize(_n);
        }
        
        cuSafeCall(cuMemFree(DeviceMem));
        n = _n;
        cuSafeCall(cuMemAlloc(&DeviceMem, n*sizeof(T)));

        if(copyBack)
          h2d();
      }
    }

    void zeroMem() {
      assert(ContextFlag);
      assert(n > 0);
      if(pinned)
          memset(&HostMemPinned[0], 0, n*sizeof(T));      
      else
          memset(&HostMem[0], 0, n*sizeof(T));
      cuSafeCall(cuMemsetD8(DeviceMem, 0, n*sizeof(T)));
    }

    void set(const std::vector<T> &in) {
      const int n = in.size();
      allocate(n, DeviceMemFlags);
      for (int i = 0; i < n; i++)
      {
        if(pinned)
          HostMemPinned[i] = in[i];          	
        else
          HostMem[i] = in[i];          
      }
    }

    void device2host() {d2h();}
    void host2device() {h2d();}

    void d2h(const int number, int offset, const bool OCL_BLOCKING = true, const CUstream stream = 0)   {
      assert(ContextFlag);
      assert(n > 0);
      assert(number > 0);

      offset = offset*sizeof(T);        //Convert the number into actual bytes

      if (OCL_BLOCKING)
      {
        if(pinned){
          cuSafeCall(cuMemcpyDtoH(&HostMemPinned[0], DeviceMem+offset, number*sizeof(T)));          
        }else{
          cuSafeCall(cuMemcpyDtoH(&HostMem[0], DeviceMem+offset, number*sizeof(T)));
        }
      } else{
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned);
        cuSafeCall(cuMemcpyDtoHAsync(&HostMemPinned[0], DeviceMem, number*sizeof(T), stream));
      }
    }


    void d2h(const int number, const bool OCL_BLOCKING = true, const CUstream stream = 0)   {
      assert(ContextFlag);
      assert(n > 0);
      assert(number > 0);

      if (OCL_BLOCKING) {
        if(pinned){
          cuSafeCall(cuMemcpyDtoH(&HostMemPinned[0], DeviceMem, number*sizeof(T)));          
        }else{
          cuSafeCall(cuMemcpyDtoH(&HostMem[0], DeviceMem, number*sizeof(T)));
        }
      } else{
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned);
        cuSafeCall(cuMemcpyDtoHAsync(&HostMemPinned[0], DeviceMem, number*sizeof(T), stream));
      }
    }

    void h2d(const int number, const bool OCL_BLOCKING  = true, const CUstream stream = 0)   {
      assert(ContextFlag);
      assert(n > 0);
      assert(number > 0);

      if(OCL_BLOCKING) {
        if(pinned){
          cuSafeCall(cuMemcpyHtoD(DeviceMem, &HostMemPinned[0], number*sizeof(T)));
        }else{
          cuSafeCall(cuMemcpyHtoD(DeviceMem, &HostMem[0], number*sizeof(T)));
        }          
      } else {
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned);
        cuSafeCall(cuMemcpyHtoDAsync(DeviceMem, &HostMemPinned[0], number*sizeof(T), stream));
      }
    }


    void d2h(const bool OCL_BLOCKING = true, const CUstream stream = 0)   {
      assert(ContextFlag);
      assert(n > 0);

      if (OCL_BLOCKING) {
        if(pinned){
          cuSafeCall(cuMemcpyDtoH(&HostMemPinned[0], DeviceMem, n*sizeof(T)));          
        }else{
          cuSafeCall(cuMemcpyDtoH(&HostMem[0], DeviceMem, n*sizeof(T)));
        }
      } else{
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned);
        cuSafeCall(cuMemcpyDtoHAsync(&HostMemPinned[0], DeviceMem, n*sizeof(T), stream));
      }
    }

    void h2d(const bool OCL_BLOCKING  = true, const CUstream stream = 0)   {
      assert(ContextFlag);
      assert(n > 0);
      
      if(OCL_BLOCKING) {
        if(pinned){
          cuSafeCall(cuMemcpyHtoD(DeviceMem, &HostMemPinned[0], n*sizeof(T)));
        }else{
          cuSafeCall(cuMemcpyHtoD(DeviceMem, &HostMem[0], n*sizeof(T)));
        }          
      } else {
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned);
        cuSafeCall(cuMemcpyHtoDAsync(DeviceMem, &HostMemPinned[0], n*sizeof(T), stream));
      }
    }


    void copy(const memory &src, const int n, const bool OCL_BLOCKING = true) {
      assert(ContextFlag);
      if (n < src.n) {
	cuda_free();
// 	HostMem.resize(src.n);
	allocate(src.n, DeviceMemFlags, this->pinned);
      }

      //Copy on the device
      cuSafeCall(cuMemcpyDtoD(DeviceMem, src.DeviceMem, n*sizeof(T)));
      
      //Copy on the host
      if(pinned)
      {
        memcpy (((void*) &HostMemPinned[0]), ((void*) &src[0]), n*sizeof(T));        
      }
      else
      {
        memcpy (((void*) &HostMem[0]), ((void*) &src[0]), n*sizeof(T));
      }
    }

    const T& operator[] (const int i) const
    {
      if(pinned)
        return HostMemPinned[i];       
      else
        return HostMem[i];       
    }
    T& operator[](const int i)
    {
      if(pinned)
        return HostMemPinned[i];       
      else
        return HostMem[i];       
    }    

    const CUdeviceptr& get_device_mem() {return DeviceMem;}
    void*   p() {return (void*)&DeviceMem;}
    void* ptr() {return p();}
 
    size_t size(){return n;}

    const CUcontext& get_context() const {return Context;}
    int        get_command_queue() const {return -1;}
  };

  class kernel {
  protected:
    char *KernelFilename;
    char *KernelName;

    CUcontext   Context;
    CUmodule    cuModule;
    CUfunction  Kernel;

    std::vector<size_t> GlobalWork, LocalWork;

    std::vector<void*> argumentList;
    std::vector<int> argumentOffset;

    bool ContextFlag;
    bool KernelFlag;
    bool ProgramFlag;
    bool WorkFlag;

    int sharedMemorySize;
    int paramOffset;

    int WorkGroupSizeMultiple;
    int WorkGroupMaxSize;
    
    int computeMode;
    
    int CommandQueue; //Note this variable is not used only to prevent compiler warnings
    
    void clean() {
      KernelName     = (char*)malloc(256);
      KernelFilename = (char*)malloc(1024);
      GlobalWork.clear();
      LocalWork.clear();

      ContextFlag = false;
      KernelFlag  = false;
      ProgramFlag = false;
      WorkFlag    = false;

      sharedMemorySize = 0;
      paramOffset      = 0;

      WorkGroupSizeMultiple = 0;
      WorkGroupMaxSize = 0;
    }


    //Function to get wall-clock timings
    double get_time_test() {
      struct timeval Tvalue;
      struct timezone dummy;

      gettimeofday(&Tvalue,&dummy);
      return ((double) Tvalue.tv_sec +
              1.e-6*((double) Tvalue.tv_usec));
    }


    void setContext(const CUcontext &context, const int  &command_queue) {
      assert(!ContextFlag);
      Context      = context;
      ContextFlag  = true;
      WorkGroupSizeMultiple = 32;
      WorkGroupMaxSize = 256;
      //In CUDA command_queue is not really used, this is to prevent unused warnings
      CommandQueue = command_queue;
    }


    void load_source(const char *fileName, std::string &ptx_source){
      FILE *fp;

      fp = fopen(fileName, "rb");

      if(fp == NULL) {
        fprintf(stderr, "Cannot open source file: %s \n", fileName);
#ifdef __INCLUDE_KERNELS__

        fprintf(stderr, "Checking for compiled in version of file: %s\n", fileName);

        string temp = string(fileName);
        
        if(temp.rfind("kernels.ptx") != string::npos)
        {
            ptx_source.append((const char*)CUDAKernels_kernels_ptx,CUDAKernels_kernels_ptx_len);
            ptx_source.append(512, '\0');
            fprintf(stderr, "Found compiled in version of file: %s\n", fileName);
            return;
        }
        else
        {
            fprintf(stderr, "Cannot find compiled in replacement for: %s \n", fileName);
            assert(false);
        }
#else
        assert(false);
#endif
      }

      fseek(fp, 0, SEEK_END);
      const int file_size = ftell(fp);
      ptx_source.reserve(file_size + 512);
      fseek(fp, 0, SEEK_SET);
      const size_t read = fread(&ptx_source[0], sizeof(char), file_size, fp);

      //Set the last char to NULL else old values in the extra memory
      //buffer will crash the compiler....
      ptx_source[file_size] = '\0';

      if(read == 0) {
        fprintf(stderr, "Cannot read source file: %s \n", fileName);
        assert(false);
      }

      fclose(fp);
    }

  public:
    kernel() {clean();}
    ~kernel() {
      free(KernelName);
      free(KernelFilename);
      if (ProgramFlag) cuModuleUnload(cuModule);
    }
    kernel(class context &c) {clean(); setContext(c);}
    kernel(class kernel &k) {clean(); setContext(k.get_context(), k.get_command_queue());}

    void setContext(const context &c) {
      setContext(c.get_context(), c.get_command_queue());
      computeMode = c.getDefaultComputeMode();
      WorkGroupSizeMultiple = c.get_workGroupSizeMultiple();
    }

    void load_source(const char *kernel_name, const char *subfolder,
                     const char *compilerOptions = "") {

      assert(ContextFlag);
      assert(!ProgramFlag);

      //In cuda version we assume that the code is already compiled into ptx
      //so that the file loaded/specified is in fact a PTX file
      sprintf(KernelFilename, "%s%s", subfolder, kernel_name);
      string temp = string(KernelFilename);
      
      if(temp.rfind("ptx") != string::npos)
      {
        const unsigned int maxJitOptions = 6;
        CUjit_option *jitOptions = new CUjit_option[maxJitOptions];
        void **jitOptVals        = new void*[maxJitOptions];


        int jitOptionCount = 0;
        //use JIT compiling to set the max register number
        //         if(maxrregcount > 0) {
        //           //Set the maximum number of registers option
        //           jitOptions[jitOptionCount] = CU_JIT_MAX_REGISTERS;
        //           int jitRegCount = maxrregcount;
        //           jitOptVals[jitOptionCount] = (void *)jitRegCount;
        //           jitOptionCount++;
        //         }


#if CUDA_VERSION < 6000
        if(computeMode < CU_TARGET_COMPUTE_20)
        {
          fprintf(stderr,"Sapporo2 requires at least a Fermi or newer NVIDIA architecture.\n");
          exit(-1);
        }
#endif

        //Set the architecture
        //         {
        //           jitOptions[jitOptionCount] = CU_JIT_TARGET;
        //           int arch = computeMode;
        //           jitOptVals[jitOptionCount] = (void *)arch;
        //           jitOptionCount++;
        // 
        //           std::cout << "Using compute mode: " << computeMode << "\tSource file: " << KernelFilename << std::endl;
        //         }

        std::string ptxSource;
        load_source(KernelFilename, ptxSource);

        cuSafeCall(cuModuleLoadDataEx(&cuModule, ptxSource.c_str(), jitOptionCount, jitOptions, (void **)jitOptVals));
        
        
        //This is to make compiler happy about our unused variable compiler_options
        temp.assign(compilerOptions);
        

        delete[] jitOptVals;
        delete[] jitOptions;
      }
      else
      {
        //Load CUBIN source
        string ptxSource;
        load_source(KernelFilename, ptxSource);
        cuSafeCall(cuModuleLoad(&cuModule, KernelFilename));
      }

      ProgramFlag = true;
    }

    void create(const char *kernel_name) {
      assert(ProgramFlag);
      assert(!KernelFlag);
      sprintf(KernelName, kernel_name,"");

      fprintf(stderr, "Creating kernel: %s \n", kernel_name);
      cuSafeCall(cuModuleGetFunction(&Kernel, cuModule, KernelName));
      
      
      cuSafeCall(cuFuncGetAttribute(&WorkGroupMaxSize, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Kernel));
      fprintf(stderr,"Maximum work group size: %d Optimal work group multiple: %d \n",(int) WorkGroupMaxSize, (int) WorkGroupSizeMultiple);
      KernelFlag = true;
    }


    //NVIDIA macro
    #define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) -1)
    //'size'  is used for dynamic shared memory
    //Cuda does not have a function like clSetKernelArg
    //therefore we keep track of a vector with arguments
    //that will be processed when we launch the kernel
    template<class T>
    void set_arg(const unsigned int arg, void* ptr, const int size = 1)  {
      assert(KernelFlag);

      //Resize the argumentList if it is too small
      if(arg >= argumentList.size()) {
        argumentList.resize(arg+1);
        argumentOffset.resize(arg+1);
        argumentOffset[arg] = -1;
      }

      //First check if the call is to allocate shared
      //memory: ptr==NULL and size > 1
      if(ptr == NULL && size > 1)  {
        //Sometimes we reuse the same kernel which sets the shared
        //memory again, therefore first reduce the sum with the
        //previous value and then add the new value
        if(argumentOffset[arg] != -1)
          sharedMemorySize  -= argumentOffset[arg];

        argumentOffset[arg] = size*sizeof(T);
        sharedMemorySize  +=  size*sizeof(T);

        return;
      }

      if(argumentOffset[arg] >= 0) {
        //Already set this parameter once before
        //so no need to redo all calculations
        int tempOffset = argumentOffset[arg];
        ALIGN_UP(tempOffset, __alignof(T));
        cuSafeCall(cuParamSetv(Kernel, tempOffset, ptr, sizeof(T)));
      } else {
        argumentOffset[arg] = paramOffset;
        ALIGN_UP(paramOffset, __alignof(T));
        cuSafeCall(cuParamSetv(Kernel, paramOffset, ptr, sizeof(T)));
        paramOffset += sizeof(T);
      }

    }


    void setWork(const std::vector<size_t> &global_work, const std::vector<size_t> &local_work) {
      assert(KernelFlag);
      assert(global_work.size() == local_work.size());

      GlobalWork.resize(3);
      LocalWork.resize(3);

      LocalWork [0] =  local_work[0];
      GlobalWork[0] = global_work[0];

      LocalWork [1] = ( local_work.size() > 1) ?  local_work[1] : 1;
      GlobalWork[1] = (global_work.size() > 1) ? global_work[1] : 1;

      LocalWork [2] = ( local_work.size() > 2) ?  local_work[2] : 1;
      GlobalWork[2] = (global_work.size() > 2) ? global_work[2] : 1;

      //Since the values between CUDA and OpenCL differ:
      //Cuda is specific size of each block, while OpenCL
      //is the combined size of the lower blocks and this block
      //we have to divide the values

      GlobalWork[0] /= LocalWork[0];
      GlobalWork[1] /= LocalWork[1];
      GlobalWork[2] /= LocalWork[2];

      WorkFlag = true;
    }

    void setWork(const int nx_threads, const int nx_items,
		 const int ny_threads, const int ny_items) {
      std::vector<size_t> localWork(2), globalWork(2);
      const int ngx = (nx_items - 1) / nx_threads + 1;
      const int ngy = (ny_items - 1) / ny_threads + 1;
      globalWork[0] = ngx*nx_threads;  globalWork[1] = ngy*ny_threads;
      localWork [0] = nx_threads;      localWork [1] = ny_threads;
      setWork(globalWork, localWork);
    }

    void setWork_block1D(const int n_threads, const int blocks) {
      std::vector<size_t> localWork(2), globalWork(2);
      const int nx = blocks;
      const int ny = 1;
      globalWork[0] = nx*n_threads;  globalWork[1] = ny;
      localWork [0] = n_threads;      localWork[1] = 1;
      setWork(globalWork, localWork);
    }

    void setWork_block2D(const int n_threads, const int blocks) {
      std::vector<size_t> localWork(2), globalWork(2);
      const int nx = (int)std::sqrt(blocks);
      const int ny = (blocks -1)/nx +  1;
      globalWork[0] = nx*n_threads;  globalWork[1] = ny;
      localWork [0] = n_threads;      localWork[1] = 1;
      setWork(globalWork, localWork);
    }

    void setWork_threadblock2D(const int nx_threads, const int ny_threads,
                               const int nx_blocks,  const int ny_blocks) {
      std::vector<size_t> localWork(2), globalWork(2);

      globalWork[0] = nx_blocks*nx_threads;  globalWork[1] = ny_blocks;
      localWork [0] = nx_threads;      localWork[1] = ny_threads;

      GlobalWork.resize(3);
      LocalWork.resize(3);

      GlobalWork[0] = nx_blocks; GlobalWork[1] = ny_blocks; GlobalWork[2] = 1;
      LocalWork[0] = nx_threads; LocalWork[1] = ny_threads; LocalWork[2] = 1;

      WorkFlag = true;

    }



    void setWork_1D(const int n_threads, const int items){
      std::vector<size_t> localWork(2), globalWork(2);
      const int ng = (items - 1) / n_threads + 1;
      const int nx = ng;
      const int ny = 1;
      globalWork[0] = nx*n_threads;  globalWork[1] = ny;
      localWork [0] = n_threads;      localWork[1] = 1;
      setWork(globalWork, localWork);
    }

    void setWork_2D(const int n_threads, const int items) {
      std::vector<size_t> localWork(2), globalWork(2);
      const int ng = (items - 1) / n_threads + 1;
      const int nx = (int)std::sqrt(ng);
      const int ny = (ng - 1)/nx +  1;
      globalWork[0] = nx*n_threads;  globalWork[1] = ny;
      localWork [0] = n_threads;      localWork[1] = 1;
      setWork(globalWork, localWork);
    }

    void printWorkSize()
    {
      printf("Blocks: (%ld, %ld, %ld) Threads: (%ld, %ld, %ld) \n",
              GlobalWork[0], GlobalWork[1], GlobalWork[2],
              LocalWork[0],  LocalWork[1],  LocalWork[2]);
    }

    void execute(int* event = NULL) {
      assert(KernelFlag);
      assert(WorkFlag);

      cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 1024*1024*50);


      cuSafeCall(cuParamSetSize(Kernel, paramOffset));

      if(sharedMemorySize > 0)
        cuSafeCall(cuFuncSetSharedSize(Kernel, sharedMemorySize));

      //Set the thread-block size configuration
      cuSafeCall(cuFuncSetBlockShape(Kernel,  LocalWork[0],  LocalWork[1],  LocalWork[2]));



//       #define DEBUG_PRINT
      #ifdef DEBUG_PRINT
        CUevent start,stop;

        (cuEventCreate(&start, CU_EVENT_DEFAULT));
        (cuEventCreate(&stop, CU_EVENT_DEFAULT));
        (cuEventRecord(start, 0));      
        fprintf(stderr, "Starting kernel: %s and waiting to finish...", KernelName);
      #endif

      cuSafeCall(cuLaunchGridAsync  (Kernel, GlobalWork[0], GlobalWork[1], 0));
      
      #ifdef DEBUG_PRINT
        double t0 = get_time_test();
        cuSafeCall(cuCtxSynchronize());
        double execTime = ((get_time_test()-t0));      
        (cuEventRecord(stop, 0));
        (cuEventSynchronize(stop));
        float time;
        (cuEventElapsedTime(&time, start, stop));
        (cuEventDestroy(start));
        (cuEventDestroy(stop));
        fprintf(stderr, "Complete, took: %lg \t%f \n", execTime, time);
      #endif

//      fprintf(stderr,"%s took:\t%f\t millisecond\n", KernelName,time);


#ifdef TIMING_STATS
     double t0 = get_time_test();
     cuSafeCall(cuCtxSynchronize());
     unsigned long execTime = ((get_time_test()-t0))*1000000000;
//     fprintf(stderr,"Executing on command queue: Kernel: %s\t Took: %d  Threads: %d\n", KernelName, execTime, LocalWork[0]);
     fprintf(stderr, "Kernel: %s   TOOK: %lg\tNTHREAD: %d\tNMULTI: %d \n", KernelName,  get_time_test() - t0, NTHREADS, NBLOCKS_PER_MULTI);
#endif
    // double t0 = get_time_test();
  //   cuSafeCall(cuCtxSynchronize());
//     fprintf(stderr, "Kernel: %s   TOOK: %lg\tNTHREAD: %d\tNPIPES: %d\tNMULTI: %d \n", KernelName,  get_time_test() - t0, NTHREADS, NPIPES, NBLOCKS_PER_MULTI);

//      cuSafeCall(cuCtxSynchronize());
  
        //Prevent compiler warning, event variable is for OpenCL function definition compatability
        event = event;
    }
    ////

    const CUfunction&  get_kernel() const {return Kernel;}
    const CUmodule&   get_program() const {return cuModule;}  //Program == module?
    const CUcontext& get_context() const {return Context;}
    int        get_command_queue() const {return -1;}
    int localDim()  const {return  LocalWork[0]* LocalWork[1];};
    int globalDim() const {return GlobalWork[0]*GlobalWork[1]*localDim();};
    int num_groups() const {return globalDim()/localDim();};
    int	            get_workGroupMultiple() const {return (int)WorkGroupSizeMultiple;}
    int               get_workGroupMaxSize()  const {return (int)WorkGroupMaxSize;}
    
    
    
    void wait() const {
      cuSafeCall(cuCtxSynchronize());
    }

  };

};

#endif // __CUDADEV_H__


