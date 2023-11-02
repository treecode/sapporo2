#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <string>

// #define DEBUG_PRINT

//Neighbour information
#define NGB_PB 256
#define NGB


#ifndef NPIPES
#define NPIPES        16384
#endif

#ifndef NTHREADS
#define NTHREADS      256

//Number of threads used to integrate the NTHREADS particles. If these two 
//are different with NTHREADS << NTHREADS2 you can make 2D thread-blocks
//when using NTHREADS particles. 
#define NTHREADS2     512

#else
#define NTHREADS2     NTHREADS    
#endif


enum { GRAPE5   = 0, FOURTH, SIXTH, EIGHT};        //0, 1, 2, 3
enum { FLOAT  = 0, DOUBLESINGLE, DOUBLE}; //default is 0, double precision 1

//See kernels.cu for the implementation/configuration of the different
//integrators and or to add more combinations
inline const char* get_kernelName(const int integrator,
                                  const int precision,
                                  int       &perThreadSM)
{
  switch(integrator)
  {
    case GRAPE5:
      if(precision == FLOAT){
        perThreadSM = sizeof(float4);
        return "dev_evaluate_gravity_second_float";
      }else if(precision == DOUBLESINGLE){
        perThreadSM = sizeof(float4)*2;
        return "dev_evaluate_gravity_second_DS";}
    case FOURTH:
      if(precision == DOUBLESINGLE){
        perThreadSM = sizeof(float4)*2 + sizeof(float4);
        return "dev_evaluate_gravity_fourth_DS";
      }else if(precision == DOUBLE){
        perThreadSM = sizeof(double4) + sizeof(double4);
        return "dev_evaluate_gravity_fourth_double";}
    case SIXTH:
      if(precision == DOUBLESINGLE)
      {
#ifdef _OCL_
          fprintf(stderr, "ERROR: Sixth order integrator with double single precision");
          fprintf(stderr, "ERROR: is not implemented in OpenCL, only in CUDA. Please");
          fprintf(stderr, "ERROR: file an issue on GitHub if you need this combination.");
          exit(1);
#else
          perThreadSM = sizeof(float4)*2 + sizeof(float4) + sizeof(float3);
#endif
          return "dev_evaluate_gravity_sixth_DS"; 
      }
      else if(precision == DOUBLE){
#ifdef _OCL_
        perThreadSM = sizeof(double4) + sizeof(double4) + sizeof(double4);
#else
        perThreadSM = sizeof(double4) + sizeof(double4) + sizeof(double3);
#endif
        return "dev_evaluate_gravity_sixth_double";
      }
    default:
      break;
  };//switch
  
  //Here we come if all switch/case/if combo's failed
  fprintf(stderr,"ERROR: Unknown combination of integrator type ( %d ) and precision ( %d ) \n", integrator, precision);
  fprintf(stderr,"ERROR: See 'defines.h' for the possible combinations \n");
  exit(0);
  return "";
}

//The next line will enable some extra hand tuned block-size
//optimizations. But this can be device/resource dependent and 
//should only be used if you know that it will work 
//(test if needed to find out if it works)
// #define ENABLE_THREAD_BLOCK_SIZE_OPTIMIZATION



//Enable this define to let smallN be handled by CPU
//#define CPU_SUPPORT


//GPU config configuration
#ifndef NBLOCKS_PER_MULTI

//Put this in this file since it is a setting
inline int getBlocksPerSM(int devMajor, int devMinor, std::string deviceName)
{
  devMinor = devMinor; //Prevent unused warnings
  
  switch(devMajor)
  { 
    case -1:
      //Non nvidia
      if(deviceName.find("Tahiti") != std::string::npos)
        return 2; //AMD GPU Tahiti
      if(deviceName.find("Cypress") != std::string::npos)
        return 2; //AMD GPU Cypress
      
      return 2;    //default NON-nvidia
    
    case 1:     //GT200, G80
      return 2;    
    case 2:     //Fermi
      return 2;     
    case 3:     //Kepler
      return 4;
    case 6:     //Pascal
      return 16;
    default:    //Future proof...
      return 16;
  }  
}
#else  /* NBLOCKS_PER_MULTI */

inline int getBlocksPerSM(int devMajor, int devMinor, std::string deviceName)
{
  return NBLOCKS_PER_MULTI; //Some Makefile default
}
#endif


/*
  Putting this comment here for lack of better place
  To reach performance within 5% of SDK example:
  put __launch_bounds__(1024,1) in the gravity kernel definition
  NTHREADS and NTHREADS2 set to 1024
  blocksPerSM (3.5): 2
  and add "--ftz=true"  to the NVCC flags
  And launch: ./test_performance_rangeN_g5_cuda 262144 CUDA/kernels.ptx 0 0
*/



#endif
