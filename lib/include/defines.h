#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <string>

// #define DEBUG_PRINT

//GPU configuration settings

//Neighbour information
#define NGB_PP 256
#define NGB_PB 256
#define NGB

//The next line will enable some extra hand tuned block-size
//optimizations. But this can be device/resource dependend and 
//should only be used if you know that it will work 
//(test if needed to find out if it works)
#define ENABLE_THREAD_BLOCK_SIZE_OPTIMIZATION



//Enable this define to let smallN be handled by CPU
#define CPU_SUPPORT



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
    default:    //Future proof...
      return 4;
  }  
}
#else  /* NBLOCKS_PER_MULTI */

inline int getBlocksPerSM(int devMajor, int devMinor, string deviceName)
{
  return NBLOCKS_PER_MULTI; //Some Makefile default
}
#endif










#ifndef NPIPES
#define NPIPES        256
#endif

#ifndef NTHREADS
#define NTHREADS      256
#endif




#endif
