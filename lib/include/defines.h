#ifndef __DEFINES_H__
#define __DEFINES_H__


// #define DEBUG_PRINT
// #define REMAP



//GPU configuration settings

//Neighbour information
#define NGB_PP 256
#define NGB_PB 256
#define NGB

//GPU config configuration
//GPU config configuration
#ifndef NBLOCKS_PER_MULTI


#define _KEPLER_

#ifdef _KEPLER_
#define NBLOCKS_PER_MULTI  4
#else  /* FERMI, GT200, Cypress, Tahiti */
#define NBLOCKS_PER_MULTI  2 
#endif

#endif  /* NBLOCKS_PER_MULTI */

#ifndef NPIPES
#define NPIPES        256
#endif

#ifndef NTHREADS
#define NTHREADS      256
#endif




#endif
