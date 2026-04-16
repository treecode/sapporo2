#include "sapporohostclass.h"

#include <sys/time.h>
#include <algorithm>

#include "hostFunc.h"

/*


Combined variables:

pos_i.w = h2    --> Used as neighbourhood sphere radius
vel_i.w = eps2
vel_j.w = eps2

*/

#ifdef __INCLUDE_KERNELS__
#ifdef _OCL_
#include "OpenCL/kernels4th.clh"
#include "OpenCL/kernels4thDP.clh"
#include "OpenCL/kernels6th.clh"
#include "OpenCL/kernelsG5DS.clh"
#include "OpenCL/kernelsG5SP.clh"
#include "OpenCL/sharedKernels.clh"
#else
#include "CUDA/kernels.ptxh"
#endif
#endif


inline int host_float_as_int(float val)
{
  union{float f; int i;} u; //__float_as_int
  u.f           = val;
  return u.i;
}

inline int n_norm(int n, int j) {
  n = ((n-1)/j) * j + j;
  if (n == 0) n = j;
  return n;
}

static __inline__  double4 make_double4(double x, double y, double z, double w)
{
  double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__  double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

static __inline__ int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}



double __inline__ get_time() {
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +
          1.e-6*((double) Tvalue.tv_usec));
}

//Have to make this a static pointer in order to be able to
//use it in combination with the threadprivate directive
static sapporo2::device   *sapdevice;
#pragma omp threadprivate(sapdevice)

int numberOfGPUUsedBySapporo;

//References to the sapdevice per thread
sapporo2::device **deviceList;


/*

Application library interface

*/


int sapporo::open(std::string kernelFile, int *devices, 
                  int nprocs, int order, int precision)  
{
  //Set the integration order
  integrationOrder      = order;
  integrationPrecision  = precision;

  cout << "Integration order used: " << integrationOrder << " (0=GRAPE5, 1=4th, 2=6th, 3=8th)\n";
  cout << "Integration precision used: " << precision << " (0=FLOAT, 1 = DOUBLESINGLE, 2=DOUBLE)\n";

  dev::context        contextTest;  //Only used to retrieve the number of devices

  int numDev = 0;
  
  
  #ifdef __OPENCL_DEV__
    numDev = contextTest.getDeviceCount(CL_DEVICE_TYPE_GPU, 0);
  #else
    numDev = contextTest.getDeviceCount();
  #endif

  cout << "Number of cpus available: " << omp_get_num_procs() << endl;
  cout << "Number of gpus available: " << numDev << endl;

  // create as many CPU threads as there are CUDA devices and create the contexts

  int numThread = abs(nprocs);

  if(numThread == 0)    //Use as many as available
  {
    numThread = numDev;
  }
  
  deviceList = new sapporo2::device*[numThread];
  
  numberOfGPUUsedBySapporo = numThread;

//   omp_set_num_threads(numThread);
  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    //Create context for each thread
    unsigned int tid      = omp_get_thread_num();
    sapdevice             = new sapporo2::device();
     
    deviceList[tid] = sapdevice;

    //Let the driver try to get a device if nprocs < 0
    //Use 1...N if nprocs == 0
    //Use list if nprocs > 0
    int dev = -1;

    if(nprocs == 0) //Device ID is thread ID
    {
      dev = tid;
    }
    else if(nprocs > 0)
    {     
      dev = devices[tid]; //The user gave us a set of device ids
    }

    //Assign the device and load the kernels
    sapdevice->assignDevice(dev, integrationOrder);
    const char *gravityKernel = get_kernelName(integrationOrder, precision,sapdevice->sharedMemPerThread);
    sapdevice->loadComputeKernels(kernelFile.c_str(), gravityKernel);

    if(tid == 0)
    {
      nCUDAdevices = omp_get_num_threads();
    }
    
    //Allocate initial memory for 16k particles per device
    sapdevice->allocateMemory(16384, get_n_pipes());
    nj_max = 16384;    
  }//end pragma omp parallel

  //Used to store j-memory particle counters
  jCopyInformation.resize(nCUDAdevices);
  
  
  CPUThreshold = -1; //By Default GPU is always used
  
  
#if 0
  #ifdef CPU_SUPPORT
  
    const int nMaxTest = 2049;
    const int nMaxLoop = 2049;
    const int nIncrease = 16;
    //At the start of the program figure out at which point the GPU will be faster
    //than the host CPU. This can either be based on ni, nj, or on a combination 
    //of ni*nj = #interactions. Then if #interactions < GPUOptimal do host compute
    //otherwise do GPU compute. Stored in CPUThreshold
  
    //#pragma omp parallel
    {
      //First fill the ids with valid info otherwise testing might fail, if all ids are 0
      for(int i=0; i < nMaxTest; i++) 
      {
        if(i < NPIPES)
        {
          sapdevice->id_i[i]    = i;
          sapdevice->pos_i[i].x =  (1.0 - 2.0*drand48());
          sapdevice->pos_i[i].y =  (1.0 - 2.0*drand48());
          sapdevice->pos_i[i].z =  (1.0 - 2.0*drand48());
          sapdevice->pos_i[i].w =  1./1024;
          sapdevice->vel_i[i].x =  drand48() * 0.1;
          sapdevice->vel_i[i].y =  drand48() * 0.1;
          sapdevice->vel_i[i].z =  drand48() * 0.1;
        }
  
        if(i < nj_max)
        {
          sapdevice->id_j[i]     = i;
          sapdevice->pPos_j[i].x =  (1.0 - 2.0*drand48());
          sapdevice->pPos_j[i].y =  (1.0 - 2.0*drand48());
          sapdevice->pPos_j[i].z =  (1.0 - 2.0*drand48());
          sapdevice->pPos_j[i].w =  1./1024;
          sapdevice->pVel_j[i].x =  drand48() * 0.1;
          sapdevice->pVel_j[i].y =  drand48() * 0.1;
          sapdevice->pVel_j[i].z =  drand48() * 0.1;          
        }
      }
      
      
      //Some temp buffers, are being used multiple
      //times and contain only bogus data      
      double (*pos)[3]  = new double[nMaxTest][3];
      double (*vel)[3]  = new double[nMaxTest][3];
      double (*acc)[3]  = new double[nMaxTest][3];
      double (*jrk)[3]  = new double[nMaxTest][3];
      double  *tempBuff = new double[nMaxTest];
      
      double *timingMatrixGPU = new double[nMaxTest*nMaxTest];
      double *timingMatrixCPU = new double[nMaxTest*nMaxTest];
      
      //First call to initialize device
      evaluate_gravity(1, 1);            
      retrieve_i_particle_results(1);
      
      double tTime = 0;
      CPUThreshold = -1; //Negative to force GPU timings
      for(int k=0; k < nMaxLoop; k+=nIncrease) //number of i-particles
      {
        for(int m=0; m < nMaxLoop; m+=nIncrease) //number of j-particles
        {
          int kk=k, mm=m;
          if(k==0) kk = 1;
          if(m==0) mm = 1;
          
          timingMatrixGPU[m*nMaxTest+k] = 0;
          for(int n=0; n < 10; n++)
          {
            double t0 = get_time();
            set_time(tTime);//set time
            startGravCalc(mm,kk,
                          &sapdevice->id_i[0], pos,
                          vel,acc, acc,tempBuff,
                          1./ nMaxTest, tempBuff, NULL);
            getGravResults(mm,kk,
                          &sapdevice->id_i[0], pos,
                          vel, 1./ nMaxTest, NULL,
                          acc, jrk, acc, jrk, tempBuff,
                          NULL, tempBuff, false);
//             fprintf(stderr, "TEST DEV: Took: nj: %d  ni: %d \t %g\n", m, k,   get_time() - t0);
            
            timingMatrixGPU[m*nMaxTest+k] += get_time() - t0;
            tTime += 0.0001;
          }//for n
        }//for m
      }//for k
      
#if 0
      CPUThreshold = 10e10; //Huge to force CPU timings
      //First call outside loop, to boot-up openMP
      evaluate_gravity_host(1, 1);  
      evaluate_gravity_host_vector(1, 1);  
      
      tTime = 0;
      for(int k=1; k < nMaxLoop; k+=nIncrease)
      {
        for(int m=1; m < nMaxLoop; m+=nIncrease)
        {
          timingMatrixCPU[m*nMaxTest+k] = 0;
          for(int n=0; n < 10; n++)
          {         
            int kk=k, mm=m;
            if(k==0) kk = 1;
            if(m==0) mm = 1;            
            double t0 = get_time();
            set_time(tTime);//set time
            startGravCalc(mm,kk,
                          &sapdevice->id_i[0], pos,
                          vel,acc, acc,tempBuff,
                          1./ nMaxTest, tempBuff, NULL);
            getGravResults(mm,kk,
                          &sapdevice->id_i[0], pos,
                          vel, 1./ nMaxTest, NULL,
                          acc, jrk, acc, jrk, tempBuff,
                          NULL, tempBuff, false);
//             fprintf(stderr, "TEST CPU: Took: nj: %d  ni: %d \t %g\n", m, k,   get_time() - t0);
            timingMatrixCPU[m*nMaxTest+k] += get_time() - t0;
            tTime += 0.0001;  
          }//for n
        }//for m
      } //for k
    #endif  
    
    //Write timing data to file
    FILE *foutT = fopen("data.txt","w");
      //Print timing results GPU
      fprintf(stderr, "GPU timings:\nni");
      fprintf(foutT, "ni");
      for(int i=1; i < nMaxLoop; i+=nIncrease)
        fprintf(foutT, "\t%d", i);      
      fprintf(foutT, "\n");
      fprintf(foutT, "nj\n");
      
      for(int j=1; j < nMaxLoop; j+=nIncrease)
      {
        fprintf(foutT, "%d\t", j);
        for(int i=1; i < nMaxLoop; i+=nIncrease)
        {
          fprintf(foutT, "%f\t", timingMatrixGPU[j*nMaxTest+i]);
        }
        fprintf(foutT, "\n");
      }
      fclose(foutT);
      exit(0);
      fprintf(stderr, "\nCPU timings:\nni");
      for(int i=1; i < nMaxLoop; i+=nIncrease)
        fprintf(stderr, "\t%d", i);      
      fprintf(stderr, "\n");
      fprintf(stderr, "nj\n");
      
      for(int j=1; j < nMaxLoop; j+=nIncrease)
      {
        fprintf(stderr, "%d\t", j);
        for(int i=1; i < nMaxLoop; i+=nIncrease)
        {
          fprintf(stderr, "%f\t", timingMatrixCPU[j*nMaxTest+i]);
        }
        fprintf(stderr, "\n");
      }      
      
      
      fprintf(stderr, "GPU timings:\n");
           
      for(int j=1; j < nMaxLoop; j+=nIncrease)
      {        
        for(int i=1; i < nMaxLoop; i+=nIncrease)
        {
//           fprintf(stderr,"%f\t%f\t%f\t%f\n",
            fprintf(stderr,"%d\t%f\t%f\n", 
                  i*j,  timingMatrixGPU[j*nMaxTest+i],
                   timingMatrixCPU[j*nMaxTest+i]);
                  
//                   j / timingMatrixGPU[j*nMaxTest+i],
//                   i / timingMatrixGPU[j*nMaxTest+i],
//                   j / timingMatrixCPU[j*nMaxTest+i],
//                   i / timingMatrixCPU[j*nMaxTest+i]);          
        }//for i        
        fprintf(stderr, "\n");
      } //for j
      
      
      //TODO set some interaction count number that is the break-even point 
      //between CPU and GPU computations
      delete[] pos;
      delete[] vel;
      delete[] acc;
      delete[] jrk;
      delete[] tempBuff;
      delete[] timingMatrixGPU;
      delete[] timingMatrixCPU;      
    }
    
    
    exit(0);
  #endif //ifdef CPU support
#endif

  return 0;
}

void sapporo::cleanUpDevice()
{
  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    if(sapdevice != NULL)
    {
      delete sapdevice;
      sapdevice = NULL;      
    }
  } //end omp parallel
  
  delete[] deviceList;
  deviceList = NULL;
}



int sapporo::close() {
  cerr << "Sapporo::close\n";
  isFirstSend = true;
  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    //TODO call the free memory function
    delete sapdevice;
    sapdevice = NULL;
  }

  return 0;
}

//Set integrator prediction time-step
int sapporo::set_time(double time) {
  #ifdef DEBUG_PRINT
    cerr << "set_time: " << time << endl;
  #endif

  t_i           = time;
  predict       = true;
  predJOnHost   = false;

  return 0;
}

int sapporo::set_no_time() {
  #ifdef DEBUG_PRINT
    cerr << "set_no_time" << endl;
  #endif

  t_i           = t_i;
  predict       = false; //No prediction when no predict is called
  predJOnHost   = false;

  return 0;
}

//Get the number of particles that can be integrated at the same time
int sapporo::get_n_pipes() {
  return n_pipes;
}

int sapporo::set_j_particle(int    address,
                            int    id,
                            double tj, double dtj,
                            double mass,
                            double k18[3],       double j6[3],
                            double a2[3],        double v[3],
                            double x[3],         double snp[3],
                            double crk[3],       double eps) {

  #ifdef DEBUG_PRINT
    cerr << "set_j_particle (Addr: " << address << "  Id: " << id << " )\n";
  #endif

  //Prevent unused compiler warning
  k18 = k18;
    
  predJOnHost  = false; //Reset the buffers on the device since they can be modified
  nj_updated   = true;  //There are particles that are updated

  //Check if the address does not fall outside the allocated memory range
  //if it falls outside that range increase j-memory by 10%
    
  if (address >= nj_max) {
    fprintf(stderr, "Increasing nj_max! Nj_max was: %d  to be stored address: %d \n",
            nj_max, address);
    increase_jMemory();

    //Extra check, if we are still outside nj_max, we quit since particles are not
    //nicely send in order
    if (address >= nj_max) {
      fprintf(stderr, "Increasing nj_max was not enough! Send particles in order to the library! Exit\n");
      exit(-1);
    }
  }

  //Memory has been allocated, now we can store the particles
  //First calculate on which device this particle has to be stored
  //and on which physical address on that device. Note that the particles
  //are distributed to the different devices in a round-robin way (based on the addres)
  int dev           = address % nCUDAdevices;
  int devAddr       = address / nCUDAdevices;
  int storeLoc      = jCopyInformation[dev].count;

  //Store this information, incase particles get overwritten
  map<int, int4>::iterator iterator = mappingFromIndexToDevIndex.find(address);
  map<int, int4>::iterator end      = mappingFromIndexToDevIndex.end();


  if(iterator != end)
  {
    //Particle with this address has been set before, retrieve previous
    //calculated indices and overwrite them with the new info
    int4 addrInfo = (*iterator).second;
    dev           = addrInfo.x;
    storeLoc      = addrInfo.y;
    devAddr       = addrInfo.z;
  }
  else
  {
    //New particle not set before, save address info and increase particles
    //on that specific device by one
    mappingFromIndexToDevIndex[address] = make_int4(dev, storeLoc, devAddr, -1);
    jCopyInformation[dev].count++;
  }


  deviceList[dev]->pos_j_temp[storeLoc] = make_double4(x[0], x[1], x[2], mass);
  deviceList[dev]->address_j[storeLoc]  = devAddr;
  
  if(integrationOrder > GRAPE5)
  {
    deviceList[dev]->t_j_temp[storeLoc]          = make_double2(tj, dtj);
    deviceList[dev]->vel_j_temp[storeLoc]        = make_double4(v[0], v[1], v[2], eps);
    deviceList[dev]->acc_j_temp[storeLoc]        = make_double4(a2[0], a2[1], a2[2], 0.0);
    deviceList[dev]->jrk_j_temp[storeLoc]        = make_double4(j6[0], j6[1], j6[2], 0.0);
    deviceList[dev]->id_j_temp[storeLoc]         = id;
    //For 6th and 8 order we need more parameters
    if(integrationOrder > FOURTH)
    {
      deviceList[dev]->snp_j_temp[storeLoc]        = make_double4(snp[0], snp[1], snp[2], 0.0);
      deviceList[dev]->crk_j_temp[storeLoc]        = make_double4(crk[0], crk[1], crk[2], 0.0);
    }
  }
  

  #ifdef CPU_SUPPORT
    //Put the new j particles directly in the correct location on the host side.
    deviceList[dev]->pos_j[devAddr] = make_double4(x[0], x[1], x[2], mass);
    
    if(integrationOrder > GRAPE5)
    {
      deviceList[dev]->t_j[devAddr]          = make_double2(tj, dtj);
      deviceList[dev]->vel_j[devAddr]        = make_double4(v[0], v[1], v[2], eps);
      deviceList[dev]->acc_j[devAddr]        = make_double4(a2[0], a2[1], a2[2], 0.0);
      deviceList[dev]->jrk_j[devAddr]        = make_double4(j6[0], j6[1], j6[2], 0.0);
      deviceList[dev]->id_j[devAddr]         = id;
      //For 6th and 8 order we need more parameters
      if(integrationOrder > FOURTH)
      {
        deviceList[dev]->snp_j[devAddr]        = make_double4(snp[0], snp[1], snp[2], 0.0);
        deviceList[dev]->crk_j[devAddr]        = make_double4(crk[0], crk[1], crk[2], 0.0);
      }
    }  
  #endif


  #ifdef DEBUG_PRINT
    if(integrationOrder == GRAPE5)
    {
      fprintf(stderr, "Setj ad: %d\tid: %d storeLoc: %d \tpos: %f %f %f m: %f \n", address, id, storeLoc, x[0],x[1],x[2], mass);
    }
    else
    {
      fprintf(stderr, "Setj ad: %d\tid: %d storeLoc: %d \tpos: %f %f %f\t mass: %f \tvel: %f %f %f", address, id, storeLoc, x[0],x[1],x[2],mass, v[0],v[1],v[2]);
      fprintf(stderr, "\tacc: %f %f %f \n", a2[0],a2[1],a2[2]);
      if(integrationOrder > FOURTH)
      {
        fprintf(stderr, "\tsnp: %f %f %f ", snp[0],snp[1],snp[2]);
        fprintf(stderr, "\tcrk: %f %f %f \n", crk[0],crk[1],crk[2]);
      }
    }
  #endif

  return 0;
};

void sapporo::increase_jMemory()
{
  #ifdef DEBUG_PRINT
    cerr << "Increase jMemory\n";
  #endif

  //Increase by 10 % and round it of so we can divide it by the number of devices
  int temp = nj_max * 1.1;

  temp = temp / nCUDAdevices;
  temp++;
  temp = temp * nCUDAdevices; 

  nj_max = temp;  


  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    //Compute number of particles to allocate on this device
    int nj_max_local    = nj_max / nCUDAdevices;

    sapdevice->reallocJParticles(nj_max_local);
  } //end parallel section
}


void sapporo::startGravCalc(int    nj,          int ni,
                            int    id[],        double xi[][3],
                            double vi[][3],     double a[][3],
                            double j6old[][3],  double phiold[3],
                            double eps2,        double h2[],
                            double eps2_i[]) {

  #ifdef DEBUG_PRINT
    cerr << "calc_firsthalf ni: " << ni << "\tnj: " << nj << "integrationOrder: "<< integrationOrder << endl;
  #endif

  if(ni == 0 || nj == 0)  return;

  //Prevent unused compiler warning
  j6old  = j6old;
  phiold = phiold;

  //Its not allowed to send more particles than n_pipes
  assert(ni <= get_n_pipes());
  
  EPS2     = eps2;  

  //Copy i-particles to device structures, first only to device 0
  //then from device 0 we use memcpy to get the data into the 
  //other devs buffers
  int toDevice = 0;
  
  for (int i = 0; i < ni; i++)
  {
    deviceList[toDevice]->pos_i[i] = make_double4(xi[i][0], xi[i][1], xi[i][2], h2[i]);

    if(integrationOrder > GRAPE5)
    {
      deviceList[toDevice]->id_i[i]  = id[i];        
      deviceList[toDevice]->vel_i[i] = make_double4(vi[i][0], vi[i][1], vi[i][2], eps2);
      
      if(eps2_i != NULL)  //Seperate softening for i-particles
        deviceList[toDevice]->vel_i[i].w = eps2_i[i];              
      
      if(integrationOrder > FOURTH)
      {
        deviceList[toDevice]->acc_i[i] = make_double4(a[i][0], a[i][1], a[i][2], 0);
      }      
    }


    #ifdef DEBUG_PRINT
      if(integrationOrder == GRAPE5)
      {
        fprintf(stderr, "Inpdevice= %d,\ti: %d\tindex: %d\teps2: %f\t%f\t%f\t%f \n",
              -1,i, 0, eps2, xi[i][0],xi[i][1],xi[i][2]);
      }
      else
      {
        fprintf(stderr, "Inpdevice= %d,\ti: %d\tindex: %d\teps2: %f\t%f\t%f\t%f\t%f\t%f\t%f",
              -1,i,id[i], eps2, xi[i][0],xi[i][1],xi[i][2],vi[i][0],vi[i][1] ,vi[i][2]);
        
        if(integrationOrder > FOURTH)
          fprintf(stderr, "\t%f %f %f\n", a[i][0], a[i][1], a[i][2]);
        else
          fprintf(stderr, "\n");    
      }
    #endif
  }//for i

  //Copy i particles from host buffer of device 0 to the devices host side buffers  
  for(int i = toDevice+1;  i < nCUDAdevices; i++)
  {
      memcpy(&deviceList[i]->pos_i[0], &deviceList[0]->pos_i[0], sizeof(double4) * ni);
      if(integrationOrder > GRAPE5)
      {
        memcpy(&deviceList[i]->vel_i[0], &deviceList[0]->vel_i[0], sizeof(double4) * ni);
        memcpy(&deviceList[i]->id_i[0],  &deviceList[0]->id_i[0],  sizeof(int)     * ni);  
        
        if(integrationOrder > FOURTH)
          memcpy(&deviceList[i]->acc_i[0], &deviceList[0]->acc_i[0], sizeof(double4) * ni);        
      }
  }
  


  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    if (nj_updated) {
      //Get the number of particles set for this device
      int devCount = jCopyInformation[omp_get_thread_num()].count;
      if(devCount > 0)
      {
        send_j_particles_to_device(devCount);
      }
    }

    //ni is the number of particles in the pipes
    send_i_particles_to_device(ni);

    //nj is the total number of particles to which the i particles have to
    //be calculated. For direct N-body this is usually equal to the total
    //number of nj particles that have been set by the calling code


    //Calculate the number of nj particles that are used per device
    int nj_per_dev = nj / nCUDAdevices;
    if(omp_get_thread_num() < (nj  % nCUDAdevices))
      nj_per_dev++;

    evaluate_gravity(ni,  nj_per_dev);

    sapdevice->dev_ni = ni;
  }//end parallel section


  nj_modified   = -1;
  predict       = false;
  nj_updated    = false;

  //Clear the address to dev/location mapping
  mappingFromIndexToDevIndex.clear();
} //end calc_first

int sapporo::getGravResults(int nj, int ni,
                            int index[],
                            double xi[][3],      double vi[][3],
                            double eps2,         double h2[],
                            double acc[][3],     double jerk[][3],
                            double snp[][3],     double crk[][3],
                            double pot[],        int nnbindex[],
                            double dsmin_i[],    bool ngb) {

  #ifdef DEBUG_PRINT
    fprintf(stderr, "calc_lasthalf2 device= %d, ni= %d nj = %d \n", -1, ni, nj);
  #endif
    
  if(ni == 0 || nj == 0)  return 0;
  //Prevent unused compiler warning    
  nj = nj; index = index; xi = xi; vi = vi; eps2 = eps2; h2 = h2;
  
  double ds_min[NPIPES];
  for (int i = 0; i < ni; i++) {
    pot[i] = acc[i][0]  = acc[i][1]  = acc[i][2]  = 0;
    if(integrationOrder > GRAPE5)
    {
      jerk[i][0] = jerk[i][1] = jerk[i][2] = 0;
    }

    if(ngb)
      nnbindex[i] = 0;
    ds_min[i] = 1.0e10;

    if(integrationOrder > FOURTH)
    {
      snp[i][0] = snp[i][1] = snp[i][2] = 0;
      crk[i][0] = crk[i][1] = crk[i][2] = 0;
    }
  }

  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    //Retrieve data from the devices (in parallel)
    retrieve_i_particle_results(ni);
  }


  //Reduce the data from the different devices into one final results
  for (int dev = 0; dev < nCUDAdevices; dev++) {
    for (int i = 0; i < ni; i++)
     {
        pot[i]    += deviceList[dev]->iParticleResults[i].w;
        acc[i][0] += deviceList[dev]->iParticleResults[i].x;
        acc[i][1] += deviceList[dev]->iParticleResults[i].y;
        acc[i][2] += deviceList[dev]->iParticleResults[i].z;      

      if(integrationOrder > GRAPE5)
      {
        jerk[i][0] += deviceList[dev]->iParticleResults[ni+i].x;
        jerk[i][1] += deviceList[dev]->iParticleResults[ni+i].y;
        jerk[i][2] += deviceList[dev]->iParticleResults[ni+i].z;

        double  ds  = deviceList[dev]->ds_i[i].y;        

        if(ngb)   //If we want nearest neighbour
        {
          if (ds < ds_min[i]) {
            //int nnb     = (int)(deviceList[dev]->iParticleResults[ni+i].w);
            int nnb =  host_float_as_int(deviceList[dev]->ds_i[i].x);
            nnbindex[i] = nnb;
            ds_min[i]   = ds;
            if(dsmin_i != NULL)
              dsmin_i[i]  = ds;
          }
        }
      } //End if > GRAPE5

      if(integrationOrder > FOURTH)
      {
        snp[i][0] += deviceList[dev]->iParticleResults[2*ni+i].x;
        snp[i][1] += deviceList[dev]->iParticleResults[2*ni+i].y;
        snp[i][2] += deviceList[dev]->iParticleResults[2*ni+i].z;

        // Possible 8th order extension        
        // crk[i][0] += deviceList[dev]->crk_i[i].x;
        // crk[i][1] += deviceList[dev]->crk_i[i].y;
        // crk[i][2] += deviceList[dev]->crk_i[i].z;
      }
      
      

//       fprintf(stderr,"%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",
//               index[i], pot[i], acc[i][0], acc[i][1], acc[i][2], jerk[i][0], jerk[i][1], jerk[i][2]);
/*      fprintf(stderr,"%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%d\t%g\n",
              index[i], pot[i], acc[i][0], acc[i][1], acc[i][2], jerk[i][0], jerk[i][1], jerk[i][2],
              nnbindex[i], ds_min[i]);   */   

    }
  }
  
  return 0;
};

int sapporo::read_ngb_list(int cluster_id)
{
  //Prevent unused compiler warning    
  cluster_id = cluster_id;
    

  #ifdef DEBUG_PRINT
    fprintf(stderr, "read_ngb_list\n");
  #endif

  bool overflow = false;
  int *ni       = new int[omp_get_max_threads()];

  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    //Retrieve data from the devices
    ni[omp_get_thread_num()] = fetch_ngb_list_from_device();
  }

  for(int dev = 0; dev < nCUDAdevices; dev++)
  {
    for (int j = 0; j < ni[dev]; j++)
    {
      if(deviceList[dev]->ngb_count_i[j]  >= NGB_PB) {
        overflow = true;
      }
    }
  }

  delete[] ni;
  return overflow;
} //end read_ngb_list

int sapporo::get_ngb_list(int cluster_id,
                          int ipipe,
                          int maxlength,
                          int &nblen,
                          int nbl[]) {
  //Prevent unused compiler warning    
  cluster_id = cluster_id;  

//   if (ipipe >= devs[0].ni) {
//     fprintf(stderr, "Fatal! ipipe= %d >= dev.ni= %d. I give up.\n",
//             ipipe, devs[0].ni);
//     exit(-1);
//   }
 #ifdef DEBUG_PRINT
  fprintf(stderr, "get_ngb_list ipipe: %d\n", ipipe);
 #endif

  

  bool overflow = false;
  nblen         = 0;
  for (int i = 0; i < nCUDAdevices; i++) {
    int offset  = NGB_PB*ipipe;
    int len     = deviceList[i]->ngb_count_i[ipipe];
    
    int toCopy = min(len, maxlength-nblen);
    memcpy(nbl+nblen, &deviceList[i]->ngb_list_i[offset], sizeof(int)*toCopy);
    nblen += len;
    if (nblen >= maxlength) {
      overflow = true;
      break;
    }
  }  
  
  sort(nbl, nbl + min(nblen, maxlength));


  if (overflow) {
    fprintf(stderr, "sapporo::get_ngb_list(..) - overflow for ipipe= %d, ngb= %d\n",
         ipipe, nblen);
  }

  return overflow;
}

/*

  Device communication functions

*/


void sapporo::send_j_particles_to_device(int nj_tosend)
{
  #ifdef DEBUG_PRINT
    cerr << "send_j_particles_to_device nj_tosend: " << nj_tosend << std::endl;
  #endif

  //This function is called inside an omp parallel section

  //Copy the particles to the device memory
  assert(nj_tosend == jCopyInformation[omp_get_thread_num()].count);

  sapdevice->pos_j_temp.h2d(nj_tosend);
  sapdevice->address_j.h2d(nj_tosend);

  if(integrationOrder > GRAPE5)
  {
    sapdevice->t_j_temp.h2d(nj_tosend);

    sapdevice->vel_j_temp.h2d(nj_tosend);
    sapdevice->acc_j_temp.h2d(nj_tosend);
    sapdevice->jrk_j_temp.h2d(nj_tosend);
    sapdevice->id_j_temp.h2d(nj_tosend);
  }

  if(integrationOrder > FOURTH)
  {
    sapdevice->snp_j_temp.h2d(nj_tosend);
    sapdevice->crk_j_temp.h2d(nj_tosend);
  }

  //Reset the number of particles, that have to be send
  jCopyInformation[omp_get_thread_num()].toCopy += jCopyInformation[omp_get_thread_num()].count;
  jCopyInformation[omp_get_thread_num()].count   = 0;

} //end send j particles


void sapporo::send_i_particles_to_device(int ni)
{
  //This is function is called inside an omp parallel section  
  #ifdef DEBUG_PRINT
    cerr << "send_i_particles_to_device ni: " << ni << endl;
  #endif

  //Copy the host side buffers to the devices
  sapdevice->pos_i.h2d(ni);

  if(integrationOrder > GRAPE5)
  {
    sapdevice->vel_i.h2d(ni);
    sapdevice->id_i.h2d(ni);
    
    if(integrationOrder > FOURTH)
    {
      sapdevice->acc_i.h2d(ni);
    }
  }

} //end send_i_particles_to_device

void sapporo::retrieve_i_particle_results(int ni)
{
  #ifdef DEBUG_PRINT
    cerr << "retrieve_i_particle_results\n";
  #endif
    
  if(executedOnHost == true) return;

  //Called inside an OMP parallel section
    
  sapdevice->iParticleResults.d2h(ni + ni*integrationOrder);
  
  if(integrationOrder > GRAPE5)
  {
    sapdevice->ds_i.d2h(ni);
  }
  
}//retrieve i particles


void sapporo::retrieve_predicted_j_particle(int    addr,        double &mass,
                                            double &id,         double &eps2,
                                            double pos[3],      double vel[3],
                                            double acc[3])
{

  #ifdef DEBUG_PRINT
    cerr << "retrieve_predicted_j_particle address: " << addr << endl;
  #endif


  if(predJOnHost == false)
  {
    //We need to copy the particles back to the host
    #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
    {
      //Retrieve data from the devices (in parallel)
      sapdevice->pPos_j.d2h();
      sapdevice->pVel_j.d2h();
      sapdevice->id_j.d2h();

      if(integrationOrder > FOURTH)
      {
        sapdevice->pAcc_j.d2h();
      }
    }
  }

  //Copy values in the correct buffers, of the calling function
  //NOTE that we have to convert addr into the correct addr and
  //device information before we can retrieve the data
  int dev           = addr % nCUDAdevices;
  int devAddr       = addr / nCUDAdevices;
  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    if(omp_get_thread_num() == dev)
    {
      //This is the device that has the predicted particle stored, now
      //copy back the particle data into the calling function
      pos[0] = sapdevice->pPos_j[devAddr].x;
      pos[1] = sapdevice->pPos_j[devAddr].y;
      pos[2] = sapdevice->pPos_j[devAddr].z;
      mass   = sapdevice->pPos_j[devAddr].w;

      vel[0] = sapdevice->pVel_j[devAddr].x;
      vel[1] = sapdevice->pVel_j[devAddr].y;
      vel[2] = sapdevice->pVel_j[devAddr].z;
      eps2   = sapdevice->pVel_j[devAddr].w;

      if(integrationOrder > FOURTH)
      {
        acc[0] = sapdevice->pAcc_j[devAddr].x;
        acc[1] = sapdevice->pAcc_j[devAddr].y;
        acc[2] = sapdevice->pAcc_j[devAddr].z;
      }

      id     = sapdevice->id_j[devAddr];
    }
  }

  //Indicate that we copied the predicted-j particles back to the host
  predJOnHost = true;

  #ifdef DEBUG_PRINT
    fprintf(stderr, "Getj %d\t%lf\tpos: %g %g %g\tvel: %g %g %g\tacc: %g %g %g\n",
            addr,
            id,
            pos[0],pos[1],pos[2],
            vel[0],vel[1],vel[2],
            acc[0],acc[1],acc[2]);
  #endif

}


//Returns all values of the J-particle as it is set in the device
void sapporo::retrieve_j_particle_state(int addr,       double &mass,
                               double &id,     double &eps2,
                               double pos[3],  double vel[3],
                               double acc[3],  double jrk[3], double ppos[3],
                               double pvel[3], double pacc[3])
{

  #ifdef DEBUG_PRINT
    cerr << "retrieve_j_particle_state, address: " << addr << endl;
  #endif

  if(predJOnHost == false)
  {
    //We need to copy the particles back to the host
    #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
    {
      //Retrieve data from the devices (in parallel)
      sapdevice->pos_j.d2h();

      if(integrationOrder > GRAPE5)
      {
        sapdevice->id_j.d2h();
        sapdevice->pPos_j.d2h();
        sapdevice->pVel_j.d2h();
        sapdevice->vel_j.d2h();
        sapdevice->acc_j.d2h();
        sapdevice->jrk_j.d2h();
      }

      if(integrationOrder > FOURTH)
      {
        sapdevice->pAcc_j.d2h();
      }
    }
  }

  //Copy values in the correct buffers, of the calling function
  //NOTE that we have to convert addr into the correct addr and
  //device information before we can retrieve the data
  int dev           = addr % nCUDAdevices;
  int devAddr       = addr / nCUDAdevices;
  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    if(omp_get_thread_num() == dev)
    {
      //This is the device that has the predicted particle stored, now
      //copy back the particle data into the calling function

      mass   = sapdevice->pos_j[devAddr].w;

      pos[0] = sapdevice->pos_j[devAddr].x;
      pos[1] = sapdevice->pos_j[devAddr].y;
      pos[2] = sapdevice->pos_j[devAddr].z;

      if(integrationOrder > GRAPE5)
      {
        ppos[0] = sapdevice->pPos_j[devAddr].x;
        ppos[1] = sapdevice->pPos_j[devAddr].y;
        ppos[2] = sapdevice->pPos_j[devAddr].z;

        pvel[0] = sapdevice->pVel_j[devAddr].x;
        pvel[1] = sapdevice->pVel_j[devAddr].y;
        pvel[2] = sapdevice->pVel_j[devAddr].z;
        eps2   = sapdevice->pVel_j[devAddr].w;

        vel[0] = sapdevice->vel_j[devAddr].x;
        vel[1] = sapdevice->vel_j[devAddr].y;
        vel[2] = sapdevice->vel_j[devAddr].z;

        acc[0] = sapdevice->acc_j[devAddr].x;
        acc[1] = sapdevice->acc_j[devAddr].y;
        acc[2] = sapdevice->acc_j[devAddr].z;

        jrk[0] = sapdevice->jrk_j[devAddr].x;
        jrk[1] = sapdevice->jrk_j[devAddr].y;
        jrk[2] = sapdevice->jrk_j[devAddr].z;

        id     = sapdevice->id_j[devAddr];
      }

      if(integrationOrder > FOURTH)
      {
        pacc[0] = sapdevice->pAcc_j[devAddr].x;
        pacc[1] = sapdevice->pAcc_j[devAddr].y;
        pacc[2] = sapdevice->pAcc_j[devAddr].z;
      }
    } //omp_get_thread_num() == dev
  } // pragma omp parallel


  //Indicate that we copied the predicted-j particles back to the host
  predJOnHost = true;

  #ifdef DEBUG_PRINT
    fprintf(stderr, "GetjState %d\t%lf\tpos: %g %g %g\tvel: %g %g %g\tacc: %g %g %g\n",
            addr,
            id,
            pos[0],pos[1],pos[2],
            vel[0],vel[1],vel[2],
            acc[0],acc[1],acc[2]);
  #endif

}

int sapporo::fetch_ngb_list_from_device() {

 #ifdef DEBUG_PRINT
  cerr << "fetch_ngb_list_from_device\n";
 #endif


  //Copy only the active ni particles  
  int ni = sapdevice->dev_ni;
  sapdevice->ngb_count_i.d2h(ni);
  sapdevice->ngb_list_i.d2h(ni*NGB_PB);
 
  return ni;
}

void sapporo::forcePrediction(int nj)
{

  #pragma omp parallel num_threads(numberOfGPUUsedBySapporo)
  {
    if (nj_updated)
    {
      //Get the number of particles set for this device
      int particleOnDev = jCopyInformation[omp_get_thread_num()].count;
      if(particleOnDev > 0)
      {
        send_j_particles_to_device(particleOnDev);
      }
    } //nj_updated

    //Calculate the number of nj particles that are used per device
    int temp = nj / nCUDAdevices;
    if(omp_get_thread_num() < (nj  % nCUDAdevices))
      temp++;

    copyJInDev(temp);
    predictJParticles(temp);

  }//end parallel

  nj_modified   = -1;
  predict       = false;
  nj_updated    = false;

  //Clear the address to dev/location mapping
  mappingFromIndexToDevIndex.clear();
}

/*
*
* Functions to start the GPU Kernels
* Functions; Reorder particles on the device, predict, evaluate
*
*/

void sapporo::copyJInDev(int nj)
{
  #ifdef DEBUG_PRINT
    cerr << "copyJInDev nj: " << nj << endl;
  #endif
  nj = nj;
  //This function is called inside an omp parallel section

  //If there are particles updated, put them in the correct locations
  //in the device memory. From the temp buffers to the final location.
  if(jCopyInformation[omp_get_thread_num()].toCopy > 0)
  {
    //Set arguments
    int njToCopy = jCopyInformation[omp_get_thread_num()].toCopy;
    jCopyInformation[omp_get_thread_num()].toCopy = 0;

    int argIdx = 0;
    sapdevice->copyJParticles.set_arg<int  >(argIdx++, &njToCopy);
    sapdevice->copyJParticles.set_arg<int  >(argIdx++, &integrationOrder);
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pos_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pos_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->address_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->t_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pPos_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pVel_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->vel_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->acc_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->jrk_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->id_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->t_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->vel_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->acc_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->jrk_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->id_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pAcc_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->snp_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->crk_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->snp_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->crk_j_temp.ptr());

    //Set execution configuration and start the kernel
    sapdevice->copyJParticles.setWork_2D(128, njToCopy);
    sapdevice->copyJParticles.execute();
  }
}

void sapporo::predictJParticles(int nj)
{
  //This function is called inside an omp parallel section
  #ifdef DEBUG_PRINT
    cerr << "predictJParticles nj: " << nj << endl;
  #endif  

  if(integrationOrder == GRAPE5)
    return; //GRAPE 5 has no prediction

  //if we have to predict call predict Kernel, we do not have to predict if we 
  //already had a gravity evaluation call before that called the predict kernel
  if(predict)
  {
    int argIdx = 0;
    //Set arguments
    sapdevice->predictKernel.set_arg<int   >(argIdx++, &nj);
    sapdevice->predictKernel.set_arg<double>(argIdx++, &t_i);
    sapdevice->predictKernel.set_arg<int   >(argIdx++, &integrationOrder);
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->t_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->pPos_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->pos_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->pVel_j.ptr());    
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->vel_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->acc_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->jrk_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->pAcc_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->snp_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(argIdx++, sapdevice->crk_j.ptr());

    //Set execution config and execute
    sapdevice->predictKernel.setWork_2D(128, nj);
    sapdevice->predictKernel.execute();
  }
}


void sapporo::predictJParticles_host(int nj)
{

  if(integrationOrder == GRAPE5)
  {
    //GRAPE5 has no prediction
    memcpy(&sapdevice->pPos_j[0], &sapdevice->pos_j[0], sizeof(double4) * nj);
    return;
  }

  
  
  double4 snp = make_double4(0,0,0,0);
  double4 crk = make_double4(0,0,0,0);
  
  for(int i=0; i < nj; i++)
  {
    double dt  = t_i  - sapdevice->t_j[i].x;
    double dt2 = (1./2.)*dt;
    double dt3 = (1./3.)*dt;
    double dt4 = (1./4.)*dt;
    double dt5 = (1./5.)*dt;

    double4  pos         = sapdevice->pos_j[i];
    double4  vel         = sapdevice->vel_j[i];
    double4  acc         = sapdevice->acc_j[i];
    double4  jrk         = sapdevice->jrk_j[i];
    
    if(integrationOrder > FOURTH)
    {
      snp         = sapdevice->snp_j[i];
      crk         = sapdevice->crk_j[i];
    }

    //Positions
    pos.x += dt  * (vel.x +  dt2 * (acc.x + dt3 * (jrk.x + 
             dt4 * (snp.x +  dt5 * (crk.x)))));
    pos.y += dt  * (vel.y +  dt2 * (acc.y + dt3 * (jrk.y + 
             dt4 * (snp.y +  dt5 * (crk.y)))));
    pos.z += dt  * (vel.z +  dt2 * (acc.z + dt3 * (jrk.z + 
             dt4 * (snp.z +  dt5 * (crk.z)))));
    sapdevice->pPos_j[i] = pos;
    

    //Velocities
    vel.x += dt  * (acc.x + dt2  * (jrk.x + 
             dt3 * (snp.x +  dt4 * (crk.x))));
    vel.y += dt  * (acc.y + dt2  * (jrk.y + 
             dt3 * (snp.y +  dt4 * (crk.y))));
    vel.z += dt  * (acc.z + dt2  * (jrk.z + 
             dt3 * (snp.z +  dt4 * (crk.z))));
    sapdevice->pVel_j[i] = vel;


    if(integrationOrder > FOURTH)
    {
      //Accelerations
      acc.x += dt * (jrk.x + dt2 * (snp.x +  dt3 * (crk.x)));
      acc.y += dt * (jrk.y + dt2 * (snp.y +  dt3 * (crk.y)));
      acc.z += dt * (jrk.z + dt2 * (snp.z +  dt3 * (crk.z)));
      sapdevice->pAcc_j[i] = acc;
    }
  }//for i  
  
}
void sapporo::evaluate_gravity_host_vector(int ni_total, int nj)
{
#ifdef CPU_SUPPORT
  executedOnHost = true;

  forces_jb(
    nj,    
    &sapdevice->pPos_j[0],
    &sapdevice->pVel_j[0],
    &sapdevice->id_j  [0],
    ni_total,
    &sapdevice->pos_i[0],
    &sapdevice->vel_i[0],
    &sapdevice->id_i [0],
    &sapdevice->iParticleResults[0],
    &sapdevice->iParticleResults[ni_total],
    &sapdevice->ds_i[0],
    EPS2);
#endif
}

void sapporo::evaluate_gravity_host(int ni_total, int nj)
{
  executedOnHost = true;
  #pragma omp for
  for(int i=0; i < ni_total; i++)
  {
    double4 pos_i = sapdevice->pos_i[i];
    double4 vel_i = sapdevice->vel_i[i];
    int      id_i = sapdevice->id_i[i];
    double4 acc_i = make_double4(0,0,0,0);
    double4 jrk_i = make_double4(0,0,0,0);    
    
    double ds_min = 10e10;
    int    nnb    = -1;
    
    for(int j=0; j < nj; j++)
    {
      double4 pos_j = sapdevice->pPos_j[j];
      double4 vel_j = sapdevice->pVel_j[j];
      int      id_j = sapdevice->id_j  [j];
      
      if(id_i == id_j)
        continue;       //Skip self-gravity
      
      //Compute the force
      const double4 dr = make_double4(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z, 0);
      const double ds2 = ((dr.x*dr.x + (dr.y*dr.y)) + dr.z*dr.z);
      
      if(ds2 < ds_min) //keep track of nearest neighbour
      {
        ds_min = ds2;
        nnb    = id_j;
      }
      
      
      const double inv_ds = 1.0/sqrt(ds2+EPS2);

      const double mass   = pos_j.w;
      const double minvr1 = mass*inv_ds; 
      const double  invr2 = inv_ds*inv_ds; 
      const double minvr3 = minvr1*invr2;

      // Acceleration
      acc_i.x += minvr3 * dr.x;
      acc_i.y += minvr3 * dr.y;
      acc_i.z += minvr3 * dr.z;
      acc_i.w += (-1.0)*minvr1;

      //Jerk
      const double4 dv = make_double4(vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z -  vel_i.z, 0);
      const double drdv = (-3.0) * (minvr3*invr2) * (dr.x*dv.x + dr.y*dv.y + dr.z*dv.z);

      jrk_i.x += minvr3 * dv.x + drdv * dr.x;  
      jrk_i.y += minvr3 * dv.y + drdv * dr.y;
      jrk_i.z += minvr3 * dv.z + drdv * dr.z;   
    }//for j
    
    
    sapdevice->ds_i[i].x =    nnb;
    sapdevice->ds_i[i].y = ds_min;
    
    sapdevice->iParticleResults[i         ] = acc_i;
    sapdevice->iParticleResults[i+ni_total] = jrk_i;
    
  }//for i
}



double sapporo::evaluate_gravity(int ni_total, int nj)
{
  //This function is called inside an omp parallel section  
  #ifdef DEBUG_PRINT
    cerr << "evaluate_gravity ni: " << ni_total << "\tnj: " << nj << endl;
  #endif

  if(ni_total == 0 || nj == 0)  return 0.0;


  //Use this to indicate we did gravity on the host, to disable memory copies
  executedOnHost = false; 
  
  #ifdef CPU_SUPPORT
    //Compute number of interactions to be done and compare to CPU threshold
    long long int nInter = ni_total* (long long int) nj;    
    if (nInter < CPUThreshold)
    {
        fprintf(stderr, "CPU EXEC || ni: %d  nj: %d BThreshold: %d nInter: %lld\n", ni_total, nj, CPUThreshold, nInter);
        
        predictJParticles_host(nj);                    //Predict the particles
        //evaluate_gravity_host(ni_total, nj);         //Non-vector version
        evaluate_gravity_host_vector(ni_total, nj);    //Vector version
        executedOnHost = true;
        return 0.0;
    }    
  #endif
    
  //ni is the number of i-particles that is set and for which we compute the force
  //nj is the current number of j-particles that are used as sources

  //If there are particles updated, put them in the correct locations
  //in the device memory. From the temp buffers to the final location.
  copyJInDev(nj);

  //Execute prediction if necessary
  predictJParticles(nj);
  
  
    
  //Reset the memory buffers in order to be able to do atomicAdds
  int argIdx          = 0;
  int doNGB          = true;
  int doNGBList      = true;
  
  sapdevice->resetDevBuffers.set_arg<int  >(argIdx++, &ni_total);
  sapdevice->resetDevBuffers.set_arg<int  >(argIdx++, &doNGB);
  sapdevice->resetDevBuffers.set_arg<int  >(argIdx++, &doNGBList);
  sapdevice->resetDevBuffers.set_arg<int  >(argIdx++, &integrationOrder);
  sapdevice->resetDevBuffers.set_arg<void*>(argIdx++, sapdevice->iParticleResults.ptr());
  sapdevice->resetDevBuffers.set_arg<void*>(argIdx++, sapdevice->ds_i.ptr());
  sapdevice->resetDevBuffers.set_arg<void*>(argIdx++, sapdevice->ngb_count_i.ptr());
  sapdevice->resetDevBuffers.setWork_2D(256, ni_total);
  sapdevice->resetDevBuffers.execute();

  int ni = 0;
  //Loop over the ni-particles in jumps equal to the number of threads
  for(int ni_offset = 0; ni_offset < ni_total; ni_offset += NTHREADS)
  {
    //Determine number of particles to be integrated
    ni = min(ni_total - ni_offset, NTHREADS);
    
    //Setting the properties for the gravity kernel

    //Calculate the number of blocks, groups, etc. For efficiency we always 
    //launch a multiple number of blocks of the warpsize/wavefront size
    int multipleSize = 1;
    
    if(integrationOrder <= FOURTH)
    {
      //This is only possible for 4th order, the 6th order requires too many 
      //resources to launch big thread-blocks. The get_workGroupMultiple 
      //retrieves the warp/wavefront size
      if(ni > 128)
        multipleSize = sapdevice->evalgravKernelTemplate.get_workGroupMultiple();
      else if(ni > 96)
        multipleSize =  sapdevice->evalgravKernelTemplate.get_workGroupMultiple() / 2;
    }
    
    //Force ni to be a multiple of the warp/wavefront size. Note we can let ni be a multiple
    //since we ignore all results of non-used (non-requested) particles
#if 1  //Disable this for block timing. BLOCK_TIMING
    int temp = ni / multipleSize; 
    if((ni % multipleSize) != 0 ) temp++;
    ni = temp * multipleSize;
#endif

    //Dimensions of one thread-block, this can be of the 2D form if there are multiple 
    //y dimensions (q) with an x-dimension of p.
    int p = ni;
    int q = min(NTHREADS2/ni, 32); //NOTE NTHREADS2 to make 2D blocks for devices with enough resources
                                   //to allow for 2x NTHREADS block-sizes
    
    //The above is the default and works all the time, we can do some extra device/algorithm
    //specific tunings using the code below.

    //Set the amount of shared-memory and possibly improve the 2D block sizes 
    //by using specific optimizations. 
    //NOTE this is also device/resource dependend and can cause 'out of resource' crashes!!
    if(integrationOrder == FOURTH)
    {
      if(integrationPrecision == DOUBLESINGLE)
      {
        #ifdef ENABLE_THREAD_BLOCK_SIZE_OPTIMIZATION
          //This is most optimal one for Fourth order Double-Single. 
          if(ni <= 256 && ni >= 32)   
            q = min(sapdevice->evalgravKernelTemplate.get_workGroupMaxSize()/ni, 32);      
        #endif            
      }
    }
    
//     q = 1; //Use this when testing optimal thread/block/multi size. Disables 2D thread-blocks . BLOCK_TIMING 
    
    int sharedMemSizeEval = p*q*(sapdevice->sharedMemPerThread);
    


    //Compute the number of nj particles used per-block (note can have multiple blocks per thread-block in 2D case)
    int nj_scaled       = n_norm(nj, q*(sapdevice->get_NBLOCKS()));
    int thisBlockScaled = nj_scaled/((sapdevice->get_NBLOCKS())*q);

    #ifdef DEBUG_PRINT
      fprintf(stderr, "Offset: %d  --> Total: %d Current step: %d \n", ni_offset, ni_total, ni);
      fprintf(stderr, "EvalGrav config: p: %d q: %d  nj: %d nj_scaled: %d thisblockscaled: %d ni: %d EPS: %f \n",
                      p,q,nj, nj_scaled, thisBlockScaled, ni, EPS2);
      fprintf(stderr, "Shared memory configuration, size eval: %d  Per thread: %d (bytes)\n",
                      sharedMemSizeEval, sapdevice->sharedMemPerThread);    
    #endif


    argIdx = 0;
    
    sapdevice->evalgravKernelTemplate.set_arg<int  >(argIdx++, &nj);      //Total number of j particles
    sapdevice->evalgravKernelTemplate.set_arg<int  >(argIdx++, &thisBlockScaled);
    sapdevice->evalgravKernelTemplate.set_arg<int  >(argIdx++, &ni_offset);    
    sapdevice->evalgravKernelTemplate.set_arg<int  >(argIdx++, &ni_total);   

    sapdevice->evalgravKernelTemplate.set_arg<void* >(argIdx++,  sapdevice->pPos_j.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void* >(argIdx++,  sapdevice->pos_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void* >(argIdx++,  sapdevice->iParticleResults.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<double>(argIdx++, &EPS2);

    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->pVel_j.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->id_j.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->vel_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->id_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->ds_i.ptr());     
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->ngb_count_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->ngb_list_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->acc_i.ptr());
    sapdevice->evalgravKernelTemplate.set_arg<void*>(argIdx++,  sapdevice->pAcc_j.ptr());

    sapdevice->evalgravKernelTemplate.set_arg<int>(argIdx++, NULL, (sharedMemSizeEval)/sizeof(int));  //Shared memory

    sapdevice->evalgravKernelTemplate.setWork_threadblock2D(p, q, (sapdevice->get_NBLOCKS()), 1); //Default
    sapdevice->evalgravKernelTemplate.execute();
  } //Loop over ni


  return 0.0;
} //end evaluate gravity

