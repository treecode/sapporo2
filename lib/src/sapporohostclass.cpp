#include "sapporohostclass.h"

#include <sys/time.h>
#include <algorithm>

/*


Combined variables:

pos_i.w = h2    --> Used as neighbourhood sphere radius
vel_i.w = eps2
vel_j.w = eps2

*/


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
  cout << "Integration precision used: " << precision << " (0=DEFAULT, 1=DOUBLE)\n";

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

  omp_set_num_threads(numThread);
  #pragma omp parallel
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
    sapdevice->loadComputeKernels(kernelFile.c_str());

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

  return 0;
}

void sapporo::cleanUpDevice()
{
  #pragma omp parallel
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
  #pragma omp parallel
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


  #pragma omp parallel
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
    cerr << "calc_firsthalf ni: " << ni << "\tnj: " << nj << "\tnj_total: " << nj_total << "integrationOrder: "<< integrationOrder << endl;
  #endif
    
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
        deviceList[toDevice]->accin_i[i] = make_double4(a[i][0], a[i][1], a[i][2], 0);
      }      
    }


    #ifdef DEBUG_PRINT
      if(integrationOrder == GRAPE5)
      {
        fprintf(stderr, "Inpdevice= %d,\ti: %d\tindex: %d\teps2: %f\t%f\t%f\t%f",
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
          memcpy(&deviceList[i]->accin_i[0], &deviceList[0]->accin_i[0], sizeof(double4) * ni);        
      }
  }
  


  #pragma omp parallel
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
    
  //Prevent unused compiler warning    
  nj = nj; index = index; xi = xi; vi = vi; eps2 = eps2; h2 = h2;
  
  double ds_min[NTHREADS];
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

  #pragma omp parallel
  {
    //Retrieve data from the devices (in parallel)
    retrieve_i_particle_results(ni);
  }

  //Reduce the data from the different devices into one final results
  for (int dev = 0; dev < nCUDAdevices; dev++) {
    for (int i = 0; i < ni; i++) {

      pot[i]    += deviceList[dev]->accin_i[i].w;
      acc[i][0] += deviceList[dev]->accin_i[i].x;
      acc[i][1] += deviceList[dev]->accin_i[i].y;
      acc[i][2] += deviceList[dev]->accin_i[i].z;

      if(integrationOrder > GRAPE5)
      {
        jerk[i][0] += deviceList[dev]->jrk_i[i].x;
        jerk[i][1] += deviceList[dev]->jrk_i[i].y;
        jerk[i][2] += deviceList[dev]->jrk_i[i].z;
        double  ds  = deviceList[dev]->ds_i[i];        

        if(ngb)   //If we want nearest neighbour
        {
          if (ds < ds_min[i]) {
            int nnb     = (int)(deviceList[dev]->jrk_i[i].w);
            nnbindex[i] = nnb;
            ds_min[i]   = ds;
            if(dsmin_i != NULL)
              dsmin_i[i]  = ds;
          }
        }
      } //End if > GRAPE5

      if(integrationOrder > FOURTH)
      {
        snp[i][0] += deviceList[dev]->snp_i[i].x;
        snp[i][1] += deviceList[dev]->snp_i[i].y;
        snp[i][2] += deviceList[dev]->snp_i[i].z;
        crk[i][0] += deviceList[dev]->crk_i[i].x;
        crk[i][1] += deviceList[dev]->crk_i[i].y;
        crk[i][2] += deviceList[dev]->crk_i[i].z;
      }

//       fprintf(stderr,"%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%d\t%g\n",
//               index[i], pot[i], acc[i][0], acc[i][1], acc[i][2], jerk[i][0], jerk[i][1], jerk[i][2],
//               nnbindex[i], ds_min[i]);

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

  #pragma omp parallel
  {
    //Retrieve data from the devices
    ni[omp_get_thread_num()] = fetch_ngb_list_from_device();
  }

  for(int dev = 0; dev < nCUDAdevices; dev++)
  {
    for (int j = 0; j < ni[dev]; j++)
    {
      if(deviceList[dev]->ngb_list_i[j*NGB_PP]  >= NGB_PP) {
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
  fprintf(stderr, "get_ngb_list\n");
 #endif


  bool overflow = false;
  nblen         = 0;
  for (int i = 0; i < nCUDAdevices; i++) {
    int offset  = NGB_PP*ipipe;
    int len     = deviceList[i]->ngb_list_i[offset];
    
    memcpy(nbl+nblen, &deviceList[i]->ngb_list_i[offset+1], sizeof(int)*min(len, maxlength - len));
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
      sapdevice->accin_i.h2d(ni);
    }
  }

} //end send_i_particles_to_device

void sapporo::retrieve_i_particle_results(int ni)
{
  #ifdef DEBUG_PRINT
    cerr << "retrieve_i_particle_results\n";
  #endif

  //Called inside an OMP parallel section
  
  sapdevice->accin_i.d2h(ni);
  
  if(integrationOrder > GRAPE5)
  {
    sapdevice->jrk_i.d2h(ni);
    sapdevice->ds_i.d2h(ni);

    if(integrationOrder > FOURTH)
    {
      sapdevice->snp_i.d2h(ni);
      sapdevice->crk_i.d2h(ni);
    }
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
    #pragma omp parallel
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
  #pragma omp parallel
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
    #pragma omp parallel
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
  #pragma omp parallel
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
  sapdevice->ngb_list_i.d2h(ni*NGB_PP, NTHREADS*NGB_PB*(sapdevice->get_NBLOCKS()));

  return ni;
}

void sapporo::forcePrediction(int nj)
{

  #pragma omp parallel
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
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pos_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pos_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->address_j.ptr());

    if(integrationOrder > GRAPE5)
    {
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

      if(integrationOrder > FOURTH)
      {
        sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->pAcc_j.ptr());
        sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->snp_j.ptr());
        sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->crk_j.ptr());
        sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->snp_j_temp.ptr());
        sapdevice->copyJParticles.set_arg<void*>(argIdx++, sapdevice->crk_j_temp.ptr());
      }
    }

    //Set execution configuration and start the kernel
    sapdevice->copyJParticles.setWork_2D(128, njToCopy);
    sapdevice->copyJParticles.execute();
  }
}

void sapporo::predictJParticles(int nj)
{
  //This function is called inside an omp parallel section

  if(integrationOrder == GRAPE5)
    return; //GRAPE 5 has no prediction

  //if we have to predict call predict Kernel, we do not have to predict if we 
  //already had a gravity evaluation call before that called the predict kernel
  if(predict)
  {
    //Set arguments
    sapdevice->predictKernel.set_arg<int   >(0, &nj);
    sapdevice->predictKernel.set_arg<double>(1, &t_i);
    sapdevice->predictKernel.set_arg<void* >(2, sapdevice->t_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(3, sapdevice->pPos_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(4, sapdevice->pVel_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(5, sapdevice->pos_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(6, sapdevice->vel_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(7, sapdevice->acc_j.ptr());
    sapdevice->predictKernel.set_arg<void* >(8, sapdevice->jrk_j.ptr());

    if(integrationOrder > FOURTH)
    {
      sapdevice->predictKernel.set_arg<void* >(9, sapdevice->pAcc_j.ptr());
      sapdevice->predictKernel.set_arg<void* >(10, sapdevice->snp_j.ptr());
      sapdevice->predictKernel.set_arg<void* >(11, sapdevice->crk_j.ptr());
    }

    //Set execution config and execute
    sapdevice->predictKernel.setWork_2D(128, nj);
    sapdevice->predictKernel.execute();
  }
}


double sapporo::evaluate_gravity(int ni, int nj)
{
  //This function is called inside an omp parallel section  
  #ifdef DEBUG_PRINT
    cerr << "evaluate_gravity ni: " << ni << "\tnj: " << nj << endl;
  #endif

  //ni is the number of i-particles that is set and for which we compute the force
  //nj is the current number of j-particles that are used as sources

  //If there are particles updated, put them in the correct locations
  //in the device memory. From the temp buffers to the final location.
  copyJInDev(nj);

  //Execute prediction if necessary
  predictJParticles(nj);

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
      multipleSize = sapdevice->evalgravKernel.get_workGroupMultiple();
    else if(ni > 96)
      multipleSize =  sapdevice->evalgravKernel.get_workGroupMultiple() / 2;
  }
  
  //Force ni to be a multiple of the warp/wavefront size. Note we can let ni be a multiple
  //since we ignore all results of non-used (non-requested) particles
  int temp = ni / multipleSize; 
  if((ni % multipleSize) != 0 ) temp++;
  ni = temp * multipleSize;
  

  //Dimensions of one thread-block, this can be of the 2D form if there are multiple 
  //y dimensions (q) with an x-dimension of p.
  int p = ni;
  int q = min(NTHREADS/ni, 32);
  
  //The above is the default and works all the time, we can do some extra device/algorithm
  //specific tunings using the code below.
  

  //Set the amount of shared-memory and possibly improve the 2D block sizes 
  //by using specific optimizations. 
  //NOTE this is also device/resource dependend and can cause 'out of resource' crashes!!
  
  int sharedMemSizeEval   = 0;
  int sharedMemSizeReduce = 0;
  
  if(integrationOrder == GRAPE5)
  {
    if(integrationPrecision == DEFAULT)
    {
      //Single Precision
      sharedMemSizeEval    = p*q*(sizeof(float4)); //G5 Single precision
      sharedMemSizeReduce  = (sapdevice->get_NBLOCKS())*(sizeof(float4)); //G5 Single precision
    }
    if(integrationPrecision == DOUBLE)
    {
      //Double Single Precision      
      sharedMemSizeEval    = p*q*(sizeof(DS4)); //G5 DS precision
      sharedMemSizeReduce  = (sapdevice->get_NBLOCKS())*(sizeof(float4)); //G5 DS precision (acc is SP)            
    }        
  }
  
  if(integrationOrder == FOURTH)
  {
    if(integrationPrecision == DEFAULT)
    {
      #ifdef ENABLE_THREAD_BLOCK_SIZE_OPTIMIZATION
        //This is most optimal one for Fourth order Double-Single. 
        if(ni <= 256 && ni >= 32)   
          q = min(sapdevice->evalgravKernel.get_workGroupMaxSize()/ni, 32);      
      #endif
          
      sharedMemSizeEval    = p*q*(sizeof(DS4) + sizeof(float4)); //4th order Double Single
      sharedMemSizeReduce  = (sapdevice->get_NBLOCKS())*(2*sizeof(float4) + 3*sizeof(int)); //4th DS  
          
    }
    if(integrationPrecision == DOUBLE)
    {
      sharedMemSizeEval   = p*q*(sizeof(double4) + sizeof(double4) + sizeof(int)*2 + sizeof(double));
      sharedMemSizeReduce = sapdevice->get_NBLOCKS()*(2*sizeof(double4) + 2*sizeof(int) + sizeof(double));
    }
  }
  
  if(integrationOrder == SIXTH)
  {
    //Only has a double precision version
    sharedMemSizeEval =   p*q*(sizeof(double4) + sizeof(double4) + sizeof(double4) + sizeof(int)*2 + sizeof(double));
    sharedMemSizeReduce = sapdevice->get_NBLOCKS()*(3*sizeof(double4) + 2*sizeof(int) + sizeof(double));   //6th order
  }
  
  //q = 1; //Use this when testing optimal thread/block/multi size. Disables 2D thread-blocks  
    
  //Compute the number of nj particles used per-block (note can have multiple blocks per thread-block in 2D case)
  int nj_scaled       = n_norm(nj, q*(sapdevice->get_NBLOCKS()));
  int thisBlockScaled = nj_scaled/((sapdevice->get_NBLOCKS())*q);
  int nthreads        = NTHREADS;
  


  #ifdef DEBUG_PRINT
    fprintf(stderr, "EvalGrav config: p: %d q: %d  nj: %d nj_scaled: %d thisblockscaled: %d nthreads: %d ni: %d EPS: %f \n",
                    p,q,nj, nj_scaled, thisBlockScaled, nthreads, ni, EPS2);
    fprintf(stderr, "Shared memory configuration, size eval: %d  \t size reduc: %d \n",
                    sharedMemSizeEval, sharedMemSizeReduce);    
  #endif


  sapdevice->evalgravKernel.set_arg<int  >(0, &nj);      //Total number of j particles
  sapdevice->evalgravKernel.set_arg<int  >(1, &thisBlockScaled);
  sapdevice->evalgravKernel.set_arg<int  >(2, &nthreads);

  if(integrationOrder == GRAPE5)
  {
    sapdevice->evalgravKernel.set_arg<void*>(3, sapdevice->pos_j.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(4, sapdevice->pos_i.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(5, sapdevice->accin_i.ptr());
    sapdevice->evalgravKernel.set_arg<double>(6, &EPS2);
    sapdevice->evalgravKernel.set_arg<int>(7, NULL, (sharedMemSizeEval)/sizeof(int));  //Shared memory
  }
  else
  {
    sapdevice->evalgravKernel.set_arg<void*>(3, sapdevice->pPos_j.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(4, sapdevice->pos_i.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(5, sapdevice->accin_i.ptr());
    sapdevice->evalgravKernel.set_arg<double>(6, &EPS2);

    sapdevice->evalgravKernel.set_arg<void*>(7,  sapdevice->pVel_j.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(8,  sapdevice->id_j.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(9,  sapdevice->vel_i.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(10, sapdevice->jrk_i.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(11, sapdevice->id_i.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(12, sapdevice->ngb_list_i.ptr());
    if(integrationOrder == FOURTH)
      sapdevice->evalgravKernel.set_arg<int>(13, NULL, (sharedMemSizeEval)/sizeof(int));  //Shared memory
  }

  if(integrationOrder > FOURTH)
  {
    sapdevice->evalgravKernel.set_arg<void*>(13, sapdevice->pAcc_j.ptr());
    sapdevice->evalgravKernel.set_arg<void*>(14, sapdevice->snp_i.ptr());
    sapdevice->evalgravKernel.set_arg<int>(15, NULL, (sharedMemSizeEval)/sizeof(int));  //Shared memory
  }

  sapdevice->evalgravKernel.setWork_threadblock2D(p, q, (sapdevice->get_NBLOCKS()), 1); //Default
  sapdevice->evalgravKernel.execute();

  //Kernel reduce
  nthreads        = (sapdevice->get_NBLOCKS());
  int nblocks     = ni;

  int tempNTHREADS  = NTHREADS;
  int tempNGBOffset = NGB_PB*(sapdevice->get_NBLOCKS())*NTHREADS;

  sapdevice->reduceForces.set_arg<void*>(0, sapdevice->accin_i.ptr());
  if(integrationOrder == GRAPE5)
  {
    sapdevice->reduceForces.set_arg<int>(1, NULL, (sharedMemSizeReduce)/sizeof(int));  //Shared memory
  }
  if(integrationOrder > GRAPE5)
  {
    sapdevice->reduceForces.set_arg<void*>(1, sapdevice->jrk_i.ptr());
    sapdevice->reduceForces.set_arg<void*>(2, sapdevice->ds_i.ptr());
    sapdevice->reduceForces.set_arg<void*>(3, sapdevice->vel_i.ptr());
    sapdevice->reduceForces.set_arg<int  >(4, &tempNTHREADS);  //offset_ds
    sapdevice->reduceForces.set_arg<int  >(5, &tempNGBOffset);  //offset
    sapdevice->reduceForces.set_arg<void*>(6, sapdevice->ngb_list_i.ptr());
    if(integrationOrder == FOURTH)
      sapdevice->reduceForces.set_arg<int>(7, NULL, (sharedMemSizeReduce)/sizeof(int));  //Shared memory
  }

  if(integrationOrder > FOURTH)
  {
    sapdevice->reduceForces.set_arg<void*>(7, sapdevice->snp_i.ptr());
    sapdevice->reduceForces.set_arg<int>(8, NULL, (sharedMemSizeReduce)/sizeof(int));  //Shared memory
  }
  sapdevice->reduceForces.setWork_threadblock2D(nthreads, 1, nblocks, 1);
  sapdevice->reduceForces.execute();

  return 0.0;
} //end evaluate gravity







