#include "sapporohostclass.h"

#include <sys/time.h>
#include <algorithm>

/*


Combined variables:

pos_i.w = h2    --> Used as neighbourhood sphere radius
vel_i.w = eps2
vel_j.w = eps2

*/

int remapList[16384];

inline int n_norm(int n, int j) {
  n = ((n-1)/j) * j + j;
  if (n == 0) n = j;
  return n;
}


double get_time() {
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

#ifdef DEBUG_PRINT
  static int callCount = 0;
#endif

// int sapporo::open(std::string kernelFile, int *devices, 
//                   int nprocs = 1, int order = FOURTH,
//                   int precision = DEFAULT)
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

    if(nprocs == 0) //
    {
      dev = tid;
    }
    if(nprocs > 0)
    {
      //The user gave us a set of device ids
      dev = devices[tid];
    }

    //Assign the device and load the kernels
    sapdevice->assignDevice(dev, integrationOrder);

    sapdevice->loadComputeKernels(kernelFile.c_str());

    if(tid == 0)
    {
      nCUDAdevices = omp_get_num_threads();
    }

    //Set the number of blocks used

  }//end pragma omp parallel

  //Used to store direct pointers to the memory of various threads
  //that handle the device communication
  jMemAddresses.resize(nCUDAdevices);

  //Allocate the memory that depends on the number of devices
  acc_i.resize(n_pipes*nCUDAdevices);
  jrk_i.resize(n_pipes*nCUDAdevices);
  ds_i.resize(n_pipes*nCUDAdevices);
  id_i.resize(n_pipes*nCUDAdevices);
  accin_i.resize(n_pipes*nCUDAdevices);

  ngb_list_i.resize(n_pipes*NGB_PP*nCUDAdevices);

  //Allocate memory for arrays used with 6th order
  if(integrationOrder > FOURTH)
  {
    snp_i.resize(n_pipes*nCUDAdevices);
    crk_i.resize(n_pipes*nCUDAdevices);

    accin_i.clear();
    snp_i.clear();
    crk_i.clear();
  }

  if(integrationOrder > SIXTH)
  {
      //8th order memory
  }


  #ifdef REMAP
  //Make out magic remap list
  for(int i=0; i < 16384; i++)
  {
//     remapList[i] = 999-i;
    remapList[i] = 16383-i;
  }
  #endif
  
  return 0;
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

  address_j.clear();
  pos_j.clear();

  if(integrationOrder > GRAPE5)
  {
    t_j.clear();
    vel_j.clear();
    acc_j.clear();
    jrk_j.clear();
    id_j.clear();
  }

  if(integrationOrder > FOURTH)
  {
    snp_j.clear();
    crk_j.clear();
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
    

  #ifdef REMAP
    //Put the address on a random other location
    address = remapList[address];
  #endif

  predJOnHost = false; //Reset the buffers on the device since they can be modified
  nj_updated = true;  //There are particles that are updated
  int storeLoc = -1;    //-1 if isFirstSend, otherwise it will give direct location in memory

  double4 temp4; 
  double2 temp2;
  
  if(isFirstSend)
  {

    //Store particles in temporary vectors
    temp4.x = x[0]; temp4.y = x[1]; temp4.z = x[2]; temp4.w = mass;
    pos_j.push_back(temp4);
    address_j.push_back(address);

    if(integrationOrder > GRAPE5)
    {
      temp2.x = tj; temp2.y = dtj;
      t_j.push_back(temp2);
      temp4.x = v[0]; temp4.y = v[1]; temp4.z = v[2]; temp4.w = eps;
      vel_j.push_back(temp4); //Store eps in vel.w
      temp4.x = a2[0]; temp4.y = a2[1]; temp4.z = a2[2]; temp4.w = 0.0;
      acc_j.push_back(temp4 );
      temp4.x = j6[0]; temp4.y = j6[1]; temp4.z = j6[2]; temp4.w = 0.0;
      jrk_j.push_back(temp4);
      id_j.push_back ( id);
    }
      
    //For 6th and 8 order we need more parameters
    if(integrationOrder > FOURTH)
    {
      temp4.x = snp[0]; temp4.y = snp[1]; temp4.z = snp[2]; temp4.w = 0.0;
      snp_j.push_back(temp4);
      temp4.x = crk[0]; temp4.y = crk[1]; temp4.z = crk[2]; temp4.w = 0.0;
      crk_j.push_back(temp4);
    }

    nj_modified = pos_j.size();
  }//if is firstsend
  else
  {
    //Check if the address does not fall outside the allocated memory range
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
    //and on which physical address on that device
    int dev           = address % nCUDAdevices;
    int devAddr       = address / nCUDAdevices;
    storeLoc          = jMemAddresses[dev].count;

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
      //New, not set before particle, save address info and increase particles
      //on that specific device by one
      int4 tempI4;
      tempI4.x = dev; tempI4.y = storeLoc; tempI4.z = devAddr; tempI4.w = -1; 
      mappingFromIndexToDevIndex[address] = tempI4;
      jMemAddresses[dev].count++;
    }

    temp4.x = x[0]; temp4.y = x[1]; temp4.z = x[2]; temp4.w = mass;
    jMemAddresses[dev].pos_j[storeLoc]        = temp4;
    jMemAddresses[dev].address[storeLoc]      = devAddr;

    if(integrationOrder > GRAPE5)
    {
      temp2.x = tj; temp2.y = dtj;            
      jMemAddresses[dev].t_j[storeLoc]          = temp2;
      temp4.x = v[0]; temp4.y = v[1]; temp4.z = v[2]; temp4.w = eps;      
      jMemAddresses[dev].vel_j[storeLoc]        = temp4;  //Store eps in vel.w
      temp4.x = a2[0]; temp4.y = a2[1]; temp4.z = a2[2]; temp4.w = 0.0;            
      jMemAddresses[dev].acc_j[storeLoc]        = temp4;
      temp4.x = j6[0]; temp4.y = j6[1]; temp4.z = j6[2]; temp4.w = 0.0;
      jMemAddresses[dev].jrk_j[storeLoc]        = temp4;
      jMemAddresses[dev].id_j[storeLoc]         = id;
    }

    //For 6th and 8 order we need more parameters
    if(integrationOrder > FOURTH)
    {
      temp4.x = snp[0]; temp4.y = snp[1]; temp4.z = snp[2]; temp4.w = 0.0;      
      jMemAddresses[dev].snp_j[storeLoc]        = temp4;
      temp4.x = crk[0]; temp4.y = crk[1]; temp4.z = crk[2]; temp4.w = 0.0;      
      jMemAddresses[dev].crk_j[storeLoc]        = temp4;
    }
  }

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

  int curJMax = nj_max;

  //Increase by 10 %
  int temp = curJMax * 1.1;

  //Minimum of 16k
  if (temp < 16384)
    temp = 16384;


  int temp2 = temp / nCUDAdevices;
  temp2++;
  temp2 = temp2 * nCUDAdevices; //Total number of particles

  nj_max = temp2;       //If address goes over nj_max we realloc


  #pragma omp parallel
  {
    //Number of particles on this device:
    int nj_local = nj_modified / nCUDAdevices;

    if(omp_get_thread_num() < (nj_modified  % nCUDAdevices))
      nj_local++;


    sapdevice->nj_local = nj_local;

    int nj_max_local = nj_max / nCUDAdevices;


    cerr << "Before realloc : " << nj_max_local << "\tparticles" << std::endl; //TODO
    sapdevice->reallocJParticles(nj_max_local);

    cerr << "Allocated memory for : " << nj_max_local << "\tparticles" << std::endl; //TODO

    //Store the memory pointers for direct indexing, only change pointers
    //keep counters
    memPointerJstruct tempStruct         = jMemAddresses[omp_get_thread_num()];

    tempStruct.address = &sapdevice->address_j[0];
    tempStruct.pos_j   = &sapdevice->pos_j_temp[0];


    if(integrationOrder > GRAPE5)
    {
      tempStruct.t_j     = &sapdevice->t_j_temp[0];
      tempStruct.vel_j   = &sapdevice->vel_j_temp[0];
      tempStruct.acc_j   = &sapdevice->acc_j_temp[0];
      tempStruct.jrk_j   = &sapdevice->jrk_j_temp[0];
      tempStruct.id_j    = &sapdevice->id_j_temp[0];
    }

    if(integrationOrder > FOURTH)
    {
      tempStruct.snp_j   = &sapdevice->snp_j_temp[0];
      tempStruct.crk_j   = &sapdevice->crk_j_temp[0];
    }

    jMemAddresses[omp_get_thread_num()] = tempStruct;
  } //end parallel section
}

void sapporo::initialize_firstsend()
{
  #ifdef DEBUG_PRINT
    cerr << "initialize_firstsend nj_modified: " << nj_modified << "\n";
  #endif

  isFirstSend = false;

  nj_total = nj_modified;

  //Make nj max a multiple of the number of devices
  //and increase its size by 10% for a small
  //extra buffer incase extra particle are added
  int temp  = nj_total * 1.1;
  int temp2 = temp / nCUDAdevices;
  temp2++;
  temp2     = temp2 * nCUDAdevices; //Total number of particles

  nj_max = temp2;       //If address goes over nj_max we realloc

  #pragma omp parallel
  {
    //Number of particles on this device:
    int nj_local = nj_modified / nCUDAdevices;

    if(omp_get_thread_num() < (nj_modified  % nCUDAdevices))
      nj_local++;

    sapdevice->nj_local = nj_local;

    int nj_max_local = nj_max / nCUDAdevices;
    sapdevice->allocateMemory(nj_max_local, get_n_pipes());


    //Store the memory pointers for direct indexing
    memPointerJstruct tempStruct;
    tempStruct.pos_j   = &sapdevice->pos_j_temp[0];
    tempStruct.address = &sapdevice->address_j[0];

    if(integrationOrder > GRAPE5)
    {
      tempStruct.t_j     = &sapdevice->t_j_temp[0];
      tempStruct.vel_j   = &sapdevice->vel_j_temp[0];
      tempStruct.acc_j   = &sapdevice->acc_j_temp[0];
      tempStruct.jrk_j   = &sapdevice->jrk_j_temp[0];
      tempStruct.id_j    = &sapdevice->id_j_temp[0];
    }

    if(integrationOrder > FOURTH)
    {
      tempStruct.snp_j   = &sapdevice->snp_j_temp[0];
      tempStruct.crk_j   = &sapdevice->crk_j_temp[0];
    }


    tempStruct.count   = 0;   //Number of particles currently stored
    tempStruct.toCopy  = 0;   //Number of particles to be copied on the device it self
    jMemAddresses[omp_get_thread_num()] = tempStruct;
  } //end parallel section

  //Set & send the initial particles using the set j particle function
  for(int i=0 ; i < nj_total; i++)
  {
    double k18[3] = {0,0,0};
    double a2[3], v[3], x[3], j6[3], snp[3], crk[3];

    x[0] = pos_j[i].x; x[1] = pos_j[i].y; x[2] = pos_j[i].z;

    if(integrationOrder > GRAPE5)
    {
      a2[0] = acc_j[i].x; a2[1] = acc_j[i].y; a2[2] = acc_j[i].z;
      j6[0] = jrk_j[i].x; j6[1] = jrk_j[i].y; j6[2] = jrk_j[i].z;
      v[0] = vel_j[i].x; v[1] = vel_j[i].y; v[2] = vel_j[i].z;
    }

    if(integrationOrder > FOURTH)
    {
      snp[0] = snp_j[i].x; snp[1] = snp_j[i].y; snp[2] = snp_j[i].z;
      crk[0] = crk_j[i].x; crk[1] = crk_j[i].y; crk[2] = crk_j[i].z;
    }

    if(integrationOrder == GRAPE5)
    {
      //GRAPE5 has less properties so only use the properties that are set
      this->set_j_particle(address_j[i], 0, 0, 0, pos_j[i].w,
                         k18, j6, a2, v,x, snp, crk, 0);

    }
    else
    {
      this->set_j_particle(address_j[i], id_j[i], t_j[i].x, t_j[i].y, pos_j[i].w,
                         k18, j6, a2, v,x, snp, crk, vel_j[i].w);
    }
  }

 
  //Clear the temprorary vectors, they are not used anymore from now on
  address_j.clear();
  t_j.clear();
  pos_j.clear();
  vel_j.clear();
  acc_j.clear();
  jrk_j.clear();
  id_j.clear();

  if(integrationOrder > FOURTH)
  {
    snp_j.clear();
    crk_j.clear();
  }

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

  //If this is the first send, then we have to allocate memory
  //distribute the particles etc.
  if(isFirstSend)
  {
    initialize_firstsend();
  }

  double4 temp4;
  //Copy i-particles to device structures, first only to device 0
  //then from device 0 we use memcpy to get the data into the 
  //other devs buffers
  int toDevice = 0;
  
  for (int i = 0; i < ni; i++)
  {
    temp4.x  = xi[i][0]; temp4.y = xi[i][1]; temp4.z = xi[i][2]; temp4.w = h2[i];
    deviceList[toDevice]->pos_i[i] = temp4;
    pos_i[i] = temp4;


    if(integrationOrder > GRAPE5)
    {
      temp4.x  = vi[i][0]; temp4.y = vi[i][1]; temp4.z = vi[i][2]; temp4.w = eps2;   
      
      if(eps2_i != NULL)  //Seperate softening for i-particles
        temp4.w = eps2_i[i];        
      
      deviceList[toDevice]->id_i[i]  = id[i];        
      deviceList[toDevice]->vel_i[i] = temp4;
      
      vel_i[i] = temp4;
      id_i[i]  = id[i];      
    }


    if(integrationOrder > FOURTH)
    {
      temp4.x    = a[i][0]; temp4.y = a[i][1]; temp4.z = a[i][2]; temp4.w = 0;      
      deviceList[toDevice]->accin_i[i] = temp4;
      
      accin_i[i] = temp4;
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

  //Copy i particles to the other host side device buffers  
  for(int i = toDevice+1;  i < nCUDAdevices; i++)
  {
      memcpy(&sapdevice->pos_i[i], &sapdevice->pos_i[toDevice], sizeof(double4) * ni);
      if(integrationOrder > GRAPE5)
      {
        memcpy(&sapdevice->vel_i[0], &vel_i[0], sizeof(double4) * ni);
        memcpy(&sapdevice->id_i[0],  &id_i[0],  sizeof(int)     * ni);  
        
        if(integrationOrder > FOURTH)
          memcpy(&sapdevice->accin_i[0], &accin_i[0], sizeof(double4) * ni);        
      }
  }
  


  #pragma omp parallel
  {
    if (nj_updated) {
      //Get the number of particles set for this device
      int devCount = jMemAddresses[omp_get_thread_num()].count;
      if(devCount > 0)
      {
        send_j_particles_to_device(devCount);
      }
    }

    //ni is the number of particles in the pipes
    send_i_particles_to_device(ni);

    //nj is the total number of particles to which the i particles have to
    //be calculated. For direct n-body this is usually equal to the total
    //number of nj particles that have been set by the calling code


    //Calculate the number of nj particles that are used per device
    int thisDevNj;
    int temp = nj / nCUDAdevices;
    if(omp_get_thread_num() < (nj  % nCUDAdevices))
      temp++;

    thisDevNj = temp;
    sapdevice->nj_local = temp;

    evaluate_gravity(ni, thisDevNj);

    sapdevice->dev_ni = ni;
  }//end parallel section

  //TODO sync?
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
    fprintf(stderr, "calc_lasthalf2 device= %d, ni= %d nj = %d callCount: %d\n", -1, ni, nj, callCount++);
  #endif
    
  //Prevent unused compiler warning    
  nj = nj; index = index; xi = xi; vi = vi; eps2 = eps2; h2 = h2;
  
    
//double t0 = get_time();
  double ds_min[NTHREADS];
  for (int i = 0; i < ni; i++) {
    pot[i] = 0;
    acc[i][0]  = acc[i][1]  = acc[i][2]  = 0;
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
//double t1 = get_time();
  #pragma omp parallel
  {
    //Retrieve data from the devices (in parallel)
    retrieve_i_particle_results(ni);
  }
//double t2 = get_time();
  //Reduce the data from the different devices into one final results
  for (int dev = 0; dev < nCUDAdevices; dev++) {
    for (int i = 0; i < ni; i++) {
      double4 acci = acc_i[i + dev*n_pipes];

      pot[i]    += acci.w;
      acc[i][0] += acci.x;
      acc[i][1] += acci.y;
      acc[i][2] += acci.z;

      //fprintf(stdout, "device= %d, ni= %d pot = %g\n", dev, i, acci.x);
      //fprintf(stderr, "Outdevice= %d,\t%d\t%f\t%f\t%f\n", dev,i, acci.x, acci.y, acci.z);

      if(integrationOrder > GRAPE5)
      {
        double4 jrki = jrk_i[i + dev*n_pipes];
        double  ds   = ds_i[i  + dev*n_pipes];

        jerk[i][0] += jrki.x;
        jerk[i][1] += jrki.y;
        jerk[i][2] += jrki.z;

        if(ngb)   //If we want nearest neighbour
        {
          if (ds < ds_min[i]) {
            int nnb     = (int)(jrki.w);
            nnbindex[i] = nnb;
            ds_min[i]   = ds;
            if(dsmin_i != NULL)
              dsmin_i[i]  = ds;
          }
        }
      } //End if > GRAPE5

      if(integrationOrder > FOURTH)
      {
        double4 snpi = snp_i[i + dev*n_pipes];
        double4 crki = crk_i[i + dev*n_pipes];

        snp[i][0] += snpi.x;
        snp[i][1] += snpi.y;
        snp[i][2] += snpi.z;
        crk[i][0] += crki.x;
        crk[i][1] += crki.y;
        crk[i][2] += crki.z;
      }

//       fprintf(stderr,"%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%d\t%g\n",
//               index[i], pot[i], acc[i][0], acc[i][1], acc[i][2], jerk[i][0], jerk[i][1], jerk[i][2],
//               nnbindex[i], ds_min[i]);
// //

    }
  
  }

  /*
  double t3 = get_time();
  double ta = t1-t0;
  double tb = t3-t2;
  double tc = t2-t1;
  fprintf(stderr, "TOOKA: %g  TOOKB: %g  TOOKC: %g  %g\n", ta, tb, tc, tc*1000000);
*/
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
  int *ni = new int[omp_get_max_threads()];

  #pragma omp parallel
  {
    //Retrieve data from the devices
    ni[omp_get_thread_num()] = fetch_ngb_list_from_device();
  }

   for(int dev = 0; dev < nCUDAdevices; dev++)
   {
    for (int j = 0; j < ni[dev]; j++)
    {
      if (ngb_list_i[dev*NGB_PP*n_pipes + j*NGB_PP] >= NGB_PP) {
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

//
  bool overflow = false;
  nblen         = 0;
  for (int i = 0; i < nCUDAdevices; i++) {
    int offset  = i*NGB_PP*n_pipes + NGB_PP*ipipe;
    int len     = ngb_list_i[offset];
    memcpy(nbl+nblen, &ngb_list_i[offset+1], sizeof(int)*min(len, maxlength - len));
    nblen += len;
    if (nblen >= maxlength) {
      overflow = true;
      break;
    }
  }
  sort(nbl, nbl + min(nblen, maxlength));


//   if (overflow) {
//     fprintf(stderr, "sapporo::get_ngb_list(..) - overflow for ipipe= %d, ngb= %d\n",
//          ipipe, nblen);
//   }

  return overflow;
}

/*

Device communication functions

*/


void sapporo::send_j_particles_to_device(int nj_tosend)
{
  #ifdef DEBUG_PRINT
    cerr << "send_j_particles_to_device nj_tosend: " << nj_tosend << "\tnj_local: "<< sapdevice->nj_local ;
  #endif

  //This function is called inside an omp parallel section

  //Copy the particles to the device memory
  assert(nj_tosend == jMemAddresses[omp_get_thread_num()].count);

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
  jMemAddresses[omp_get_thread_num()].toCopy += jMemAddresses[omp_get_thread_num()].count;
  jMemAddresses[omp_get_thread_num()].count = 0;

  #ifdef DEBUG_PRINT
    cerr << "...send complete\n";
  #endif
} //end send j particles


void sapporo::send_i_particles_to_device(int ni)
{
  #ifdef DEBUG_PRINT
    cerr << "send_i_particles_to_device ni: " << ni << endl;
  #endif

  //This is function is called inside an omp parallel section

  //First we copy the data into the device structures
  //and then to the host...bit double TODO make it better!
  memcpy(&sapdevice->pos_i[0], &pos_i[0], sizeof(double4) * ni);
  sapdevice->pos_i.h2d(ni);

  if(integrationOrder > GRAPE5)
  {
    memcpy(&sapdevice->vel_i[0], &vel_i[0], sizeof(double4) * ni);
    memcpy(&sapdevice->id_i[0],  &id_i[0],  sizeof(int)     * ni);

    sapdevice->vel_i.h2d(ni);
    sapdevice->id_i.h2d(ni);
  }

  if(integrationOrder > FOURTH)
  {
    memcpy(&sapdevice->accin_i[0], &accin_i[0], sizeof(double4) * ni);
    sapdevice->accin_i.h2d(ni);
  }

} //end i particles

void sapporo::retrieve_i_particle_results(int ni)
{
 #ifdef DEBUG_PRINT
  cerr << "retrieve_i_particle_results\n";
 #endif

  //Called inside an OMP parallel section
  
  //TODO make this better as with the other memory copies
  //double t0 = get_time();
  sapdevice->accin_i.d2h(ni);
  //fprintf(stderr, "TOOK D: %g \n", get_time()-t0);
  memcpy(&acc_i[n_pipes*omp_get_thread_num()], &sapdevice->accin_i[0], sizeof(double4) * ni);

  if(integrationOrder > GRAPE5)
  {
    sapdevice->jrk_i.d2h(ni);
    sapdevice->ds_i.d2h(ni);

    memcpy(&jrk_i[n_pipes*omp_get_thread_num()], &sapdevice->jrk_i[0], sizeof(double4) * ni);
    memcpy(&ds_i[n_pipes *omp_get_thread_num()], &sapdevice->ds_i[0],  sizeof(double)  * ni);
  }

  if(integrationOrder > FOURTH)
  {
    sapdevice->snp_i.d2h(ni);
    sapdevice->crk_i.d2h(ni);
    memcpy(&snp_i[n_pipes*omp_get_thread_num()], &sapdevice->snp_i[0], sizeof(double4) * ni);
    memcpy(&crk_i[n_pipes*omp_get_thread_num()], &sapdevice->crk_i[0], sizeof(double4) * ni);
  }
}//retrieve i particles


void sapporo::retrieve_predicted_j_particle(int addr,  double &mass,
                                            double &id, double &eps2,
                                            double pos[3], double vel[3],
                                            double acc[3])
{

  #ifdef DEBUG_PRINT
    cerr << "retrieve_predicted_j_particle address: " << addr << endl;
  #endif

  #ifdef REMAP
    //Put the address on a random other location
    addr = remapList[addr];
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
//     cerr << "retrieve_predicted_j_particle res, from devAddr: " << addr / nCUDAdevices << endl;
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

  #ifdef REMAP
    //Put the address on a random other location
    addr = remapList[addr];
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
//     cerr << "retrieve_predicted_j_particle res, from devAddr: " << addr / nCUDAdevices << endl;
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

  int ni = sapdevice->dev_ni;
  //Copy only the final selection
  sapdevice->ngb_list_i.d2h(ni*NGB_PP, NTHREADS*NGB_PB*(sapdevice->get_NBLOCKS()));

  //Now only copy a small part of it into the other memory
  memcpy(&ngb_list_i[n_pipes*NGB_PP*omp_get_thread_num()], &sapdevice->ngb_list_i[0], sizeof(int) * ni*NGB_PP);

  return ni;
}

void sapporo::forcePrediction(int nj)
{

  if(isFirstSend)
  {
    initialize_firstsend();
  }

  #pragma omp parallel
  {
    if (nj_updated)
    {
      //Get the number of particles set for this device
      int particleOnDev = jMemAddresses[omp_get_thread_num()].count;
      if(particleOnDev > 0)
      {
        send_j_particles_to_device(particleOnDev);
      }
    } //nj_updated

    //Calculate the number of nj particles that are used per device
    int thisDevNj;
    int temp = nj / nCUDAdevices;
    if(omp_get_thread_num() < (nj  % nCUDAdevices))
      temp++;

    thisDevNj = temp;
    sapdevice->nj_local = temp;

    copyJInDev(thisDevNj);
    predictJParticles(thisDevNj);

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
  if(jMemAddresses[omp_get_thread_num()].toCopy > 0)
  {
    //Set arguments
    int njToCopy = jMemAddresses[omp_get_thread_num()].toCopy;
    jMemAddresses[omp_get_thread_num()].toCopy = 0;

    sapdevice->copyJParticles.set_arg<int  >(0, &njToCopy);
    sapdevice->copyJParticles.set_arg<int  >(1, &sapdevice->nj_local);
    sapdevice->copyJParticles.set_arg<void*>(2, sapdevice->pos_j.ptr());
    sapdevice->copyJParticles.set_arg<void*>(3, sapdevice->pos_j_temp.ptr());
    sapdevice->copyJParticles.set_arg<void*>(4, sapdevice->address_j.ptr());

    if(integrationOrder > GRAPE5)
    {
      sapdevice->copyJParticles.set_arg<void*>(5, sapdevice->t_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(6, sapdevice->pPos_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(7, sapdevice->pVel_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(8, sapdevice->vel_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(9, sapdevice->acc_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(10, sapdevice->jrk_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(11, sapdevice->id_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(12, sapdevice->t_j_temp.ptr());
      sapdevice->copyJParticles.set_arg<void*>(13, sapdevice->vel_j_temp.ptr());
      sapdevice->copyJParticles.set_arg<void*>(14, sapdevice->acc_j_temp.ptr());
      sapdevice->copyJParticles.set_arg<void*>(15, sapdevice->jrk_j_temp.ptr());
      sapdevice->copyJParticles.set_arg<void*>(16, sapdevice->id_j_temp.ptr());
    }


    if(integrationOrder > FOURTH)
    {
      sapdevice->copyJParticles.set_arg<void*>(17, sapdevice->pAcc_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(18, sapdevice->snp_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(19, sapdevice->crk_j.ptr());
      sapdevice->copyJParticles.set_arg<void*>(20, sapdevice->snp_j_temp.ptr());
      sapdevice->copyJParticles.set_arg<void*>(21, sapdevice->crk_j_temp.ptr());
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
      sharedMemSizeEval    = p*q*(sizeof(DS4) + sizeof(float4)); //4th order Double Single
      sharedMemSizeReduce  = (sapdevice->get_NBLOCKS())*(2*sizeof(float4) + 3*sizeof(int)); //4th DS  
      
      #ifdef ENABLE_THREAD_BLOCK_SIZE_OPTIMIZATION
        //This is most optimal one for Fourth order Double-Single. 
        if(ni <= 256 && ni >= 32)   
          q = min(sapdevice->evalgravKernel.get_workGroupMaxSize()/ni, 32);      
      #endif
      
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

  sapdevice->reduceForces.setWork_threadblock2D(nthreads, 1, nblocks, 1);

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

  sapdevice->reduceForces.execute();

  return 0.0;
} //end evaluate gravity







