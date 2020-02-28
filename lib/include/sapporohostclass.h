/****************************
 *                          
 *     libSAPPORO  v2.0     
 *  a multiGPU GRAPE6 and beyond library !
 *        library           
 *                         
 * (c) 2010                 
 *
 * T
 *
 ********************************/

#ifndef _SAPPORO_H_
#define _SAPPORO_H_

#include <iostream>
#include <vector>
#include <map>

#include <math.h>
using namespace std;


#define HUGE 10e10

#include "sapdevclass.h"


struct jCopyInfo
{
  int     count;
  int     toCopy;
} typedef jCopyInfo;



class sapporo {
protected:
   
  int n_pipes;
  int nCUDAdevices;

  int nj_modified;     //Modified number of particles that has to be send to the device  
  int nj_max;          //Total number of allocated memory particles
  int nj_total;        //Total number of j particles in the system (same as nj max?)
 

  bool nj_updated;      //Indicates if there are updated j particles
  bool predJOnHost;     //Indicate if the predicted J-particles are on the host or not

  double EPS2;
  
  map<int, int4> mappingFromIndexToDevIndex;
  //int4.x = device
  //int4.y = arraylocation
  //int4.z = device address

  
  double t_i;
  
  vector<jCopyInfo> jCopyInformation;
  
  
  bool ngb_list_copied;
  bool predict;
  
  void cleanUpDevice();
  void free_cuda_memory(int);
  void allocate_cuda_memory(int);
  void send_j_particles_to_device(int);
  void send_i_particles_to_device(int, int);
  void fetch_data_from_device(int, int );
  void retrieve_i_particle_results(int ni);
  int  fetch_ngb_list_from_device();
  void increase_jMemory(); 
  
  void copyJInDev(int nj);
  void predictJParticles(int nj)  ;

  double evaluate_gravity(int, int);
  
  
  //Host prediction and evaluation functions
  void predictJParticles_host(int nj);
  void evaluate_gravity_host(int ni_total, int nj);
  
  void evaluate_gravity_host_vector(int ni_total, int nj);
  
  
  bool isFirstSend;             //Used to check if this is the first time we sent particles to the
                                //device, so we have to allocate memory
  int integrationOrder;         //Order of the integrator we use, should be set during open call, default is fourth order
  int integrationPrecision;     //The precision of the integrator for shared-memory calculation. Default is DOUBLESINGLE
  
  bool executedOnHost;
  int CPUThreshold;             //The number of interactions from which point on GPU will be faster than CPU
  

public:
  sapporo() {
    n_pipes = NPIPES;

    t_i = 0.0;

    ngb_list_copied = false;
    
    predict     = false;
    isFirstSend = true;
    nj_updated  = false;
    
    
    integrationOrder            = FOURTH;
    integrationPrecision        = DOUBLESINGLE;
  };
  ~sapporo() {
     cleanUpDevice();
  };
  
  
  //Device communication functions
  void send_j_particles_to_device();
  void send_i_particles_to_device(int i);
    
  //Library interface functions
  int open(std::string kernelFile, int *devices, int nprocs = 1, 
           int order = FOURTH, int precision = DOUBLESINGLE);
  int close();
  int get_n_pipes();
  int set_time(double ti);
  int set_no_time();

  int set_j_particle(int address,
                     int index,
                     double tj, double dtj,
                     double mass,
                     double k18[3], double j6[3],
                     double a2[3], double v[3], double x[3],
                     double snp[3], double crk[3], double eps);
                     
  void startGravCalc(int nj, int ni,
                     int index[], 
                     double xi[][3], double vi[][3],
                     double aold[][3], double j6old[][3],
                     double phiold[3], 
                     double eps2, double h2[],
                     double eps2_i[]);

  int getGravResults(int nj, int ni,
                     int index[], 
                     double xi[][3],      double vi[][3],
                     double eps2,         double h2[],
                     double acc[][3],     double jerk[][3], 
                     double snp[][3],     double crk[][3],
                     double pot[],        int nnbindex[],
                     double dsmin_i[],    bool ngb);
                     
  void forcePrediction(int nj);            
  void retrieve_predicted_j_particle(int addr,       double &mass, 
                                    double &id,     double &eps2,
                                    double pos[3],  double vel[3],
                                    double acc[3]);
                                    
  void retrieve_j_particle_state(int addr,       double &mass, 
                                double &id,     double &eps2,
                                double pos[3],  double vel[3],
                                double acc[3],  double jrk[3], double ppos[3],
                                double pvel[3], double pacc[3]);
  
  int fetch_ngb_list_from_device(int);
  int read_ngb_list(int);
  int get_ngb_list(int cluster_id,
                  int ipipe,
                  int maxlength,
                  int &nblen,
                  int nbl[]);
};

#endif 
