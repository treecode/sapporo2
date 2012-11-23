/*

Sapporo 2 device kernels

Version 1.0
CUDA Double Precision 6th order hermite 

*/

#include <stdio.h>

#include "include/defines.h"

#define inout
#define __out

__device__ __forceinline__ void body_body_interaction(
                                      double2  &ds2_min,
                                      int     &n_ngb,
                                      int     *ngb_list,
                                      double4 &accNew_i, 
                                      double4 &jrkNew_i,
                                      double4 &snpNew_i,
                                      double4 pos_i, 
                                      double4 vel_i,
                                      double4 acc_i,                                        
                                      double4 pos_j, 
                                      double4 vel_j,
                                      double3 acc_j,
                                      int pjID,
                                      int piID,
                                      double &EPS2) {


  if(pjID != piID)
  {

    const double3 dr = {pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z}; //3FLOP
    const double ds2 = ((dr.x*dr.x + (dr.y*dr.y)) + dr.z*dr.z); //5 FLOP

    #if 0

        if (ds2 <= pos_i.w.x && n_ngb < NGB_PB)
          ngb_list[n_ngb++] = jID;

    #else

    #if ((NGB_PB & (NGB_PB - 1)) != 0)
    #error "NGB_PB is not a power of 2!"
    #endif

        /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
        ds2_min = (ds2_min.x < ds2) ? ds2_min : (double2){ds2, (double)pjID}; //

        if (ds2 <= pos_i.w)
        {
          ngb_list[n_ngb & (NGB_PB-1)] = pjID;
          n_ngb++;
        }

    #endif
    
    //const double inv_ds = rsqrt(ds2+EPS2);
    const double inv_ds = rsqrtf(ds2+EPS2);
    //   double inv_ds  = (1.0 / sqrt(ds2)) * (pjID != piID);

    const double mass    = pos_j.w;
    const double minvr1  = mass*inv_ds; 
    const double inv_ds2 = inv_ds*inv_ds;                         // 1 FLOP
    const double inv_ds3 = mass * inv_ds*inv_ds2;                 // 2 FLOP
    
    // 3*4 + 3 = 15 FLOP    
    accNew_i.x += inv_ds3 * dr.x;
    accNew_i.y += inv_ds3 * dr.y;
    accNew_i.z += inv_ds3 * dr.z;    
    accNew_i.w +=  (-1.0)*minvr1;; //Potential
    

    const double3 dv  = {vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z - vel_i.z}; 
    const double3 da  = {acc_j.x - acc_i.x, acc_j.y - acc_i.y, acc_j.z - acc_i.z};
    const double  v2  = (dv.x*dv.x) + (dv.y*dv.y) + (dv.z*dv.z);
    const double  ra  = (dr.x*da.x) + (dr.y*da.y) + (dr.z*da.z);

    double alpha = (((dr.x*dv.x) + dr.y*dv.y) + dr.z*dv.z) * inv_ds2;
    double beta  = (v2 + ra) * inv_ds2 + alpha * alpha;

    //Jerk
    alpha       *= -3.0;
    double3     jerk;
    jerk.x     = (inv_ds3 * dv.x) + alpha * (inv_ds3 * dr.x);
    jerk.y     = (inv_ds3 * dv.y) + alpha * (inv_ds3 * dr.y);
    jerk.z     = (inv_ds3 * dv.z) + alpha * (inv_ds3 * dr.z);

    //Snap
    alpha       *= 2.0;
    beta        *= -3.0;
    snpNew_i.x = snpNew_i.x + (inv_ds3 * da.x) + alpha * jerk.x + beta * (inv_ds3 * dr.x);
    snpNew_i.y = snpNew_i.y + (inv_ds3 * da.y) + alpha * jerk.y + beta * (inv_ds3 * dr.y);
    snpNew_i.z = snpNew_i.z + (inv_ds3 * da.z) + alpha * jerk.z + beta * (inv_ds3 * dr.z);

    //Had to reuse jerk for snap so only add to total now
    jrkNew_i.x += jerk.x;
    jrkNew_i.y += jerk.y;
    jrkNew_i.z += jerk.z;

  }

  // TOTAL 50 FLOP (or 60 FLOP if compared against GRAPE6)  
}


/*
 *  blockDim.x = ni
 *  gridDim.x  = 16, 32, 64, 128, etc. 
 */ 

//Kernel for same softening value for all particles

//TODO should make this depending on if we use Fermi or GT80/GT200
//#define ajc(i, j) (i + __mul24(blockDim.x,j))
// #define ajc(i, j) (i + blockDim.x*j)


extern "C" __global__ void dev_evaluate_gravity(
                                         const int        nj_total, 
                                         const int        nj,
                                         const int        ni_offset,
                                         const double4    *pos_j, 
                                         const double4    *pos_i,
                                         __out double4    *acc_i_out,
                                               double     EPS2,
                                         const double4    *vel_j,
                                         const int        *id_j,
                                         __out double4    *vel_i,
                                         __out double4    *jrk_i,
                                         const int        *id_i,
                                         __out double     *ds2min_i,
                                         __out int        *ngb_count_i,
                                         __out  int       *ngb_list,
                                          double4         *acc_i_in,   
                                          double4         *acc_j,       
                                          double4         *snp_i) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx =  blockIdx.x;
  const int Dim = blockDim.x*blockDim.y;


//    __shared__ char shared_mem[NTHREADS*(sizeof(double4) + sizeof(double4) + sizeof(double4) + sizeof(int)*2 + sizeof(double))];
//   __shared__ char shared_mem[NTHREADS*(sizeof(double4) + sizeof(double4) + sizeof(double4))];
//   __shared__ char shared_mem[NTHREADS*(sizeof(double4) + sizeof(double4) + sizeof(double3) + sizeof(int))];
  extern __shared__ char shared_mem[];
  double4 *shared_pos = (double4*)&shared_mem[0];
  double4 *shared_vel = (double4*)&shared_pos[Dim];
  double3 *shared_acc = (double3*)&shared_vel[Dim];
  int     *shared_id  = (int*)&shared_acc[Dim];


  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  //Read the i-particle properties for the particle that belongs to
  //this thread
  double4 pos    = pos_i[tx+ni_offset];
  double4 vel    = vel_i[tx+ni_offset];
  double4 acc    = acc_i_in[tx+ni_offset];
  int particleID = id_i [tx+ni_offset];

  //Set the softening for the i-particle
  EPS2 = vel.w;

  const float LARGEnum = 1e10f;

  double2 ds_min2;
  ds_min2.x = LARGEnum;
  ds_min2.y = (double)(-1);
  
  double4 accNew = {0.0, 0.0, 0.0, 0.0};
  double4 jrkNew = {0.0, 0.0, 0.0, 0.0};
  double4 snpNew = {0.0, 0.0, 0.0, 0.0};

  int tile = 0;
  int ni    = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      //Read j-particles into shared memory
      shared_pos[addr] = pos_j[i + tx];
      shared_vel[addr] = vel_j[i + tx];
      shared_acc[addr] = (double3){acc_j[i + tx].x, acc_j[i + tx].y, acc_j[i + tx].z};
      shared_id [addr] = id_j [i + tx]; 
    } else {
      shared_pos[addr] = (double4){LARGEnum,LARGEnum,LARGEnum,0};
      shared_id[addr]    = -1; 
      shared_vel[addr]   = (double4){0.0, 0.0, 0.0, 0.0};
      shared_acc[addr]   = (double3){0.0, 0.0, 0.0};
    }
    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) {
          body_body_interaction(ds_min2, n_ngb, local_ngb_list,
                                accNew, jrkNew, snpNew, pos, vel, acc,
                                shared_pos[offy+k], shared_vel[offy+k], 
                                shared_acc[offy+k], shared_id[offy+k], 
                                particleID, EPS2);
        }
        
    for (int k = j1; k < j; k++) {
          body_body_interaction(ds_min2, n_ngb, local_ngb_list,
                                accNew, jrkNew, snpNew, pos, vel, acc,
                                shared_pos[offy+k], shared_vel[offy+k], 
                                shared_acc[offy+k], shared_id[offy+k], 
                                particleID, EPS2);
        }
  
    __syncthreads();

    tile++;
  } //end for

  //Combine seperate results if more than one thread is used
  //per particle
  double4 *shared_acc2 = (double4*)&shared_pos[0];
  double4 *shared_snp = (double4*)&shared_acc2[Dim];
  
  acc.w = -acc.w;
  
  const int addr    = offy + tx;
  shared_acc2[addr] = accNew;
  shared_snp[addr]  = snpNew;
  __syncthreads();

  if (ty == 0) {
    for (int i = blockDim.x; i < Dim; i += blockDim.x) {
      const int addr = i + tx;
      double4 acc1 = shared_acc2[addr];
      double4 snp1 = shared_snp[addr];
     
      accNew.x += acc1.x;
      accNew.y += acc1.y;
      accNew.z += acc1.z;
      accNew.w += acc1.w;
      
      snpNew.x += snp1.x;
      snpNew.y += snp1.y;
      snpNew.z += snp1.z;
    }
  }
  __syncthreads();


  double4 *shared_jrk = (double4*)&shared_pos[0];
  int    *shared_ngb  = (int*   )&shared_jrk[Dim];
  int    *shared_ofs  = (int*   )&shared_ngb[Dim];
  double *shared_ds   = (double*)&shared_ofs[Dim];
  
//TODO re-enable this after testing
//   n_ngb = 0;


  double ds2_min = ds_min2.x;
  jrkNew.w       = ds_min2.y;
  shared_jrk[addr] = jrkNew;
  shared_ngb[addr] = n_ngb;
  shared_ofs[addr] = 0;
  shared_ds [addr] = ds2_min;
  __syncthreads();

  if (threadIdx.y == 0) {
    for (int i = blockDim.x; i < Dim; i += blockDim.x) {
      const int addr = i + tx;
      double4 jrk1   = shared_jrk[addr];
      double  ds1    = shared_ds [addr];

      jrkNew.x += jrk1.x;
      jrkNew.y += jrk1.y;
      jrkNew.z += jrk1.z;


      if (ds1  < ds2_min) {
        jrkNew.w   = jrk1.w;
        ds2_min  = ds1;
      }

      shared_ofs[addr] = min(n_ngb, NGB_PB);
      n_ngb           += shared_ngb[addr];
    }
    n_ngb  = min(n_ngb, NGB_PB);
  }
  __syncthreads();
 
  if (threadIdx.y == 0) {
    //Store the results
    const int addr = bx*blockDim.x + tx;
    ds2min_i[      addr] = ds2_min;
    acc_i_out[     addr] = accNew;
    jrk_i[         addr] = jrkNew;
    snp_i[         addr] = snpNew;
    ngb_count_i[   addr] = n_ngb;
  }

  //Write the neighbour list
  {
    int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
    offset     += shared_ofs[addr];
    n_ngb       = shared_ngb[addr];
    for (int i = 0; i < n_ngb; i++) 
      ngb_list[offset + i] = local_ngb_list[i];
  }


//   {
//     //int offset  = threadIdx.x * NBLOCKS*NGB_PB + blockIdx.x * NGB_PB;
//     int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
//     offset += shared_ofs[ajc(threadIdx.x, threadIdx.y)];
// 
//     if (threadIdx.y == 0)
//       ngb_list[offset++] = n_ngb;
// 
//     n_ngb = shared_ngb[ajc(threadIdx.x, threadIdx.y)];
//     for (int i = 0; i < n_ngb; i++) 
//       ngb_list[offset + i] = local_ngb_list[i];
//   }
}



/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni

Double precision version
 */ 
extern "C" __global__ void dev_reduce_forces(
                                        double4          *acc_i_temp, 
                                        double4          *jrk_i_temp,
                                        double           *ds_i_temp,
                                        int              *ngb_count_i_temp,
                                        int              *ngb_list_i_temp,
                                        __out double4    *result_i, 
                                        __out double     *ds_i,
                                        __out int        *ngb_count_i,
                                        __out int        *ngb_list,
                                        int               offset_ni_idx,
                                        int               ni_total,
                                        double4          *snp_i_temp) {
  //NBLOCKS*(3*sizeof(double4) + 2*sizeof(int) + sizeof(double));   
  extern __shared__ double4 shared_acc[];
  double4 *shared_jrk = (double4*)&shared_acc[blockDim.x];
  double4 *shared_snp = (double4*)&shared_jrk[blockDim.x];
  int    *shared_ngb  = (int*   )&shared_snp[blockDim.x];
  int    *shared_ofs  = (int*   )&shared_ngb[blockDim.x];
  double *shared_ds   = (double* )&shared_ofs[blockDim.x];

//   __shared__ double4 shared_acc[NBLOCKS];
//   __shared__ double4 shared_jrk[NBLOCKS];
//   __shared__ double4 shared_snp[NBLOCKS];
//   __shared__ int     shared_ngb[NBLOCKS];
//   __shared__ int     shared_ofs[NBLOCKS];
//   __shared__ double  shared_ds[NBLOCKS];

  int index = threadIdx.x * gridDim.x + blockIdx.x;

  //Early out if we are a block for non existent particle
  if((blockIdx.x + offset_ni_idx) >= ni_total)
    return;

  //Convert the data to floats
  shared_acc[threadIdx.x] = acc_i_temp[index];
  shared_jrk[threadIdx.x] = jrk_i_temp[index];
  shared_snp[threadIdx.x] = snp_i_temp[index];
  shared_ds [threadIdx.x] = ds_i_temp[index]; 

  shared_ngb[threadIdx.x] = ngb_count_i_temp[index];
  shared_ofs[threadIdx.x] = 0;

//   int ngb_index = threadIdx.x * NGB_PB + blockIdx.x * NGB_PB*blockDim.x;
//   shared_ngb[threadIdx.x] = ngb_list[ngb_index];
//   shared_ofs[threadIdx.x] = 0;
         
  __syncthreads();

  int n_ngb = shared_ngb[threadIdx.x];
  if (threadIdx.x == 0) {
    double4 acc0 = shared_acc[0];
    double4 jrk0 = shared_jrk[0];
    double4 snp0 = shared_snp[0];
    double  ds0  = shared_ds [0];

    for (int i = 1; i < blockDim.x; i++) {
      acc0.x += shared_acc[i].x;
      acc0.y += shared_acc[i].y;
      acc0.z += shared_acc[i].z;
      acc0.w += shared_acc[i].w;

      jrk0.x += shared_jrk[i].x;
      jrk0.y += shared_jrk[i].y;
      jrk0.z += shared_jrk[i].z;

      snp0.x += shared_snp[i].x;
      snp0.y += shared_snp[i].y;
      snp0.z += shared_snp[i].z;


      if (shared_ds[i] < ds0) {
        ds0    = shared_ds[i];
        jrk0.w = shared_jrk[i].w;
      }

      shared_ofs[i] = min(n_ngb, NGB_PB);
      n_ngb += shared_ngb[i];
    }
    n_ngb = min(n_ngb, NGB_PB);

    jrk0.w = (int)(jrk0.w);


    //Store the results
    result_i    [blockIdx.x + offset_ni_idx]              = acc0;
    result_i    [blockIdx.x + offset_ni_idx + ni_total]   = jrk0;
    result_i    [blockIdx.x + offset_ni_idx + 2*ni_total] = snp0;
    ds_i        [blockIdx.x + offset_ni_idx] = ds0;
    ngb_count_i [blockIdx.x + offset_ni_idx] = n_ngb;
  }
  __syncthreads();

  //Compute the offset of where to store the data and where to read it from
  //Store is based on ni, where to read it from is based on thread/block
  int offset     = (offset_ni_idx + blockIdx.x)  * NGB_PB + shared_ofs[threadIdx.x];
  int offset_end = (offset_ni_idx + blockIdx.x)  * NGB_PB + NGB_PB;
  int ngb_index  = threadIdx.x * NGB_PB + blockIdx.x * NGB_PB*blockDim.x;


  n_ngb = shared_ngb[threadIdx.x];
  __syncthreads();
  for (int i = 0; i < n_ngb; i++)
  {
    if (offset + i < offset_end){
        ngb_list[offset + i] = ngb_list_i_temp[ngb_index + i];
    }
  }

}


/*
 * Function that moves the (changed) j-particles
 * to the correct address location.
*/
extern "C" __global__ void dev_copy_particles(int nj, 
                                              double4   *pos_j, 
                                              double4   *pos_j_temp,
                                              int       *address_j,
                                              double2   *t_j,
                                              double4   *Ppos_j, 
                                              double4   *Pvel_j,                                              
                                              double4   *vel_j,
                                              double4   *acc_j,
                                              double4   *jrk_j,
                                              int       *id_j,
                                              double2   *t_j_temp,                                              
                                              double4   *vel_j_temp,
                                              double4   *acc_j_temp,
                                              double4   *jrk_j_temp,
                                              int       *id_j_temp,
                                              double4   *Pacc_j,
                                              double4   *snp_j,
                                              double4   *crk_j,
                                              double4   *snp_j_temp,
                                              double4   *crk_j_temp) {

  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;

  //Copy the changed particles
  if (index < nj)
  {
    t_j  [address_j[index]] = t_j_temp[index];

    Ppos_j[address_j[index]] = pos_j_temp[index];
     pos_j[address_j[index]] = pos_j_temp[index];

    Pvel_j[address_j[index]] = vel_j_temp[index];
     vel_j[address_j[index]] = vel_j_temp[ index];

    Pacc_j[address_j[index]] = acc_j_temp[index];
     acc_j[address_j[index]] = acc_j_temp[index];

    jrk_j[address_j[index]]  = jrk_j_temp[index];
    snp_j[address_j[index]]  = snp_j_temp[index];
    crk_j[address_j[index]]  = crk_j_temp[index];

    id_j[address_j[index]]   = id_j_temp[index];
  }
}

/*

Function to predict the particles
Double Precision version

6th order hermite
*/
extern "C" __global__ void dev_predictor(int nj,
                                        double  t_i_d,
                                        double2 *t_j,
                                        double4 *Ppos_j,
                                        double4 *Pvel_j,
                                        double4 *pos_j, 
                                        double4 *vel_j,
                                        double4 *acc_j,
                                        double4 *jrk_j,
                                        double4 *Pacc_j,
                                        double4 *snp_j,
                                        double4 *crk_j){

  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;
  
  if (index < nj) {

    double dt = t_i_d  - t_j[index].x;
    double dt2 = (1./2.)*dt;
    double dt3 = (1./3.)*dt;
    double dt4 = (1./4.)*dt;
    double dt5 = (1./5.)*dt;

    double4  pos         = pos_j[index];
    double4  vel         = vel_j[index];
    double4  acc         = acc_j[index];
    double4  jrk         = jrk_j[index];
    double4  snp         = snp_j[index];
    double4  crk         = crk_j[index];

    //Positions
    pos.x += dt  * (vel.x +  dt2 * (acc.x + dt3 * (jrk.x + 
             dt4 * (snp.x +  dt5 * (crk.x)))));
    pos.y += dt  * (vel.y +  dt2 * (acc.y + dt3 * (jrk.y + 
             dt4 * (snp.y +  dt5 * (crk.y)))));
    pos.z += dt  * (vel.z +  dt2 * (acc.z + dt3 * (jrk.z + 
             dt4 * (snp.z +  dt5 * (crk.z)))));
    Ppos_j[index] = pos;

    //Velocities
    vel.x += dt * (acc.x + dt2 * (jrk.x + 
             dt3 * (snp.x +  dt4 * (crk.x))));
    vel.y += dt * (acc.y + dt2 * (jrk.y + 
             dt3 * (snp.y +  dt4 * (crk.y))));
    vel.z += dt * (acc.z + dt2 * (jrk.z + 
             dt3 * (snp.z +  dt4 * (crk.z))));
    Pvel_j[index] = vel;


    //Accelerations
    acc.x += dt * (jrk.x + dt2 * (snp.x +  dt3 * (crk.x)));
    acc.y += dt * (jrk.y + dt2 * (snp.y +  dt3 * (crk.y)));
    acc.z += dt * (jrk.z + dt2 * (snp.z +  dt3 * (crk.z)));
    Pacc_j[index] = acc;
  }
}

extern "C" __global__ void dev_no_predictor(int nj,
                                        double  t_i_d,
                                        double2 *t_j,
                                        double4 *Ppos_j,
                                        double4 *Pvel_j,
                                        double4 *pos_j, 
                                        double4 *vel_j,
                                        double4 *acc_j,
                                        double4 *jrk_j,
                                        double4 *Pacc_j,
                                        double4 *snp_j,
                                        double4 *crk_j){

  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;
  
  if (index < nj) {

    double4  pos         = pos_j[index];
    double4  vel         = vel_j[index];
    double4  acc         = acc_j[index];
    //Positions

    Ppos_j[index] = pos;
    Pvel_j[index] = vel;
    Pacc_j[index] = acc;
  }
}

