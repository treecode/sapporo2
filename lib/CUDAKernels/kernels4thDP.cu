/*

Sapporo 2 device kernels

Version 1.0
CUDA Double Precision kernels

*/

#include <stdio.h>

#include "include/defines.h"

#define inout
#define __out

typedef float2 DS;  // double single;


struct DS2 {
  DS x, y;
};

__device__ DS to_DS(double a) {
  DS b;
  b.x = (float)a;
  b.y = (float)(a - b.x);
  return b;
}



struct devForce
{
  double x,y,z,w;
  __device__ devForce() {}
  __device__ devForce(const double v) : x(v), y(v), z(v), w(v) {}
  __device__ float4 to_float4() const {return (float4){x,y,z,w};}
  __device__ double4 to_double4() const {return (double4){x,y,z,w};}
};

__device__ __forceinline__ void body_body_interaction(inout double2    &ds2_min,
                                      inout int      &n_ngb,
                                      inout int      *ngb_list,
                                      inout devForce &acc_i, 
                                      inout double3   &jrk_i,
                                      const double4       pos_i, 
                                      const double4    vel_i,
                                      const double4       pos_j, 
                                      const double3    vel_j,
                                      const int         jID, 
                                      const int         iID,
                                      const double     EPS2) 
{

  if (iID != jID)    /* assuming we always need ngb */
  {
    const double3 dr = {pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z};
    const double ds2 = ((dr.x*dr.x + (dr.y*dr.y)) + dr.z*dr.z);

#if 0

    if (ds2 <= pos_i.w.x && n_ngb < NGB_PB)
      ngb_list[n_ngb++] = jID;

#else

#if ((NGB_PB & (NGB_PB - 1)) != 0)
#error "NGB_PB is not a power of 2!"
#endif

    /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
    ds2_min = (ds2_min.x < ds2) ? ds2_min : (double2){ds2, jID}; //

    if (ds2 <= pos_i.w)
    {
      ngb_list[n_ngb & (NGB_PB-1)] = jID;
      n_ngb++;
    }

#endif



    const double inv_ds = rsqrt(ds2+EPS2);

    const double mass   = pos_j.w;
    const double minvr1 = mass*inv_ds; 
    const double  invr2 = inv_ds*inv_ds; 
    const double minvr3 = minvr1*invr2;

    // 3*4 + 3 = 15 FLOP
    acc_i.x += minvr3 * dr.x;
    acc_i.y += minvr3 * dr.y;
    acc_i.z += minvr3 * dr.z;
    acc_i.w += (-1.0)*minvr1;

    const double3 dv = {vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z -  vel_i.z};
    const double drdv = (-3.0) * (minvr3*invr2) * (dr.x*dv.x + dr.y*dv.y + dr.z*dv.z);

    jrk_i.x += minvr3 * dv.x + drdv * dr.x;  
    jrk_i.y += minvr3 * dv.y + drdv * dr.y;
    jrk_i.z += minvr3 * dv.z + drdv * dr.z;
  }
  // TOTAL 50 FLOP (or 60 FLOP if compared against GRAPE6)  
}


/*
 *  blockDim.x = ni
 *  gridDim.x  = 16, 32, 64, 128, etc. 
 */ 


//TODO should make this depending on if we use Fermi or GT80/GT200
//#define ajc(i, j) (i + __mul24(blockDim.x,j))
#define ajc(i, j) (i + blockDim.x*j)
extern "C" __global__ void
//__launch_bounds__(NTHREADS)
dev_evaluate_gravity(
    const int        nj_total, 
    const int        nj,
    const int        offset,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *acc_i, 
    const double     EPS2,
    const double4    *vel_j,
    const int        *id_j,                                     
    __out double4    *vel_i,                                     
    __out double4    *jrk_i,
    const int        *id_i,
    __out int        *ngb_list) 
{

  extern __shared__ char shared_mem[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx =  blockIdx.x;
  const int Dim = blockDim.x*blockDim.y;

  double4 *shared_pos = (double4*)&shared_mem[0];
  double3 *shared_vel = (double3*)&shared_pos[Dim];
  int     *shared_id  = (int*)&shared_vel[Dim];

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;


  double4 pos = pos_i[threadIdx.x];
  //Combine the particle id into the w part of the position
  int particleID = id_i[threadIdx.x];

  double4 vel = vel_i[threadIdx.x];

  const float LARGEnum = 1.0e10f;

  double2 ds2_min2;
  ds2_min2.x = LARGEnum;
  ds2_min2.y = (double)(-1);
 
  devForce acc   (0.0);
  double3 jrk = {0.0, 0.0, 0.0};

  int tile = 0;
  int ni    = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      shared_pos[addr]     = pos_j[i + tx];
      shared_id[addr]      = id_j[i + tx]; 
      shared_vel[addr]     = (double3){
                                    vel_j[i + tx].x, 
                                    vel_j[i + tx].y,
                                    vel_j[i + tx].z};
    } else {
      shared_pos[addr] = (double4){LARGEnum,LARGEnum,LARGEnum,0};
      shared_id[addr]  = -1; 
      shared_vel[addr] = (double3){0.0, 0.0, 0.0}; 
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction(ds2_min2, n_ngb, local_ngb_list,
          acc, jrk, pos, vel,
          shared_pos[offy+k], shared_vel[offy+k],
          shared_id [offy+k], particleID, EPS2);


    for (int k = j1; k < j; k++) 
      body_body_interaction(ds2_min2, n_ngb, local_ngb_list,
          acc, jrk, pos, vel,
          shared_pos[offy+k], shared_vel[offy+k],
          shared_id [offy+k], particleID, EPS2);

    __syncthreads();

    tile++;
  } //end while


  //Reduce in two steps to save shared memory
  double4 *shared_jrk = (double4*)&shared_pos[0]; 
  double  *shared_ds  = (double* )&shared_jrk[Dim];

  double ds2_min = ds2_min2.x;

  double4 jerkNew = (double4){jrk.x, jrk.y, jrk.z,  ds2_min2.y};

  const int addr = offy + tx;
  shared_jrk[addr] = jerkNew;
  shared_ds [addr] = ds2_min;
  __syncthreads();


 if (ty == 0) {
    for (int i = blockDim.x; i < Dim; i += blockDim.x) {
      const int addr = i + tx;
      double4 jrk1 = shared_jrk[addr];
      double  ds1  = shared_ds [addr];
 
      jerkNew.x += jrk1.x;
      jerkNew.y += jrk1.y;
      jerkNew.z += jrk1.z;
      
      if (ds1  < ds2_min) {
        jerkNew.w    = jrk1.w;
        ds2_min  = ds1;
      }
    }
  }
  __syncthreads();

  double4 *shared_acc = (double4*)&shared_pos[0];  
  int     *shared_ngb = (int*   )&shared_acc[Dim];
  int     *shared_ofs = (int*   )&shared_ngb[Dim];  

  shared_acc[addr] = acc.to_double4();
  shared_ngb[addr] = n_ngb;
  shared_ofs[addr] = 0;
  __syncthreads();

 if (ty == 0) {
    for (int i = blockDim.x; i < Dim; i += blockDim.x) {
      const int addr = i + tx;
      double4 acc1 = shared_acc[addr];
     
      acc.x += acc1.x;
      acc.y += acc1.y;
      acc.z += acc1.z;
      acc.w += acc1.w;
     
      shared_ofs[addr] = min(n_ngb + 1, NGB_PB);
      n_ngb += shared_ngb[addr];
    }
    n_ngb  = min(n_ngb, NGB_PB);
  }
  __syncthreads();


  if (ty == 0) 
  {
    //Convert results to double and write
    const int addr = bx*blockDim.x + tx;
    vel_i[offset + addr].w = ds2_min;
    acc_i[         addr] = acc.to_double4();
    jrk_i[         addr] = jerkNew;
  }


  {
    //int offset  = threadIdx.x * NBLOCKS*NGB_PB + blockIdx.x * NGB_PB;
    int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
    offset += shared_ofs[ajc(threadIdx.x, threadIdx.y)];

    if (threadIdx.y == 0)
      ngb_list[offset++] = n_ngb;

    n_ngb = shared_ngb[ajc(threadIdx.x, threadIdx.y)];
    for (int i = 0; i < n_ngb; i++) 
      ngb_list[offset + i] = local_ngb_list[i];
  }
}



/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni

Double precision version
 */ 
extern "C" __global__ void dev_reduce_forces(double4 *acc_i, 
                                             double4 *jrk_i,
                                             double  *ds_i,
                                             double4 *vel_i,
                                             int     offset_ds,
                                             int     offset,
                                             int     *ngb_list) {
  
  extern __shared__ double4 shared_acc[];
//    __shared__ char shared_mem[NBLOCKS*(2*sizeof(double4) + 2*sizeof(int) + sizeof(double))];
//   double4* shared_acc = (double4*)&shared_mem[0];
// 
// 
  double4 *shared_jrk = (double4*)&shared_acc[blockDim.x];
  int    *shared_ngb = (int*   )&shared_jrk[blockDim.x];
  int    *shared_ofs = (int*   )&shared_ngb[blockDim.x];
  double *shared_ds  = (double* )&shared_ofs[blockDim.x];


//   __shared__ double4     shared_acc[NBLOCKS];
//   __shared__ double4     shared_jrk[NBLOCKS];
//   __shared__ int         shared_ngb[NBLOCKS];
//   __shared__ int         shared_ofs[NBLOCKS];
//   __shared__ double      shared_ds[NBLOCKS];

  
  int index = threadIdx.x * gridDim.x + blockIdx.x;

//   shared_acc[threadIdx.x] = acc_i[index];
//   shared_jrk[threadIdx.x] = jrk_i[index];
//   shared_ds [threadIdx.x] = vel_i[offset_ds + index].w;

  //Convert the data to floats
  shared_acc[threadIdx.x] = (double4){acc_i[index].x, acc_i[index].y, acc_i[index].z, acc_i[index].w};
  shared_jrk[threadIdx.x] = (double4){jrk_i[index].x, jrk_i[index].y, jrk_i[index].z, jrk_i[index].w};
  shared_ds [threadIdx.x] = (double)vel_i[offset_ds + index].w;  //TODO JB dont we miss the value at vel_i[0 + x] this way?


//   int ngb_index = threadIdx.x * NGB_PB + blockIdx.x * NGB_PB*NBLOCKS;
  int ngb_index = threadIdx.x * NGB_PB + blockIdx.x * NGB_PB*blockDim.x;
  shared_ngb[threadIdx.x] = ngb_list[ngb_index];
  shared_ofs[threadIdx.x] = 0;
         
  __syncthreads();

  int n_ngb = shared_ngb[threadIdx.x];
  if (threadIdx.x == 0) {
    double4 acc0 = shared_acc[0];
    double4 jrk0 = shared_jrk[0];
    double  ds0 = shared_ds [0];

    for (int i = 1; i < blockDim.x; i++) {
      acc0.x += shared_acc[i].x;
      acc0.y += shared_acc[i].y;
      acc0.z += shared_acc[i].z;
      acc0.w += shared_acc[i].w;

      jrk0.x += shared_jrk[i].x;
      jrk0.y += shared_jrk[i].y;
      jrk0.z += shared_jrk[i].z;

      if (shared_ds[i] < ds0) {
        ds0    = shared_ds[i];
        jrk0.w = shared_jrk[i].w;
      }

      shared_ofs[i] = min(n_ngb + 1, NGB_PP);
      n_ngb += shared_ngb[i];
    }
    n_ngb = min(n_ngb, NGB_PP);

//     jrk0.w = (int)__float_as_int(jrk0.w);
    jrk0.w = (int)(jrk0.w);


    //Store the results
    acc_i[blockIdx.x] = (double4){acc0.x, acc0.y, acc0.z, acc0.w};
    jrk_i[blockIdx.x] = (double4){jrk0.x, jrk0.y, jrk0.z, jrk0.w};;
    ds_i [blockIdx.x] = ds0;
  }
  __syncthreads();


  offset += blockIdx.x * NGB_PP + shared_ofs[threadIdx.x];
  int offset_end;
  if (threadIdx.x == 0) {
    shared_ofs[0] = offset + NGB_PP;
    ngb_list[offset++] = n_ngb;
  }
  __syncthreads();
  
  offset_end = shared_ofs[0];
  
  n_ngb = shared_ngb[threadIdx.x];
  for (int i = 0; i < n_ngb; i++)
    if (offset + i < offset_end)
      ngb_list[offset + i] = ngb_list[ngb_index + 1 + i];

  
}


/*
 * Function that moves the (changed) j-particles
 * to the correct address location.
*/
extern "C" __global__ void dev_copy_particles(int nj, int nj_max,
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
                                              int       *id_j_temp) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
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

    acc_j[address_j[index]]  = acc_j_temp[index];
    jrk_j[address_j[index]]  = jrk_j_temp[index];

    id_j[address_j[index]]   = id_j_temp[index];
  }
}

/*

Function to predict the particles
Double Precision version

TODO timestep info

*/
extern "C" __global__ void dev_predictor(int nj,
                              double  t_i_d,
                              double2 *t_j,
                              double4 *Ppos_j,
                              double4 *Pvel_j,
                              double4 *pos_j, 
                              double4 *vel_j,
                              double4 *acc_j,
                              double4 *jrk_j) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;

  
  if (index < nj) {

    //Convert the doubles to DS
    DS2 t;
    t.x = to_DS(t_j[index].x);
    t.y = to_DS(t_j[index].y);

    DS t_i;
    t_i = to_DS(t_i_d);

    double4  pos;
//     pos.x = to_DS(pos_j[index].x); pos.y = to_DS(pos_j[index].y);
//     pos.z = to_DS(pos_j[index].z); pos.w = to_DS(pos_j[index].w);
    pos = pos_j[index];
  
    double4 vel = (double4){vel_j[index].x, vel_j[index].y, vel_j[index].z, vel_j[index].w};
    double4 acc = (double4){acc_j[index].x, acc_j[index].y, acc_j[index].z, acc_j[index].w};
    double4 jrk = (double4){jrk_j[index].x, jrk_j[index].y, jrk_j[index].z, jrk_j[index].w};
  
    double dt = (t_i.x - t.x.x) + (t_i.y - t.x.y);
    double dt2 = dt*dt/2.0;
    double dt3 = dt2*dt/3.0;
    
    pos.x  += vel.x * dt + acc.x * dt2 + jrk.x * dt3;
    pos.y  += vel.y * dt + acc.y * dt2 + jrk.y * dt3;
    pos.z  += vel.z * dt + acc.z * dt2 + jrk.z * dt3;

    //vel.x += acc.x * dt + jrk.x * dt2;
    //vel.y += acc.y * dt + jrk.y * dt2;
    //vel.z += acc.z * dt + jrk.z * dt2;
    dt2 = dt*(1.0/2.0);


    vel.x += dt*(acc.x  + dt2*jrk.x);
    vel.y += dt*(acc.y  + dt2*jrk.y);
    vel.z += dt*(acc.z  + dt2*jrk.z);


    Ppos_j[index] = pos;
//     Ppos_j[index].x = to_double(pos.x); Ppos_j[index].y = to_double(pos.y);
//     Ppos_j[index].z = to_double(pos.z); Ppos_j[index].w = to_double(pos.w);            

    Pvel_j[index] = (double4){vel.x, vel.y, vel.z, vel.w};
  }
  __syncthreads();
}

