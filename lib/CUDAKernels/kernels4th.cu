/*

Sapporo 2 device kernels

Version 1.0
CUDA DoubleSingle kernels



*/

#include <stdio.h>

#include "include/defines.h"

#define inout
#define __out

#if 0   /* use this one to compute accelerations in DS */
#define _GACCDS_
#endif

#if 0  /* use this one to compute potentiaal in DS as well */
#define _GPOTDS_
#endif

#ifdef _GACCDS_
struct ds64
{
  float2 val;
  __host__ __device__ ds64() {}
  __host__ __device__ ds64(float x) : val(make_float2(x, x)) {}
  __host__ __device__ ds64(double x) 
  {
    val.x = (float)x;
    val.y = (float)(x - (double)val.x);
  }
  __host__ __device__ ds64 operator+=(const float x) 
  {
    const float vx = val.x + x;
    const float vy = val.y - ((vx - val.x) - x);
    val = make_float2(vx, vy);
    return *this;
  }
  __host__ __device__ double to_double() const { return (double)val.x + (double)val.y; }
  __host__ __device__ float to_float() const { return (float)((double)val.x + (double)val.y);}
};

struct devForce
{
  ds64 x, y, z;   // 6
#ifdef _GPOTDS_
  ds64 w;          // 8
#else
  float w;         // 7
  int  iPad;        // 8
#endif
  __host__ __device__ devForce() {}
  __device__ devForce(const float v) : x(v), y(v), z(v), w(v) {}
  __device__ float4 to_float4() const
  {
#ifdef _GPOTDS_
    return (float4){x.to_float(), y.to_float(), z.to_float(), w.to_float()};
#else
    return (float4){x.to_float(), y.to_float(), z.to_float(), w};
#endif
  }
  __device__ double4 to_double4() const
  {
#ifdef _GPOTDS_
    return (double4){x.to_double(), y.to_double(), z.to_double(), w.to_double()};
#else
    return (double4){x.to_double(), y.to_double(), z.to_double(), (double)w};
#endif
  }
};

#else /* not _GACCDS_ */

struct devForce
{
  float x,y,z,w;
  __device__ devForce() {}
  __device__ devForce(const float v) : x(v), y(v), z(v), w(v) {}
  __device__ float4 to_float4() const {return (float4){x,y,z,w};}
  __device__ double4 to_double4() const {return (double4){x,y,z,w};}
};

#endif

typedef float2 DS;  // double single;

struct DS4 {
  DS x, y, z, w;
};
struct DS2 {
  DS x, y;
};

__device__ __inline__ DS to_DS(double a) {
  DS b;
  b.x = (float)a;
  b.y = (float)(a - b.x);
  return b;
}

__device__ double to_double(DS a) {
  double b;
  b = (double)((double)a.x + (double)a.y);
  return b;
}


// This function computes c = a + b.
__device__ DS dsadd(DS a, DS b) {
  // Compute dsa + dsb using Knuth's trick.
  float t1 = a.x + b.x;
  float e = t1 - a.x;
  float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
  
  // The result is t1 + t2, after normalization.
  DS c;
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
  return c;
} // dsadd

// This function computes c = a + b.
__device__ DS dsadd(DS a, float b) {
  // Compute dsa + dsb using Knuth's trick.
  float t1 = a.x + b;
  float e = t1 - a.x;
  float t2 = ((b - e) + (a.x - (t1 - e))) + a.y;
  
  // The result is t1 + t2, after normalization.
  DS c;
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
  return c;
} // dsadd


__device__ __forceinline__ void body_body_interaction(
                                      inout float2   &ds2_min,
                                      inout int      &n_ngb,
                                      inout int      *ngb_list,
                                      inout devForce &acc_i, 
                                      inout float4   &jrk_i,
                                      const DS4       pos_i, 
                                      const float4    vel_i,
                                      const DS4       pos_j, 
                                      const float4    vel_j,
                                      const float     EPS2,
                                      const int       iID) 
{

  const int jID   = __float_as_int(pos_j.w.y);

//   if(iID == jID) return;

  //if (__float_as_int(pos_i.w.y) != jID)    /* assuming we always need ngb */
  if(iID != jID)
  {


    const float3 dr = {(pos_j.x.x - pos_i.x.x) + (pos_j.x.y - pos_i.x.y),
                       (pos_j.y.x - pos_i.y.x) + (pos_j.y.y - pos_i.y.y),
                       (pos_j.z.x - pos_i.z.x) + (pos_j.z.y - pos_i.z.y)};   // 3x3 = 9 FLOP


    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#if 0

    if (ds2 <= pos_i.w.x && n_ngb < NGB_PB)
      ngb_list[n_ngb++] = jID;

#else

#if ((NGB_PB & (NGB_PB - 1)) != 0)
#error "NGB_PB is not a power of 2!"
#endif

    /* WARRNING: In case of the overflow, the behaviour will be different from the original version */

    if (ds2 <= pos_i.w.x)
    {
      ngb_list[n_ngb & (NGB_PB-1)] = jID;
      n_ngb++;
    }

#endif

    ds2_min = (ds2_min.x < ds2) ? ds2_min : (float2){ds2, pos_j.w.y}; //

    const float inv_ds = rsqrtf(ds2+EPS2);

    const float mass   = pos_j.w.x;
    const float minvr1 = mass*inv_ds; 
    const float  invr2 = inv_ds*inv_ds; 
    const float minvr3 = minvr1*invr2;

    // 3*4 + 3 = 15 FLOP
    acc_i.x += minvr3 * dr.x;
    acc_i.y += minvr3 * dr.y;
    acc_i.z += minvr3 * dr.z;
    acc_i.w += (-1.0f)*minvr1;


    const float3 dv = {vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z -  vel_i.z};
    const float drdv = (-3.0f) * (minvr3*invr2) * (dr.x*dv.x + dr.y*dv.y + dr.z*dv.z);

    jrk_i.x += minvr3 * dv.x + drdv * dr.x;  
    jrk_i.y += minvr3 * dv.y + drdv * dr.y;
    jrk_i.z += minvr3 * dv.z + drdv * dr.z;

    // TOTAL 50 FLOP (or 60 FLOP if compared against GRAPE6)  
  }
}


//TODO should make this depending on if we use Fermi or GT80/GT200
// #define ajc(i, j) (i + __mul24(blockDim.x,j))
// #define ajc(i, j) (i + blockDim.x*j)
extern "C" __global__ void
//__launch_bounds__(NTHREADS)
dev_evaluate_gravity(
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *acc_i, 
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    __out double4    *jrk_i,
    const int        *id_i,
    __out double     *ds2min_i,
    __out int        *ngb_count_i,
    __out int        *ngb_list) 
{

  extern __shared__ DS4 shared_pos[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx =  blockIdx.x;
  const int Dim = blockDim.x*blockDim.y;

  float4 *shared_vel = (float4*)&shared_pos[Dim];


  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const float EPS2 = (float)EPS2_d;

  DS4 pos;
  pos.x = to_DS(pos_i[tx+ni_offset].x); 
  pos.y = to_DS(pos_i[tx+ni_offset].y);
  pos.z = to_DS(pos_i[tx+ni_offset].z);
  pos.w = to_DS(pos_i[tx+ni_offset].w);


  //Combine the particle id into the w part of the position
//   pos.w.y = __int_as_float(id_i[tx]);
  const int iID    = id_i[tx+ni_offset];

  const float4 vel = make_float4(vel_i[tx+ni_offset].x, vel_i[tx+ni_offset].y,
                                 vel_i[tx+ni_offset].z, vel_i[tx+ni_offset].w);

  const float LARGEnum = 1.0e10f;

  float2  ds2_min2;
  ds2_min2.x = LARGEnum;
  ds2_min2.y = __int_as_float(-1);

  devForce acc   (0.0f);
  float4   jrk = {0.0f, 0.0f, 0.0f, 0.0f};

  int tile = 0;
  int ni    = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      const double4 jp     = pos_j[i + tx];
      shared_pos[addr].x   = to_DS(jp.x);
      shared_pos[addr].y   = to_DS(jp.y);
      shared_pos[addr].z   = to_DS(jp.z);
      shared_pos[addr].w   = to_DS(jp.w);
      shared_pos[addr].w.y = __int_as_float(id_j[i + tx]);
      shared_vel[addr]     = (float4){vel_j[i + tx].x, 
                                      vel_j[i + tx].y, 
                                      vel_j[i + tx].z, 
                                      vel_j[i + tx].w};
    } else {
      shared_pos[addr].x = (float2){LARGEnum, 0.0f};
      shared_pos[addr].y = (float2){LARGEnum, 0.0f};
      shared_pos[addr].z = (float2){LARGEnum, 0.0f};
      shared_pos[addr].w = (float2){0.0f,  -1.0f}; 
      shared_vel[addr]   = (float4){0.0f, 0.0f, 0.0f, 0.0f};
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction(ds2_min2, n_ngb, local_ngb_list,
          acc, jrk, pos, vel,
          shared_pos[offy+k], shared_vel[offy+k], EPS2,iID);

    for (int k = j1; k < j; k++) 
      body_body_interaction(ds2_min2, n_ngb, local_ngb_list,
          acc, jrk, pos, vel,
          shared_pos[offy+k], shared_vel[offy+k], EPS2,iID);

    __syncthreads();

    tile++;
  } //end while


  float4 *shared_acc = (float4*)&shared_pos[0];
  float4 *shared_jrk = (float4*)&shared_acc[Dim];
  int    *shared_ngb = (int*   )&shared_jrk[Dim];
  int    *shared_ofs = (int*   )&shared_ngb[Dim];
  float  *shared_ds  = (float* )&shared_ofs[Dim];

  float ds2_min = ds2_min2.x;
  jrk.w         = ds2_min2.y;

  const int addr = offy + tx;
  shared_acc[addr] = acc.to_float4();
  shared_jrk[addr] = jrk;
  shared_ngb[addr] = n_ngb;
  shared_ofs[addr] = 0;
  shared_ds [addr] = ds2_min;
  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim.x; i < Dim; i += blockDim.x)
    {
      const int addr = i + tx;
      float4 acc1 = shared_acc[addr];
      float4 jrk1 = shared_jrk[addr];
      float  ds1  = shared_ds [addr];

      acc.x += acc1.x;
      acc.y += acc1.y;
      acc.z += acc1.z;
      acc.w += acc1.w;

      jrk.x += jrk1.x;
      jrk.y += jrk1.y;
      jrk.z += jrk1.z;

      if (ds1  < ds2_min) 
      {
        jrk.w   = jrk1.w;
        ds2_min  = ds1;
      }

      shared_ofs[addr] = min(n_ngb, NGB_PB);
      n_ngb           += shared_ngb[addr];
    }
    n_ngb  = min(n_ngb, NGB_PB);
  }
  __syncthreads();

  if (ty == 0) 
  {
    //Convert results to double and write
    const int addr = bx*blockDim.x + tx;
    ds2min_i[      addr]   = ds2_min;
    acc_i[         addr]   = acc.to_double4();
    jrk_i[         addr]   = (double4){jrk.x, jrk.y, jrk.z, jrk.w};
    ngb_count_i[   addr]   = n_ngb;
  }



  //Write the neighbour list
  {
    int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
    offset     += shared_ofs[addr];
    n_ngb       = shared_ngb[addr];
    for (int i = 0; i < n_ngb; i++) 
      ngb_list[offset + i] = local_ngb_list[i];
  }
}



/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni
 */ 
extern "C" __global__ void dev_reduce_forces(
    double4             *acc_i_temp, 
    double4             *jrk_i_temp,
    double              *ds_i_temp,
    int                 *ngb_count_i_temp,
    int                 *ngb_list_i_temp,
    __out double4    *result_i, 
    __out double     *ds_i,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    int               offset_ni_idx,
    int               ni_total
) {
  //  extern __shared__ float4 shared_acc[];
  //   __shared__ char shared_mem[NBLOCKS*(2*sizeof(float4) + 3*sizeof(int))];
  //   float4* shared_acc = (float4*)&shared_mem;

//   __shared__ float4     shared_acc[NBLOCKS];
//   __shared__ float4     shared_jrk[NBLOCKS];
//   __shared__ int        shared_ngb[NBLOCKS];
//   __shared__ int        shared_ofs[NBLOCKS];
//   __shared__ float      shared_ds[NBLOCKS];


  extern __shared__ float4 shared_acc[];
  float4 *shared_jrk = (float4*)&shared_acc[blockDim.x];
  int    *shared_ngb = (int*   )&shared_jrk[blockDim.x];
  int    *shared_ofs = (int*   )&shared_ngb[blockDim.x];
  float  *shared_ds  = (float* )&shared_ofs[blockDim.x];

  int index = threadIdx.x * gridDim.x + blockIdx.x;


  //Convert the data to floats
  shared_acc[threadIdx.x] = (float4){acc_i_temp[index].x, acc_i_temp[index].y, acc_i_temp[index].z, acc_i_temp[index].w};
  shared_jrk[threadIdx.x] = (float4){jrk_i_temp[index].x, jrk_i_temp[index].y, jrk_i_temp[index].z, jrk_i_temp[index].w};

  shared_ds [threadIdx.x] = (float)ds_i_temp[index];  


//   int ngb_index = threadIdx.x * NGB_PB + blockIdx.x * NGB_PB*blockDim.x;


  shared_ngb[threadIdx.x] = ngb_count_i_temp[index];
  shared_ofs[threadIdx.x] = 0;

  __syncthreads();


  int n_ngb = shared_ngb[threadIdx.x];
  if (threadIdx.x == 0) {
    float4 acc0 = shared_acc[0];
    float4 jrk0 = shared_jrk[0];
    float  ds0 = shared_ds [0];

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

      shared_ofs[i] = min(n_ngb, NGB_PP);
      n_ngb        += shared_ngb[i];

    }
    n_ngb = min(n_ngb, NGB_PP);

    jrk0.w = (int)__float_as_int(jrk0.w);

    //Store the results

    result_i    [blockIdx.x + offset_ni_idx]            = (double4){acc0.x, acc0.y, acc0.z, acc0.w};   
    result_i    [blockIdx.x + offset_ni_idx + ni_total] = (double4){jrk0.x, jrk0.y, jrk0.z, jrk0.w};
    ds_i        [blockIdx.x + offset_ni_idx] = (double) ds0;
    ngb_count_i [blockIdx.x + offset_ni_idx] = n_ngb;

  }
  __syncthreads();


  //Compute the offset of where to store the data and where to read it from
  //Store is based on ni, where to read it from is based on thread/block
  int offset     = (offset_ni_idx + blockIdx.x)  * NGB_PP + shared_ofs[threadIdx.x];
  int offset_end = (offset_ni_idx + blockIdx.x)  * NGB_PP + NGB_PP;
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
    int       *id_j_temp) {

  //int index = blockIdx.x * blockDim.x + threadIdx.x;
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
   DS version

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
  //int index = blockIdx.x * blockDim.x + threadIdx.x;

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

    DS4 pos;
    pos.x = to_DS(pos_j[index].x); pos.y = to_DS(pos_j[index].y);
    pos.z = to_DS(pos_j[index].z); pos.w = to_DS(pos_j[index].w);

    float4 vel = (float4){vel_j[index].x, vel_j[index].y, vel_j[index].z, vel_j[index].w};
    float4 acc = (float4){acc_j[index].x, acc_j[index].y, acc_j[index].z, acc_j[index].w};
    float4 jrk = (float4){jrk_j[index].x, jrk_j[index].y, jrk_j[index].z, jrk_j[index].w};

    float dt = (t_i.x - t.x.x) + (t_i.y - t.x.y);
    float dt2 = dt*dt/2.0f;
    float dt3 = dt2*dt/3.0f;

    pos.x  = dsadd(pos.x, vel.x * dt + acc.x * dt2 + jrk.x * dt3);
    pos.y  = dsadd(pos.y, vel.y * dt + acc.y * dt2 + jrk.y * dt3);
    pos.z  = dsadd(pos.z, vel.z * dt + acc.z * dt2 + jrk.z * dt3);

    vel.x += acc.x * dt + jrk.x * dt2;
    vel.y += acc.y * dt + jrk.y * dt2;
    vel.z += acc.z * dt + jrk.z * dt2;


    Ppos_j[index].x = to_double(pos.x); Ppos_j[index].y = to_double(pos.y);
    Ppos_j[index].z = to_double(pos.z); Ppos_j[index].w = to_double(pos.w);            

    Pvel_j[index] = (double4){vel.x, vel.y, vel.z, vel.w};
  }
  __syncthreads();
}






