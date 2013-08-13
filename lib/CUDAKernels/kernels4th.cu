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

__device__ __inline__ double4 to_double4(DS4 a) {


  return make_double4(to_double(a.x), to_double(a.y),
                      to_double(a.z), to_double(a.w));
}

__device__ __inline__ double4 to_double4(double4 a) { return a; }

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

struct DSX
{
  float x,y;

  __host__ __device__ __forceinline__ void operator=(const DS b) 
  {
    x = b.x; y = b.y;
  }

  __host__ __device__ __forceinline__ void operator=(const double b) 
  {
    x = (float)b;
    y = (float)(b - x);
  }

  __host__ __device__ __forceinline__ float operator-(const DSX &b) const
  {
    return  (x - b.x) + (y - b.y);
  }

  __host__ __device__ __forceinline__ float operator-(const DS &b) const
  {
    return  (x - b.x) + (y - b.y);
  }

  __host__ __device__ __forceinline__ bool operator<=(const float b) const
  {
    return  x < b;
  }
  __host__ __device__ __forceinline__ bool operator>(const float b) const
  {
    return  x > b;
  }

};

struct DS4X {
  DSX x, y, z, w;
};
struct DS2X {
  DSX x, y;
};


__device__ __forceinline__  float RSQRT(float val) { return rsqrtf(val); }
__device__ __forceinline__  double RSQRT(double val) { return rsqrt(val); }
// template<typename T> __device__ __forceinline__  T RSQRT(T val) { return rsqrt(val); }
// template<>           __device__ __forceinline__  float RSQRT(float val) { return rsqrtf(val); }
// template<>           __device__ __forceinline__  double RSQRT(double val) { return rsqrt(val); }


// template<>           __device__ __forceinline__  double RSQRT(double val) { return rsqrtf(val); }
// template<>           __device__ __forceinline__  double RSQRT(double val) { return 1.0/sqrt(val); }

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
    __out float2     *ds2min_i,
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
    ds2min_i[      addr].y = ds2_min;
    ds2min_i[      addr].x = __float_as_int(jrk.w);
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

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

extern "C" __global__ void
//__launch_bounds__(NTHREADS)
dev_evaluate_gravity_reduce(
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
//,    __out int        *atomicVal) 
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
  __syncthreads(); //TODO can be removed right?

  if (ty == 0) 
  {
    int *atomicVal = ngb_list;
    if(threadIdx.x == 0)
    {
      int res          = atomicExch(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
        shared_ds[blockDim.x] = (float)waitCounter;
        res = atomicExch(&atomicVal[0], 1); 
      }
    }
    __syncthreads();
    //Convert results to double and write
    const int addr = bx*blockDim.x + tx;
    ds2min_i[tx+ni_offset]   = ds2_min;
    double4 temp = acc.to_double4();
    acc_i[   tx+ni_offset].x   += temp.x;
    acc_i[   tx+ni_offset].y   += temp.y;
    acc_i[   tx+ni_offset].z   += temp.z;
    acc_i[   tx+ni_offset].w   += temp.w;

    jrk_i[   tx+ni_offset].x   += jrk.x;
    jrk_i[   tx+ni_offset].y   += jrk.y;
    jrk_i[   tx+ni_offset].z   += jrk.z;
    jrk_i[   tx+ni_offset].w   += jrk.w;
//     jrk_i[   tx+ni_offset]   = (double4){jrk.x, jrk.y, jrk.z, jrk.w};
/*
    atomicAdd((float*)&acc_i[   tx+ni_offset].x, (float)temp.x);
    atomicAdd((float*)&acc_i[   tx+ni_offset].y, (float)temp.y);
    atomicAdd((float*)&acc_i[   tx+ni_offset].z, (float)temp.z);
    atomicAdd((float*)&acc_i[   tx+ni_offset].w, (float)temp.w);

    atomicAdd((float*)&jrk_i[   tx+ni_offset].x, (float)jrk.x);
    atomicAdd((float*)&jrk_i[   tx+ni_offset].y, (float)jrk.y);
    atomicAdd((float*)&jrk_i[   tx+ni_offset].z, (float)jrk.z);
    atomicAdd((float*)&jrk_i[   tx+ni_offset].w, (float)jrk.w);*/

//     atomicAdd(&acc_i[   tx+ni_offset].x, temp.x);
//     atomicAdd(&acc_i[   tx+ni_offset].y, temp.y);
//     atomicAdd(&acc_i[   tx+ni_offset].z, temp.z);
//     atomicAdd(&acc_i[   tx+ni_offset].w, temp.w);
// 
//     atomicAdd(&jrk_i[   tx+ni_offset].x, jrk.x);
//     atomicAdd(&jrk_i[   tx+ni_offset].y, jrk.y);
//     atomicAdd(&jrk_i[   tx+ni_offset].z, jrk.z);
//     atomicAdd(&jrk_i[   tx+ni_offset].w, jrk.w);



    ngb_count_i[tx+ni_offset]   = n_ngb;

    if(threadIdx.x == 0)
    {
      atomicExch(&atomicVal[0], 0); //Release the lock
    }
  }


//TODO
//   //Write the neighbour list
//   {
//     int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
//     offset     += shared_ofs[addr];
//     n_ngb       = shared_ngb[addr];
//     for (int i = 0; i < n_ngb; i++) 
//       ngb_list[offset + i] = local_ngb_list[i];
//   }
}








extern "C" __global__ void
//__launch_bounds__(NTHREADS)
dev_evaluate_gravity_allinone(
    const int        nj_total, 
//     const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i, 
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
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
  const uint index = bx * blockDim.x + tx;


  float4 *shared_vel = (float4*)&shared_pos[Dim];


  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const float EPS2 = (float)EPS2_d;

  DS4 pos;
  pos.x = to_DS(pos_i[index].x); 
  pos.y = to_DS(pos_i[index].y);
  pos.z = to_DS(pos_i[index].z);
  pos.w = to_DS(pos_i[index].w);


  //Combine the particle id into the w part of the position
//   pos.w.y = __int_as_float(id_i[tx]);
  const int iID    = id_i[index];

  const float4 vel = make_float4(vel_i[index].x, vel_i[index].y,
                                 vel_i[index].z, vel_i[index].w);

  const float LARGEnum = 1.0e10f;

  float2  ds2_min2;
  ds2_min2.x = LARGEnum;
  ds2_min2.y = __int_as_float(-1);

  devForce acc   (0.0f);
  float4   jrk = {0.0f, 0.0f, 0.0f, 0.0f};

  int tile = 0;
//   int ni    = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;

  int ni = 0;              //Test
  int nend = nj_total;     //Test
  for (int i = ni; i < nend; i += blockDim.x)        //Test
//   for (int i = ni; i < ni+nj; i += blockDim.x)
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

//     const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j  = min(nj_total - tile*blockDim.x, blockDim.x); //Test
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
    {
      body_body_interaction(ds2_min2, n_ngb, local_ngb_list,
          acc, jrk, pos, vel,
          shared_pos[offy+k], shared_vel[offy+k], EPS2,iID);
    }

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
    ds2min_i[index]              = ds2_min;
    result_i[index]              = acc.to_double4();
    result_i[index+ni_total]     = (double4){jrk.x, jrk.y, jrk.z, (int)__float_as_int(jrk.w)};
    ngb_count_i[index]           = n_ngb;
  }

  //Write the neighbour list
  {
    for (int i = 0; i < n_ngb; i++) 
      ngb_list[(index * NGB_PB) + i] = local_ngb_list[i];
  }
}


/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni
 */ 
extern "C" __global__ void dev_reduce_forces(
    double4             *acc_i_temp, 
    double4             *jrk_i_temp,
    float2              *ds_i_temp,
    int                 *ngb_count_i_temp,
    int                 *ngb_list_i_temp,
    __out double4    *result_i, 
    __out float2     *ds_i,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    int               offset_ni_idx,
    int               ni_total
) {
  extern __shared__ float4 shared_acc[];
  float4 *shared_jrk = (float4*)&shared_acc[blockDim.x];
  int    *shared_ngb = (int*   )&shared_jrk[blockDim.x];
  int    *shared_ofs = (int*   )&shared_ngb[blockDim.x];
  float  *shared_ds  = (float* )&shared_ofs[blockDim.x];

  int index = threadIdx.x * gridDim.x + blockIdx.x;

  //Early out if we are a block for non existent particle
  if((blockIdx.x + offset_ni_idx) >= ni_total)
    return;

  //Convert the data to floats
  shared_acc[threadIdx.x] = (float4){acc_i_temp[index].x, acc_i_temp[index].y, acc_i_temp[index].z, acc_i_temp[index].w};
  shared_jrk[threadIdx.x] = (float4){jrk_i_temp[index].x, jrk_i_temp[index].y, jrk_i_temp[index].z, jrk_i_temp[index].w};

  shared_ds [threadIdx.x] = (float)ds_i_temp[index].y;  


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

      shared_ofs[i] = min(n_ngb, NGB_PB);
      n_ngb        += shared_ngb[i];

    }
    n_ngb = min(n_ngb, NGB_PB);

//     jrk0.w = (int)__float_as_int(jrk0.w);

    //Store the results

    result_i    [blockIdx.x + offset_ni_idx]            = (double4){acc0.x, acc0.y, acc0.z, acc0.w};   
    result_i    [blockIdx.x + offset_ni_idx + ni_total] = (double4){jrk0.x, jrk0.y, jrk0.z, jrk0.w};
    ds_i        [blockIdx.x + offset_ni_idx].x = __float_as_int(jrk0.w);
    ds_i        [blockIdx.x + offset_ni_idx].y =  ds0;
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

extern "C" __global__ void dev_copy_particles(
    int nj,
    int         integrationOrder,
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
    double4   *crk_j_temp)
{
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;

  //Copy the changed particles
  if (index < nj)
  {    
    pos_j[address_j[index]] = pos_j_temp[index];

    if(integrationOrder >= FOURTH)
    {
      t_j  [address_j[index]] = t_j_temp[index];
      Ppos_j[address_j[index]] = pos_j_temp[index];
      

      Pvel_j[address_j[index]] = vel_j_temp[index];
      vel_j[address_j[index]] = vel_j_temp[ index];

      acc_j[address_j[index]]  = acc_j_temp[index];
      jrk_j[address_j[index]]  = jrk_j_temp[index];

      id_j[address_j[index]]   = id_j_temp[index];
    }

    if(integrationOrder >= SIXTH)
    {
      Pacc_j[address_j[index]] = acc_j_temp[index];
      snp_j[address_j[index]]  = snp_j_temp[index];
      crk_j[address_j[index]]  = crk_j_temp[index];
    }

  }
}

/*
   Function to predict the particles
 */

extern "C" __global__ void dev_predictor(
    int         nj,
    double      t_i,
    int         integrationOrder,
    double2     *t_j,
    double4     *Ppos_j,
    double4     *pos_j, 
    double4     *Pvel_j,    
    double4     *vel_j,
    double4     *acc_j,
    double4     *jrk_j,
    double4     *Pacc_j,
    double4     *snp_j,
    double4     *crk_j)
{
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint index = bid * blockDim.x + tid;

  if(integrationOrder == GRAPE5) return;


  if (index < nj)
  {
#if 1
    double dt  = t_i - t_j  [index].x;
    double dt2 = (1./2.)*dt;
    double dt3 = (1./3.)*dt;
    double dt4 = (1./4.)*dt;
    double dt5 = (1./5.)*dt;

    if(integrationOrder <= FOURTH)
    {
      dt4 = 0;
      dt5 = 0;
    }

    double4 pos = pos_j[index];
    double4 vel = vel_j[index];
    double4 acc = acc_j[index];
    double4 jrk = jrk_j[index];

    double4 snp = make_double4(0,0,0,0);
    double4 crk = make_double4(0,0,0,0);

    if(integrationOrder >= SIXTH)
    {
      snp         = snp_j[index];
      crk         = crk_j[index];
    }

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
    if(integrationOrder >= SIXTH)
    {
      acc.x += dt * (jrk.x + dt2 * (snp.x +  dt3 * (crk.x)));
      acc.y += dt * (jrk.y + dt2 * (snp.y +  dt3 * (crk.y)));
      acc.z += dt * (jrk.z + dt2 * (snp.z +  dt3 * (crk.z)));
      Pacc_j[index] = acc;
    }


#else
    double tj   = t_j  [index].x;
    double dt  = (t_i - tj);
    double dt2 = dt*dt/2.0;
    double dt3 = dt2*dt/3.0;

    double4 pos = pos_j[index];
    double4 vel = vel_j[index];
    double4 acc = acc_j[index];
    double4 jrk = jrk_j[index];

    pos.x  += vel.x * dt + acc.x * dt2 + jrk.x * dt3;
    pos.y  += vel.y * dt + acc.y * dt2 + jrk.y * dt3;
    pos.z  += vel.z * dt + acc.z * dt2 + jrk.z * dt3;

    vel.x += acc.x * dt + jrk.x * dt2;
    vel.y += acc.y * dt + jrk.y * dt2;
    vel.z += acc.z * dt + jrk.z * dt2;

    Ppos_j[index] = pos;
    Pvel_j[index] = vel;


#endif
  }
}


template <typename Tpos, typename Tvel>
struct particle
{
public:
  Tpos posx, posy, posz;
  Tvel mass_h2;
  Tvel velx, vely, velz;
  int  pID;
  int padding; 

  Tvel accx, accy, accz;
   //TODO padding incase struct is not multiple of 16 bytes

  __host__ __device__ __forceinline__ void setPosMass(double4 pos) { setPosMass(pos.x, pos.y, pos.z, pos.w); }
  __host__ __device__ __forceinline__ void setVel(double4 vel) {setVel(vel.x, vel.y, vel.z);}
  __host__ __device__ __forceinline__ void setAcc(double4 acc) {setAcc(acc.x, acc.y, acc.z);}

  __host__ __device__ __forceinline__ void setPosMass(double x, double y, double z, double m)
  {
    posx = x; posy = y; posz = z; mass_h2 = m;
  }
  __host__ __device__ __forceinline__ void setVel(double x, double y, double z)
  {
    velx = x; vely = y; velz = z;
  }
  __host__ __device__ __forceinline__ void setAcc(double x, double y, double z)
  {
    accx = x; accy = y; accz = z;
  }
};

template <typename Tpos, typename Tmass>
struct __align__(16) particlePosMass
// struct particlePosMass
{
  Tpos posx, posy, posz;
  Tmass mass_h2;

  __host__ __device__ __forceinline__ void setPosMass(double4 pos) { setPosMass(pos.x, pos.y, pos.z, pos.w); }
  __host__ __device__ __forceinline__ void setPosMass(double x, double y, double z, double m)
  {
    posx = x; posy = y; posz = z; mass_h2 = m;
  }
};



template <typename Tvel>
struct __align__(16) particleVelID
// struct particleVelID
{
  Tvel velx, vely, velz;
  float pID_;  

  __host__ __device__ __forceinline__ void setVel(double4 vel) {setVel(vel.x, vel.y, vel.z);}
  __host__ __device__ __forceinline__ void setVel(double x, double y, double z)
  {
    velx = x; vely = y; velz = z;
  }

  __host__ __device__ __forceinline__ int pID() const {return __float_as_int(pID_);}
  __host__ __device__ __forceinline__ void set_pID(int pID) {pID_ = __int_as_float(pID);}
};


template <typename Tacc>
// struct __align__(16) particleAcc
struct particleAcc
{
  Tacc accx, accy, accz;

  __host__ __device__ __forceinline__      particleAcc(){}
  __host__ __device__ __forceinline__ void setAcc(double4 acc) {setAcc(acc.x, acc.y, acc.z);}
  __host__ __device__ __forceinline__ void setAcc(double x, double y, double z)
  {
    accx = x; accy = y; accz = z;
  }
};


struct dsminAndNNB
{
  float ds2min;
  int   nnb;
};

union dsminUnion
  {
    double x;
    dsminAndNNB dsnnb;
};

template <typename T, typename T4, const bool NGB2>
struct  devForce2
// struct __align__(16) devForce2
{
public:
  T accx, accy, accz;
  T pot;
  T jrkx, jrky, jrkz;
  T snpx, snpy, snpz;

  T ds2min;
  int nnb;

  __device__ devForce2() : accx(0), accy(0), accz(0), pot(0),
                           jrkx(0), jrky(0), jrkz(0) 
                           { 
                             ds2min = 10e10f;
                             nnb = -1;
                            }


  __device__ devForce2(const float v) : accx(v), accy(v), accz(v), pot(v),
                                        jrkx(v), jrky(v), jrkz(v), ds2min(10e10f), nnb(-1) {}

  __device__ __forceinline__ double4 storeAccD4() const {return (double4){accx,accy,accz,pot};}
  __device__ __forceinline__ double4 storeJrkD4() const {return (double4){jrkx, jrky, jrkz, (NGB2 ? ds2min : 0)};}
  __device__ __forceinline__ double4 storeSnpD4() const {return (double4){snpx, snpy, snpz,0};}

  __device__ __forceinline__ T4 storeAcc() const {return (T4){accx,accy,accz,pot};}
  __device__ __forceinline__ T4 storeJrk() const {return (T4){jrkx, jrky, jrkz, NGB2 ? ds2min : 0.0f};}
  __device__ __forceinline__ T4 storeSnp() const {return (T4){snpx, snpy, snpz,0};}


  __device__ __forceinline__ void setnnb(T ds2, int jID)
  {
    if(ds2 < ds2min)
    {
       ds2min = ds2;
       nnb    = jID;
    }
  }
};

// __device__ __forceinline__ double atomicMinNNB(double *address, dsminUnion val)
// {
//     unsigned long long ret  = __double_as_longlong(*address);
//     dsminUnion ret2 = (*(dsminUnion*)(address));
//     while(val.dsnnb.ds2min < ret2.dsnnb.ds2min)
//     {
//         unsigned long long old = ret;
//         if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val.x))) == old)
//             break;
//         ret2 = (*(dsminUnion*)(&ret));
//     }
//     return __longlong_as_double(ret);
// }

#if 1

template<typename outType, typename Tpos, typename Tvel,  const bool doNGB, const bool doNGBList,
         const int integrationOrder>
__device__ __forceinline__ void body_body_interaction(
                                      outType                           &outVal,
                                      inout int                         *ngb_list,
                                      int                               &n_ngb,
                                      particle<Tpos, Tvel>              iParticle,
                                      const particlePosMass<Tpos, Tvel> jpos,
                                      const particleVelID<Tvel>         jVelID,
                                      const particleAcc<Tvel>           jAcc,
                                      const Tvel                        &EPS2) 
{

  if (iParticle.pID != jVelID.pID() || integrationOrder == GRAPE5)    /* assuming we always need ngb */
  {
    const Tvel drx = (jpos.posx - iParticle.posx);
    const Tvel dry = (jpos.posy - iParticle.posy);
    const Tvel drz = (jpos.posz - iParticle.posz);
    const Tvel ds2 = drx*drx + dry*dry + drz*drz;

    if(doNGB)
    {     
      outVal.setnnb(ds2, jVelID.pID());
    }

    if(doNGBList)
    {
      #if ((NGB_PB & (NGB_PB - 1)) != 0)
        #error "NGB_PB is not a power of 2!"
      #endif

      /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
      if(iParticle.mass_h2 > ds2)
      {
        ngb_list[n_ngb & (NGB_PB-1)] = jVelID.pID();
        n_ngb++;
      }
    }

    const Tvel inv_ds = RSQRT(ds2+EPS2);

//     Tvel inv_ds = RSQRT(ds2+EPS2);     
//     const double inv_ds = rsqrtf(ds2+EPS2);   
//     inv_ds = isnan(inv_ds) ? 0 : inv_ds;

    const Tvel minvr1 = jpos.mass_h2 * inv_ds; 
    const Tvel invr2  = inv_ds       * inv_ds; 
    const Tvel minvr3 = minvr1       * invr2;

    const Tvel factor1 = -1.0;
    const Tvel factor2 = -3.0;
    const Tvel factor3 =  2.0;

    // 3*4 + 3 = 15 FLOP, acceleration and potential
    outVal.accx   += minvr3   * drx;
    outVal.accy   += minvr3   * dry;
    outVal.accz   += minvr3   * drz;
    outVal.pot    += (factor1)* minvr1;

    if(integrationOrder == GRAPE5) return;
    
//     asm("//SAPPORO > GRAPE5");

    //Jerk
    const Tvel dvx = jVelID.velx - iParticle.velx;
    const Tvel dvy = jVelID.vely - iParticle.vely;
    const Tvel dvz = jVelID.velz - iParticle.velz;
  
    if(integrationOrder == FOURTH)
    {
      const Tvel drdv = (factor2) * (minvr3*invr2) * (drx*dvx + dry*dvy + drz*dvz);
      outVal.jrkx    += minvr3 * dvx + drdv * drx;  
      outVal.jrky    += minvr3 * dvy + drdv * dry;
      outVal.jrkz    += minvr3 * dvz + drdv * drz;    
      return;
    }

//     asm("//SAPPORO > FOURTH");

    const Tvel dax = jAcc.accx - iParticle.accx;
    const Tvel day = jAcc.accy - iParticle.accy;
    const Tvel daz = jAcc.accz - iParticle.accz;

    const Tvel  v2  = (dvx*dvx) + (dvy*dvy) + (dvz*dvz);
    const Tvel  ra  = (drx*dax) + (dry*day) + (drz*daz);

    Tvel alpha = (drx*dvx + dry*dvy + drz*dvz) * invr2;
    Tvel beta  = (v2 + ra) * invr2 + alpha * alpha;


    alpha      *= factor2;    
    const Tvel jerkx  = (minvr3 * dvx) + alpha * (minvr3 * drx);
    const Tvel jerky  = (minvr3 * dvy) + alpha * (minvr3 * dry);
    const Tvel jerkz  = (minvr3 * dvz) + alpha * (minvr3 * drz);

    //Snap
    alpha       *= factor3;
    beta        *= factor2;
    outVal.snpx += (minvr3 * dax) + alpha * jerkx + beta * (minvr3 * drx);
    outVal.snpy += (minvr3 * day) + alpha * jerky + beta * (minvr3 * dry);
    outVal.snpz += (minvr3 * daz) + alpha * jerkz + beta * (minvr3 * drz);
    
    outVal.jrkx += jerkx;
    outVal.jrky += jerky;
    outVal.jrkz += jerkz;
  }
}

template<typename posType, typename T, typename T3, typename T4, const bool doNGB, 
        const bool doNGBList, const int integrationOrder>
__device__  __forceinline__ void dev_evaluate_gravity_reduce_template_dev(
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i,
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    const int        *id_i,
    __out float2     *dsminNNB,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    const  double4   *acc_i_in,   
    const  double4   *acc_j)
{
  const int tx  = threadIdx.x;
  const int ty  = threadIdx.y;
  const int bx  =  blockIdx.x;
  const int Dim =  blockDim.x*blockDim.y;

//   __shared__ particlePosMass<posType,T>  shared_posx[256];
//   __shared__ particleVelID<T>            shared_velid[256];
//   __shared__ particleAcc<T>              shared_jacc[256]; 

  extern __shared__ char* shared_mem[];  
  particlePosMass<posType,T> *shared_posx  = ( particlePosMass<posType,T>*)&shared_mem[0];
  particleVelID<T>           *shared_velid = ( particleVelID<T>*)&shared_posx[integrationOrder > GRAPE5 ? Dim : 0];
  particleAcc<T>             *shared_jacc  = ( particleAcc<T>*)&shared_velid[integrationOrder > FOURTH ? Dim : 0];

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const T EPS2 = (T)EPS2_d;
  
  particle<posType, T> iParticle;

  iParticle.setPosMass(pos_i[tx+ni_offset]);
  if(integrationOrder > GRAPE5)
  {
    iParticle.setVel    (vel_i[tx+ni_offset]);
    iParticle.pID  =      id_i[tx+ni_offset];
  }

TODO set acc

  const T LARGEnum = 1.0e10f;

  devForce2<T, T4, doNGB> out2;

  int tile       = 0;
  int ni         = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) {
      shared_posx [addr].setPosMass(pos_j[i + tx]);
      
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel (vel_j[i + tx]);
        shared_velid[addr].set_pID( id_j[i + tx]);

        if(integrationOrder > FOURTH)
        {
          shared_jacc[addr].setAcc(acc_j[i + tx]);
        }
      }
    } else {
      shared_posx [addr].setPosMass(LARGEnum,LARGEnum,LARGEnum,0);  
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel    (0.0,0.0,0.0);
        shared_velid[addr].set_pID   (-1);
        
        if(integrationOrder > FOURTH)
        {
          shared_jacc[addr].setAcc  (0.0,0.0,0.0);
        }
      }          
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction <devForce2<T, T4, doNGB>, posType,T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k], shared_jacc[offy+k],
          EPS2);


    for (int k = j1; k < j; k++) 
      body_body_interaction <devForce2<T, T4, doNGB>, posType, T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k], shared_jacc[offy+k], 
          EPS2);

    __syncthreads();

    tile++;
  } //end while

  const int addr   = offy + tx;
#if 1
  //Reduce acceleration and jerk. We know that this has enough
  //space to do in shmem because of design of shmem allocation
  T4 *shared_acc   = (T4*)&shared_posx[0];
  T4 *shared_jrk   = (T4*)&shared_acc[Dim];
//   T4 *shared_snp   = (T4*)&shared_jrk[integrationOrder > FOURTH ? Dim : 0];
  T3 *shared_snp   = (T3*)&shared_jrk[integrationOrder > FOURTH ? Dim : 0];

  shared_acc[addr] = out2.storeAcc();

  if(integrationOrder > GRAPE5)
  {
    shared_jrk[addr] = out2.storeJrk();
    if(integrationOrder > FOURTH)
    {
      //shared_snp[addr] = out2.storeSnp();
      shared_snp[addr].x = out2.snpx;
      shared_snp[addr].y = out2.snpy;
      shared_snp[addr].z = out2.snpz;
    }
  }

  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim.x; i < Dim; i += blockDim.x)
    {
      out2.accx += shared_acc[i + tx].x;
      out2.accy += shared_acc[i + tx].y;
      out2.accz += shared_acc[i + tx].z;
      out2.pot  += shared_acc[i + tx].w;
      
      if(integrationOrder > GRAPE5)
      {
        out2.jrkx += shared_jrk[i + tx].x;
        out2.jrky += shared_jrk[i + tx].y;
        out2.jrkz += shared_jrk[i + tx].z;

        if(integrationOrder > FOURTH)
        {
          out2.snpx += shared_snp[i + tx].x;
          out2.snpx += shared_snp[i + tx].y;
          out2.snpx += shared_snp[i + tx].z;            
        }
      }
    }
  }
  __syncthreads();
#endif

  //Reduce neighbours info
  int    *shared_ngb = (int*)&shared_posx[Dim];
  int    *shared_ofs = (int*)&shared_ngb[Dim];

#if 1
  if(doNGB || doNGBList)
  {
    int    *shared_nid = (int*)&shared_ofs[Dim];
    float  *shared_ds  = (float*)&shared_nid[Dim];

    shared_ngb[addr] = n_ngb;
    shared_ofs[addr] = 0;

    shared_ds [addr] = out2.ds2min;
    shared_nid[addr] = out2.nnb;

    __syncthreads();

    if (ty == 0)
    {
      for (int i = blockDim.x; i < Dim; i += blockDim.x)
      {
        const int addr2 = i + tx;

        if(doNGB)
        {
          if(shared_ds [addr2]  < out2.ds2min) 
          {
            out2.nnb    = shared_nid[addr2];
            out2.ds2min = shared_ds [addr2];
          }       
        }
        
        if(doNGBList)
        {
          shared_ofs[addr2] = min(n_ngb, NGB_PB);
          n_ngb           += shared_ngb[addr2];
        }
      }

      if(doNGBList)
        n_ngb  = min(n_ngb, NGB_PB);
    }
    __syncthreads();
  }
#endif //NGB info

  int ngbListStart = 0;

  double4 *acc_i = &result_i[0];
  double4 *jrk_i = &result_i[ni_total];
  double4 *snp_i = &result_i[ni_total*2];


  if (ty == 0) 
  {
#if 0 //Atomic section
    int *atomicVal  = ngb_count_i;
    float *waitList = (float*)&shared_posx;
    if(threadIdx.x == 0)
    {
      int res          = atomicExch(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
        waitList[0] = (float)waitCounter;
        res = atomicExch(&atomicVal[0], 1); 
      }
    }
    __syncthreads();

    //Convert results to double and write    
    double4 temp = out2.storeAccD4();
    double4 jrk = out2.storeJrkD4();
    acc_i[   tx+ni_offset].x   += temp.x;
    acc_i[   tx+ni_offset].y   += temp.y;
    acc_i[   tx+ni_offset].z   += temp.z;
    acc_i[   tx+ni_offset].w   += temp.w;

    jrk_i[   tx+ni_offset].x   += jrk.x;
    jrk_i[   tx+ni_offset].y   += jrk.y;
    jrk_i[   tx+ni_offset].z   += jrk.z;
    
    if(doNGB)
    {
      union
      {
        double x;
        float2 y;
      } temp2; 
      temp2.y = dsminNNB[tx+ni_offset];
      if(out2.ds2min <  temp2.y.y)
      {
        temp2.y.y = out2.ds2min;
        temp2.y.x = __int_as_float(out2.nnb);

        dsminNNB[tx+ni_offset] = temp2.y;

      }
    }
    if(doNGBList)
    {
      ngbListStart = ngb_count_i[tx+ni_offset];
      ngb_count_i[tx+ni_offset] += n_ngb;
    }


    if(threadIdx.x == 0)
    {
      atomicExch(&atomicVal[0], 0); //Release the lock
    }
    
#else
    double4 temp = out2.storeAccD4();

    atomicAdd(&acc_i[tx+ni_offset].x, temp.x);
    atomicAdd(&acc_i[tx+ni_offset].y, temp.y);
    atomicAdd(&acc_i[tx+ni_offset].z, temp.z);
    atomicAdd(&acc_i[tx+ni_offset].w, temp.w);

    if(integrationOrder > GRAPE5)
    {
      double4 jrk  = out2.storeJrkD4();
      atomicAdd(&jrk_i[tx+ni_offset].x, jrk.x);
      atomicAdd(&jrk_i[tx+ni_offset].y, jrk.y);
      atomicAdd(&jrk_i[tx+ni_offset].z, jrk.z);

      if(integrationOrder > FOURTH)
      {
        double4 snp  = out2.storeSnpD4();
        atomicAdd(&snp_i[tx+ni_offset].x, snp.x);
        atomicAdd(&snp_i[tx+ni_offset].y, snp.y);
        atomicAdd(&snp_i[tx+ni_offset].z, snp.z);
      }
    }

    if(doNGB)
    {
      //Use a double to encode the float distance and int neighbour ID
      union { double x; float2 y; } temp2; 

      temp2.y.y = out2.ds2min;
      temp2.y.x = __int_as_float(out2.nnb);

      atomicMin((double*)&dsminNNB[tx+ni_offset], temp2.x);
    }
    //Prefix summing for neighbour list
    if(doNGBList)
    {
      ngbListStart = atomicAdd(&ngb_count_i[tx+ni_offset],n_ngb);
    }
#endif
  } //ty == 0

  //Write the neighbour list, this blocks start-offset = ngbListStart
  if(doNGBList)
  {
    //Share ngbListStart with other threads in the block
    const int yBlockOffset = shared_ofs[addr];
    __syncthreads();
    if(ty == 0)
    {
      shared_ofs[threadIdx.x] = ngbListStart;
    }
    __syncthreads();
    ngbListStart    = shared_ofs[threadIdx.x];


    int startList   = (ni_offset + tx)  * NGB_PB;
    int prefixSum   = ngbListStart + yBlockOffset; //this blocks offset + y-block offset
    int startWrite  = startList    + prefixSum; 

    if(prefixSum + shared_ngb[addr] < NGB_PB) //Only write if we don't overflow
    {
      for (int i = 0; i < shared_ngb[addr]; i++) 
      {
         ngb_list[startWrite + i] = local_ngb_list[i];
      }
    }
  }//doNGBList
}



#endif

#define CALL( PRECISION1, PRECISION2, PRECISION3, PRECISION4, DONGB, DONGBLIST, ORDER ) { \
    dev_evaluate_gravity_reduce_template_dev<PRECISION1, PRECISION2, PRECISION3, PRECISION4, DONGB, DONGBLIST, ORDER>( \
        nj_total,  \
        nj, \
        ni_offset,\
        ni_total,\
        pos_j, \
        pos_i, \
        result_i, \
        EPS2_d, \
        vel_j, \
        id_j,\
        vel_i,\
        id_i, \
        ds2min_i,\
        ngb_count_i, \
        ngb_list, \
        acc_i, \
        acc_j); \
} \


#define CUDA_GLOBAL( FUNCNAME,  PRECISION1, PRECISION2, PRECISION3, PRECISION4, DONGB, DONGBLIST, ORDER) \
  extern "C" __global__ void \
      FUNCNAME ( \
    const int        nj_total,\
    const int        nj,\
    const int        ni_offset,\
    const int        ni_total,\
    const double4    *pos_j,\
    const double4    *pos_i,\
    __out double4    *result_i,\
    const double     EPS2_d,\
    const double4    *vel_j,\
    const int        *id_j,\
    const double4    *vel_i,\
    const int        *id_i,\
    __out float2     *ds2min_i,\
    __out int        *ngb_count_i,\
    __out int        *ngb_list,\
    const double4    *acc_i,\
    const double4    *acc_j)\
{\
CALL( PRECISION1, PRECISION2, PRECISION3, PRECISION4, DONGB, DONGBLIST, ORDER); \
} \


//Second order, float precision, no neighbour info
CUDA_GLOBAL ( dev_evaluate_gravity_second_float, float, float, float3, float4, false, false, GRAPE5 );
//Second order, default GRAPE precision. No neighbour info
CUDA_GLOBAL ( dev_evaluate_gravity_second_DS, DSX, float, float3, float4, false, false, GRAPE5);
//Fourth order, default GRAPE precision. Including nearest neighbour and neighbour list
CUDA_GLOBAL ( dev_evaluate_gravity_fourth_DS, DSX, float, float3, float4, true, true, FOURTH );
//Fourth order, full native double precision. Including nearest neighbour and neighbour list
CUDA_GLOBAL ( dev_evaluate_gravity_fourth_double, double, double, double3, double4, true, true, FOURTH );
//Sixth order, full native double precision. Including nearest neighbour and neighbour list
CUDA_GLOBAL ( dev_evaluate_gravity_sixth_double, double, double, double3, double4, true, true, SIXTH);

extern "C" __global__ void
// __launch_bounds__(NTHREADS)
dev_evaluate_gravity_reduce_templatedfdf(
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i, 
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    const int        *id_i,
    __out float2     *ds2min_i,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    const double4    *acc_i,   
    const double4    *acc_j) 
{
  
//     CALL( DSX, float, float3, float4, true, true, FOURTH);
//   if(predefinedMethod == GRAPE6){
// //     asm("//JBX");
//     CALL( DSX, float, float3, float4, true, true, FOURTH);
// //     asm("//JBXX");
//   }
//   else if(predefinedMethod == GRAPE6DP)
//   {
//     asm("//JBY");
//     CALL( double, double, double3, double4, true, true, FOURTH);
//     asm("//JBYY");
//   }


//   switch(predefinedMethod)
//   {
//     case GRAPE5SP:
//       CALL( float, float, float3, float4, false, false, GRAPE5);
//       break;
//     case GRAPE5DS:
//       CALL( DSX, float, float3, float4, false, false, GRAPE5);
//       break;
//     case GRAPE6:
//       CALL( DSX, float, float3, float4, true, true, FOURTH);
//       break;
//     case GRAPE6DP:
//       CALL( double, double, double3, double4, true, true, FOURTH);
//       break;
//     case HERMITE6:
//       CALL( double, double, double3, double4, true, true, SIXTH);
//       break;
//     break;
//     default:
//       CALL( DSX, float, float3, float4, true, true, FOURTH);
//       break;
//   }
  

 CALL( DSX, float, float3, float4, true, true, FOURTH);

    //tempalte: type of position, type of acceleration/float/etc, acc type * 4, doNGB, doNGBlist
//    dev_evaluate_gravity_reduce_template_dev<DSX, float, float4, true, true, FOURTH>(
//     dev_evaluate_gravity_reduce_template_dev<double, double, double4, true, true, SIXTH>(

 //   dev_evaluate_gravity_reduce_template_dev<DSX, float, float3, float4, true, true, FOURTH>(
//    dev_evaluate_gravity_reduce_template_dev<DSX, float, float3, float4, true, true, FOURTH>(
//                                             nj_total, 
//                                             nj,
//                                             ni_offset,
//                                             ni_total,
//                                             pos_j, 
//                                             pos_i,
//                                             result_i, 
//                                             EPS2_d,
//                                             vel_j,
//                                             id_j,                                     
//                                             vel_i,      
//                                             id_i,
//                                             ds2min_i,
//                                             ngb_count_i,
//                                             ngb_list,
//                                             acc_i,
//                                             acc_j);

}

extern "C" __global__ void
// __launch_bounds__(NTHREADS)
dev_evaluate_gravity_reduce_template5(
    const int        predefinedMethod, 
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i, 
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    const int        *id_i,
    __out float2     *ds2min_i,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    const double4    *acc_i,   
    const double4    *acc_j) 
{
  
//     CALL( DSX, float, float3, float4, true, true, FOURTH);
//   if(predefinedMethod == GRAPE6){
// //     asm("//JBX");
//     CALL( DSX, float, float3, float4, true, true, FOURTH);
// //     asm("//JBXX");
//   }
//   else if(predefinedMethod == GRAPE6DP)
//   {
//     asm("//JBY");
//     CALL( double, double, double3, double4, true, true, FOURTH);
//     asm("//JBYY");
//   }


//   switch(predefinedMethod)
//   {
//     case GRAPE5SP:
//       CALL( float, float, float3, float4, false, false, GRAPE5);
//       break;
//     case GRAPE5DS:
//       CALL( DSX, float, float3, float4, false, false, GRAPE5);
//       break;
//     case GRAPE6:
//       CALL( DSX, float, float3, float4, true, true, FOURTH);
//       break;
//     case GRAPE6DP:
//       CALL( double, double, double3, double4, true, true, FOURTH);
//       break;
//     case HERMITE6:
//       CALL( double, double, double3, double4, true, true, SIXTH);
//       break;
//     break;
//     default:
//       CALL( DSX, float, float3, float4, true, true, FOURTH);
//       break;
//   }
  

  CALL( DSX, float, float3, float4, true, true, FOURTH);


}




extern "C" __global__ void
dev_reset_buffers(
    const int         ni_total,
    const int         doNGB,
    const int         doNGBList,
    const int         integrationOrder,
    __out double4    *result_i, 
    __out float2     *ds_i,
    __out int        *ngb_count_i)
{
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;


  if(idx >= ni_total) return;

  if(doNGB)
  {
    ds_i[idx] = make_float2(-1, 10e10f);
  }
  if(doNGBList)
  {
    ngb_count_i[idx] = 0;
  }

  double4 reset = make_double4(0,0,0,0);
  result_i[idx] = reset; //Acceleration

  if(integrationOrder >= FOURTH)
    result_i[idx+ni_total] = reset; //Jrk

  if(integrationOrder >= SIXTH)
    result_i[idx+ni_total*2] = reset; //Snp

}

    
#if 0



template<typename outType, typename Tpos, typename Tvel,  const bool doNGB, const bool doNGBList,
         const int integrationOrder>
__device__ __forceinline__ void body_body_interaction2(
                                      outType                           &outVal,
                                      inout int                         *ngb_list,
                                      int                               &n_ngb,
                                      particle<Tpos, Tvel>              &iParticle,
                                      const particlePosMass<Tpos, Tvel> jpos,
                                      const particleVelID<Tvel>         jVelID,                                    
                                      const Tvel                        &EPS2) 
{

  if (iParticle.pID != jVelID.pID() || integrationOrder == GRAPE5)    /* assuming we always need ngb */
  {
    const float drx = (jpos.posx - iParticle.posx);
    const float dry = (jpos.posy - iParticle.posy);
    const float drz = (jpos.posz - iParticle.posz);
    const float ds2 = drx*drx + dry*dry + drz*drz;

    if(doNGB)
    {     
      outVal.setnnb(ds2, jVelID.pID());
    }

    if(doNGBList)
    {
      #if ((NGB_PB & (NGB_PB - 1)) != 0)
        #error "NGB_PB is not a power of 2!"
      #endif

      /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
      if(iParticle.mass_h2 > ds2)
      {
        ngb_list[n_ngb & (NGB_PB-1)] = jVelID.pID();
        n_ngb++;
      }
    }

    const float inv_ds = rsqrtf(ds2+EPS2);

    const float mass   = jpos.mass_h2;
    const float minvr1 = mass*inv_ds; 
    const float  invr2 = inv_ds*inv_ds; 
    const float minvr3 = minvr1*invr2;

    const float factor1 = -1.0;
    const float factor2 = -3.0;
    const float factor3 =  2.0;

    // 3*4 + 3 = 15 FLOP, acceleration and potential
    outVal.accx   += minvr3   * drx;
    outVal.accy   += minvr3   * dry;
    outVal.accz   += minvr3   * drz;
    outVal.pot    += (-1.0f)* minvr1;


    //Jerk
    const float dvx = jVelID.velx - iParticle.velx;
    const float dvy = jVelID.vely - iParticle.vely;
    const float dvz = jVelID.velz - iParticle.velz;

    const float drdv = (factor2) * (minvr3*invr2) * (drx*dvx + dry*dvy + drz*dvz);
    outVal.jrkx    += minvr3 * dvx + drdv * drx;  
    outVal.jrky    += minvr3 * dvy + drdv * dry;
    outVal.jrkz    += minvr3 * dvz + drdv * drz;    
  }
}
template<typename posType, typename T, typename T3, typename T4, const bool doNGB, 
        const bool doNGBList, const int integrationOrder>
__device__  __forceinline__ void dev_evaluate_gravity_reduce_template_dev2(
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i,
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    const int        *id_i,
    __out float2     *dsminNNB,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    const  double4   *acc_i_in,   
    const  double4   *acc_j)
{
  const int tx  = threadIdx.x;
  const int ty  = threadIdx.y;
  const int bx  =  blockIdx.x;
  const int Dim =  blockDim.x*blockDim.y;

//   __shared__ particlePosMass<posType,T>  shared_posx[256];
//   __shared__ particleVelID<T>            shared_velid[256];
//   __shared__ particleAcc<T>              shared_jacc[256]; 

  extern __shared__ char* shared_mem[];  
  particlePosMass<posType,T> *shared_posx  = ( particlePosMass<posType,T>*)&shared_mem[0];
  particleVelID<T>           *shared_velid = ( particleVelID<T>*)&shared_posx[Dim];

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const T EPS2 = (T)EPS2_d;
  
  particle<posType, T> iParticle;

  iParticle.setPosMass(pos_i[tx+ni_offset]);
  iParticle.setVel    (vel_i[tx+ni_offset]);
  iParticle.pID  =      id_i[tx+ni_offset];

  const T LARGEnum = 1.0e10f;

  devForce2<T, T4, doNGB> out2;

  int tile       = 0;
  int ni         = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) {
      shared_posx [addr].setPosMass(pos_j[i + tx]);
      
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel (vel_j[i + tx]);
        shared_velid[addr].set_pID( id_j[i + tx]);
      }
    } else {
      shared_posx [addr].setPosMass(LARGEnum,LARGEnum,LARGEnum,0);  
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel    (0.0,0.0,0.0);
        shared_velid[addr].set_pID   (-1);
      }          
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction2 <devForce2<T, T4, doNGB>, posType,T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k],
          EPS2);


    for (int k = j1; k < j; k++) 
      body_body_interaction2 <devForce2<T, T4, doNGB>, posType, T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k], 
          EPS2);

    __syncthreads();

    tile++;
  } //end while


#if 1
  //Reduce acceleration and jerk. We know that this has enough
  //space to do in shmem because of design of shmem allocation
  T4 *shared_acc   = (T4*)&shared_posx[0];
  T4 *shared_jrk   = (T4*)&shared_acc[Dim];
 
  const int addr   = offy + tx;
  shared_acc[addr] = out2.storeAcc();

  if(integrationOrder > GRAPE5)
  {
    shared_jrk[addr] = out2.storeJrk();  
  }

  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim.x; i < Dim; i += blockDim.x)
    {
      out2.accx += shared_acc[i + tx].x;
      out2.accy += shared_acc[i + tx].y;
      out2.accz += shared_acc[i + tx].z;
      out2.pot  += shared_acc[i + tx].w;
      
      if(integrationOrder > GRAPE5)
      {
        out2.jrkx += shared_jrk[i + tx].x;
        out2.jrky += shared_jrk[i + tx].y;
        out2.jrkz += shared_jrk[i + tx].z;
      }
    }
  }
  __syncthreads();
#endif

  //Reduce neighbours info
  int    *shared_ngb = (int*)&shared_posx[Dim];
  int    *shared_ofs = (int*)&shared_ngb[Dim];

#if 1
  if(doNGB || doNGBList)
  {
    int    *shared_nid = (int*)&shared_ofs[Dim];
    float  *shared_ds  = (float*)&shared_nid[Dim];

    shared_ngb[addr] = n_ngb;
    shared_ofs[addr] = 0;

    shared_ds [addr] = out2.ds2min;
    shared_nid[addr] = out2.nnb;

    __syncthreads();

    if (ty == 0)
    {
      for (int i = blockDim.x; i < Dim; i += blockDim.x)
      {
        const int addr2 = i + tx;

        if(doNGB)
        {
          if(shared_ds [addr2]  < out2.ds2min) 
          {
            out2.nnb    = shared_nid[addr2];
            out2.ds2min = shared_ds [addr2];
          }       
        }
        
        if(doNGBList)
        {
          shared_ofs[addr2] = min(n_ngb, NGB_PB);
          n_ngb           += shared_ngb[addr2];
        }
      }

      if(doNGBList)
        n_ngb  = min(n_ngb, NGB_PB);
    }
    __syncthreads();
  }
#endif //NGB info

  int ngbListStart = 0;

  double4 *acc_i = &result_i[0];
  double4 *jrk_i = &result_i[ni_total];
  double4 *snp_i = &result_i[ni_total*2];


// if(threadIdx.x > 256)
  if (ty == 0) 
  {
#if 0 //Atomic section
    int *atomicVal  = ngb_count_i;
    float *waitList = (float*)&shared_posx;
    if(threadIdx.x == 0)
    {
      int res          = atomicExch(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
        waitList[0] = (float)waitCounter;
        res = atomicExch(&atomicVal[0], 1); 
      }
    }
    __syncthreads();

    //Convert results to double and write    
    double4 temp = out2.storeAccD4();
    double4 jrk = out2.storeJrkD4();
    acc_i[   tx+ni_offset].x   += temp.x;
    acc_i[   tx+ni_offset].y   += temp.y;
    acc_i[   tx+ni_offset].z   += temp.z;
    acc_i[   tx+ni_offset].w   += temp.w;

    jrk_i[   tx+ni_offset].x   += jrk.x;
    jrk_i[   tx+ni_offset].y   += jrk.y;
    jrk_i[   tx+ni_offset].z   += jrk.z;
    
    if(doNGB)
    {
      union
      {
        double x;
        float2 y;
      } temp2; 
      temp2.y = dsminNNB[tx+ni_offset];
      if(out2.ds2min <  temp2.y.y)
      {
        temp2.y.y = out2.ds2min;
        temp2.y.x = __int_as_float(out2.nnb);

        dsminNNB[tx+ni_offset] = temp2.y;

      }
    }
    if(doNGBList)
    {
      ngbListStart = ngb_count_i[tx+ni_offset];
      ngb_count_i[tx+ni_offset] += n_ngb;
    }


    if(threadIdx.x == 0)
    {
      atomicExch(&atomicVal[0], 0); //Release the lock
    }
    
#else
    double4 temp = out2.storeAccD4();

    atomicAdd(&acc_i[tx+ni_offset].x, temp.x);
    atomicAdd(&acc_i[tx+ni_offset].y, temp.y);
    atomicAdd(&acc_i[tx+ni_offset].z, temp.z);
    atomicAdd(&acc_i[tx+ni_offset].w, temp.w);

    if(integrationOrder > GRAPE5)
    {
      double4 jrk  = out2.storeJrkD4();
      atomicAdd(&jrk_i[tx+ni_offset].x, jrk.x);
      atomicAdd(&jrk_i[tx+ni_offset].y, jrk.y);
      atomicAdd(&jrk_i[tx+ni_offset].z, jrk.z);

      if(integrationOrder > FOURTH)
      {
        double4 snp  = out2.storeSnpD4();
        atomicAdd(&snp_i[tx+ni_offset].x, snp.x);
        atomicAdd(&snp_i[tx+ni_offset].y, snp.y);
        atomicAdd(&snp_i[tx+ni_offset].z, snp.z);
      }
    }

    if(doNGB)
    {
      union
      {
        double x;
        float2 y;
      } temp2; 

      temp2.y.y = out2.ds2min;
      temp2.y.x = __int_as_float(out2.nnb);

      atomicMin((double*)&dsminNNB[tx+ni_offset], temp2.x);
    }
    //Prefix summing for neighbour list
    if(doNGBList)
    {
      ngbListStart = atomicAdd(&ngb_count_i[tx+ni_offset],n_ngb);
    }
#endif
  } //ty == 0



  //Write the neighbour list, this blocks start-offset = ngbListStart
  if(doNGBList)
  {
    //Share ngbListStart with other threads in the block
    const int yBlockOffset = shared_ofs[addr];
    __syncthreads();
    if(ty == 0)
    {
      shared_ofs[threadIdx.x] = ngbListStart;
    }
    __syncthreads();
    ngbListStart    = shared_ofs[threadIdx.x];


    int startList   = (ni_offset + tx)  * NGB_PB;
    int prefixSum   = ngbListStart + yBlockOffset; //this blocks offset + y-block offset
    int startWrite  = startList    + prefixSum; 

    if(prefixSum + shared_ngb[addr] < NGB_PB) //Only write if we don't overflow
    {
      for (int i = 0; i < shared_ngb[addr]; i++) 
      {
         ngb_list[startWrite + i] = local_ngb_list[i];
      }
    }
  }//doNGBList
}

#if 0
template<typename posType, typename T, typename T4, const bool doNGB, 
        const bool doNGBList, const int integrationOrder>
__device__ __forceinline__ void dev_evaluate_gravity_reduce_template_dev(
    const int        nj_total, 
    const int        nj,
    const int        ni_offset,
    const int        ni_total,
    const double4    *pos_j, 
    const double4    *pos_i,
    __out double4    *result_i,
    const double     EPS2_d,
    const double4    *vel_j,
    const int        *id_j,                                     
    const double4    *vel_i,                                     
    const int        *id_i,
    __out float2     *dsminNNB,
    __out int        *ngb_count_i,
    __out int        *ngb_list,
    const  double4   *acc_i_in,   
    const  double4   *acc_j)
{
  const int tx  = threadIdx.x;
  const int ty  = threadIdx.y;
  const int bx  =  blockIdx.x;
  const int Dim =  blockDim.x*blockDim.y;

//   __shared__ particlePosMass<posType,T>  shared_posx[256];
//   __shared__ particleVelID<T>            shared_velid[256];
//   __shared__ particleAcc<T>              shared_jacc[256]; //TODO reset to 256
// 
// if(threadIdx.x == 0 && blockIdx.x == 0)
// printf("Size: %ld %ld %ld \n", sizeof(particlePosMass<posType,T>), sizeof(particleVelID<T>), sizeof(particleAcc<T>));

  extern __shared__ char* shared_mem[];  
  particlePosMass<posType,T> *shared_posx  = ( particlePosMass<posType,T>*)&shared_mem[0];
  particleVelID<T>           *shared_velid = ( particleVelID<T>*)&shared_posx[Dim];
  particleAcc<T>             *shared_jacc  = ( particleAcc<T>*)&shared_velid[integrationOrder > FOURTH ? Dim : 0];
  


  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const T EPS2 = (T)EPS2_d;
  
  particle<posType, T> iParticle;

  iParticle.setPosMass(pos_i[tx+ni_offset]);
  iParticle.setVel    (vel_i[tx+ni_offset]);
  iParticle.pID  =      id_i[tx+ni_offset];

  const T LARGEnum = 1.0e10f;

  devForce2<T, T4, doNGB> out2;

  int tile       = 0;
  int ni         = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) {
      shared_posx [addr].setPosMass(pos_j[i + tx]);
      
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel    (vel_j[i + tx]);
        shared_velid[addr].set_pID(   id_j[i + tx]);

        if(integrationOrder > FOURTH)
        {
          shared_jacc[addr].setAcc(acc_j[i + tx]);
        }
      }

    } else {
      shared_posx [addr].setPosMass(LARGEnum,LARGEnum,LARGEnum,0);  
      if(integrationOrder > GRAPE5)
      {
        shared_velid[addr].setVel    (0.0,0.0,0.0);
        shared_velid[addr].set_pID   (-1);
        
        if(integrationOrder > FOURTH)
        {
          shared_jacc[addr].setAcc  (0.0,0.0,0.0);
        }
      }          
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction <devForce2<T, T4, doNGB>, posType,T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k], shared_jacc[offy+k],
          EPS2);


    for (int k = j1; k < j; k++) 
      body_body_interaction <devForce2<T, T4, doNGB>, posType, T,  doNGB, doNGBList, integrationOrder>(
          out2,
          local_ngb_list, n_ngb,
          iParticle,
          shared_posx[offy+k],  shared_velid[offy+k], shared_jacc[offy+k], 
          EPS2);

    __syncthreads();

    tile++;
  } //end while


#if 1
  //Reduce acceleration and jerk. We know that this has enough
  //space to do in shmem because of design of shmem allocation
  T4 *shared_acc   = (T4*)&shared_posx[0];
  T4 *shared_jrk   = (T4*)&shared_acc[Dim];
  T4 *shared_snp   = (T4*)&shared_jrk[integrationOrder > FOURTH ? Dim : 0];

  const int addr   = offy + tx;
  shared_acc[addr] = out2.storeAcc();

  if(integrationOrder > GRAPE5)
  {
    shared_jrk[addr] = out2.storeJrk();
    if(integrationOrder > FOURTH)
    {
      shared_snp[addr] = out2.storeSnp();
    }
  }

  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim.x; i < Dim; i += blockDim.x)
    {
      out2.accx += shared_acc[i + tx].x;
      out2.accy += shared_acc[i + tx].y;
      out2.accz += shared_acc[i + tx].z;
      out2.pot  += shared_acc[i + tx].w;
      
      if(integrationOrder > GRAPE5)
      {
        out2.jrkx += shared_jrk[i + tx].x;
        out2.jrky += shared_jrk[i + tx].y;
        out2.jrkz += shared_jrk[i + tx].z;

        if(integrationOrder > FOURTH)
        {
          out2.snpx += shared_snp[i + tx].x;
          out2.snpx += shared_snp[i + tx].y;
          out2.snpx += shared_snp[i + tx].z;
        }
      }
    }
  }
  __syncthreads();
#endif

  //Reduce neighbours info
  int    *shared_ngb = (int*)&shared_posx[Dim];
  int    *shared_ofs = (int*)&shared_ngb[Dim];

#if 1
  if(doNGB || doNGBList)
  {
    int    *shared_nid = (int*)&shared_ofs[Dim];
    float  *shared_ds  = (float*)&shared_nid[Dim];

    shared_ngb[addr] = n_ngb;
    shared_ofs[addr] = 0;

    shared_ds [addr] = out2.ds2min;
    shared_nid[addr] = out2.nnb;

    __syncthreads();

    if (ty == 0)
    {
      for (int i = blockDim.x; i < Dim; i += blockDim.x)
      {
        const int addr2 = i + tx;

        if(doNGB)
        {
          if(shared_ds [addr2]  < out2.ds2min) 
          {
            out2.nnb    = shared_nid[addr2];
            out2.ds2min = shared_ds [addr2];
          }       
        }
        
        if(doNGBList)
        {
          shared_ofs[addr2] = min(n_ngb, NGB_PB);
          n_ngb           += shared_ngb[addr2];
        }
      }

      if(doNGBList)
        n_ngb  = min(n_ngb, NGB_PB);
    }
    __syncthreads();
  }
#endif

  int ngbListStart = 0;

  double4 *acc_i = &result_i[0];
  double4 *jrk_i = &result_i[ni_total];
  double4 *snp_i = &result_i[ni_total*2];

// if(threadIdx.x < 256)
  if (ty == 0) 
  {
#if 0 //Atomic section
    int *atomicVal  = ngb_count_i;
    float *waitList = (float*)&shared_posx;
    if(threadIdx.x == 0)
    {
      int res          = atomicExch(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
        waitList[0] = (float)waitCounter;
        res = atomicExch(&atomicVal[0], 1); 
      }
    }
    __syncthreads();

    //Convert results to double and write    
    double4 temp = out2.storeAccD4();
    double4 jrk = out2.storeJrkD4();
    acc_i[   tx+ni_offset].x   += temp.x;
    acc_i[   tx+ni_offset].y   += temp.y;
    acc_i[   tx+ni_offset].z   += temp.z;
    acc_i[   tx+ni_offset].w   += temp.w;

    jrk_i[   tx+ni_offset].x   += jrk.x;
    jrk_i[   tx+ni_offset].y   += jrk.y;
    jrk_i[   tx+ni_offset].z   += jrk.z;
    
    if(doNGB)
    {
      union
      {
        double x;
        float2 y;
      } temp2; 
      temp2.y = dsminNNB[tx+ni_offset];
      if(out2.ds2min <  temp2.y.y)
      {
        temp2.y.y = out2.ds2min;
        temp2.y.x = __int_as_float(out2.nnb);

        dsminNNB[tx+ni_offset] = temp2.y;

      }
    }
    if(doNGBList)
    {
      ngbListStart = ngb_count_i[tx+ni_offset];
      ngb_count_i[tx+ni_offset] += n_ngb;
    }


    if(threadIdx.x == 0)
    {
      atomicExch(&atomicVal[0], 0); //Release the lock
    }
    
#else
    double4 temp = out2.storeAccD4();

    atomicAdd(&acc_i[tx+ni_offset].x, temp.x);
    atomicAdd(&acc_i[tx+ni_offset].y, temp.y);
    atomicAdd(&acc_i[tx+ni_offset].z, temp.z);
    atomicAdd(&acc_i[tx+ni_offset].w, temp.w);
//     atomicAdd((float*)&acc_i[tx+ni_offset].x, (float)temp.x);
//     atomicAdd((float*)&acc_i[tx+ni_offset].y, (float)temp.y);
//     atomicAdd((float*)&acc_i[tx+ni_offset].z, (float)temp.z);
//     atomicAdd((float*)&acc_i[tx+ni_offset].w, (float)temp.w);

    if(integrationOrder > GRAPE5)
    {
      double4 jrk  = out2.storeJrkD4();
      atomicAdd(&jrk_i[tx+ni_offset].x, jrk.x);
      atomicAdd(&jrk_i[tx+ni_offset].y, jrk.y);
      atomicAdd(&jrk_i[tx+ni_offset].z, jrk.z);
//     atomicAdd((float*)&jrk_i[tx+ni_offset].x, (float)jrk.x);
//     atomicAdd((float*)&jrk_i[tx+ni_offset].y, (float)jrk.y);
//     atomicAdd((float*)&jrk_i[tx+ni_offset].z, (float)jrk.z);

      if(integrationOrder > FOURTH)
      {
        double4 snp  = out2.storeSnpD4();
        atomicAdd(&snp_i[tx+ni_offset].x, snp.x);
        atomicAdd(&snp_i[tx+ni_offset].y, snp.y);
        atomicAdd(&snp_i[tx+ni_offset].z, snp.z);
//         atomicAdd((float*)&snp_i[tx+ni_offset].x, (float)snp.x);
//         atomicAdd((float*)&snp_i[tx+ni_offset].y, (float)snp.y);
//         atomicAdd((float*)&snp_i[tx+ni_offset].z, (float)snp.z);
      }
    }

//     acc_i[tx+ni_offset].x = temp.x;
//     acc_i[tx+ni_offset].y = temp.y;
//     acc_i[tx+ni_offset].z = temp.z;
//     acc_i[tx+ni_offset].w = temp.w;
//     jrk_i[tx+ni_offset].x = jrk.x;
//     jrk_i[tx+ni_offset].y = jrk.y;
//     jrk_i[tx+ni_offset].z = jrk.z;
//     snp_i[tx+ni_offset].x= snp.x;
//     snp_i[tx+ni_offset].y= snp.y;
//     snp_i[tx+ni_offset].z= snp.z;

    if(doNGB)
    {
      union
      {
        double x;
        float2 y;
      } temp2; 

      temp2.y.y = out2.ds2min;
      temp2.y.x = __int_as_float(out2.nnb);

      atomicMin((double*)&dsminNNB[tx+ni_offset], temp2.x);
    }
    //Prefix summing for neighbour list
    if(doNGBList)
    {
      ngbListStart = atomicAdd(&ngb_count_i[tx+ni_offset],n_ngb);
    }
#endif
  } //ty == 0



  //Write the neighbour list, this blocks start-offset = ngbListStart
  if(doNGBList)
  {
    //Share ngbListStart with other threads in the block
    const int yBlockOffset = shared_ofs[addr];
    __syncthreads();
    if(ty == 0)
    {
      shared_ofs[threadIdx.x] = ngbListStart;
    }
    __syncthreads();
    ngbListStart    = shared_ofs[threadIdx.x];


    int startList   = (ni_offset + tx)  * NGB_PB;
    int prefixSum   = ngbListStart + yBlockOffset; //this blocks offset + y-block offset
    int startWrite  = startList    + prefixSum; 

    if(prefixSum + shared_ngb[addr] < NGB_PB) //Only write if we don't overflow
    {
      for (int i = 0; i < shared_ngb[addr]; i++) 
      {
         ngb_list[startWrite + i] = local_ngb_list[i];
      }
    }
  }//doNGBList
}
#endif
#if 1
static __device__ uint retirementCountBuildNodes = 0;
  const int startOffset = blockIdx.x*ni_total*3;
  acc_i = &result_i [0          + startOffset];
  jrk_i = &result_i [1*ni_total + startOffset];
  snp_i = &result_i [2*ni_total + startOffset];


  const int writeIdx = threadIdx.x+ni_offset;

  if(ty == 0)
  {
    acc_i[writeIdx] = out2.storeAccD4();
    jrk_i[writeIdx] = out2.storeJrkD4();
    snp_i[writeIdx] = out2.storeSnpD4();
  }

  int numBlocks = gridDim.x * gridDim.y;
  if (numBlocks > 1)
  {
    __shared__ bool amLast;

    // Thread 0 takes a ticket
    if(threadIdx.x == 0 && ty == 0)
    {
      unsigned int ticket = atomicInc(&retirementCountBuildNodes, numBlocks);
      //unsigned int ticket = retirementCountBuildNodes++;

      // If the ticket ID is equal to the number of blocks, we are the last block!
      amLast = (ticket == numBlocks-1);
    }
    __syncthreads();

    if( amLast && ty == 0)
    {
        __threadfence();        //Make sure all global memory writes are completed

        acc_i = &result_i[0];
        jrk_i = &result_i[ni_total];
        snp_i = &result_i[ni_total*2];


#if 0
        for(int i=1; i < numBlocks; i++)
        {
          acc_i[writeIdx].x += acc_i[writeIdx+i*ni_total*3].x;
          acc_i[writeIdx].y += acc_i[writeIdx+i*ni_total*3].y;
          acc_i[writeIdx].z += acc_i[writeIdx+i*ni_total*3].z;
          acc_i[writeIdx].w += acc_i[writeIdx+i*ni_total*3].w;  

          jrk_i[writeIdx].x += jrk_i[writeIdx+i*ni_total*3].x;
          jrk_i[writeIdx].y += jrk_i[writeIdx+i*ni_total*3].y;
          jrk_i[writeIdx].z += jrk_i[writeIdx+i*ni_total*3].z;

          snp_i[writeIdx].x += snp_i[writeIdx+i*ni_total*3].x;
          snp_i[writeIdx].y += snp_i[writeIdx+i*ni_total*3].y;
          snp_i[writeIdx].z += snp_i[writeIdx+i*ni_total*3].z; 
        }
#elif 0
    asm("//JB2");
      double4 accBuff[52];
      double4 jrkBuff[52];
      double4 snpBuff[52];
        for(int i=0; i < 52; i++)
        {
          accBuff[i] = acc_i[writeIdx+i*ni_total*3];
        }
        for(int i=0; i < 52; i++)
        {
          jrkBuff[i] = jrk_i[writeIdx+i*ni_total*3];
        }
        for(int i=0; i < 52; i++)
        {
          snpBuff[i] = snp_i[writeIdx+i*ni_total*3];
        }
//         asm("//JB3");
        double4 temp = accBuff[0];
       for(int i=1; i < 52; i++)
        {
          double4 tempACC2 = accBuff[i];
          temp.x += tempACC2.x;
          temp.y += tempACC2.y;
          temp.z += tempACC2.z;
          temp.w += tempACC2.w;  
        }
       acc_i[writeIdx] = temp;

       double4 temp2 = jrkBuff[0];
       for(int i=1; i < 52; i++)
       {
          double4 tempACC2 = jrkBuff[i];
          temp2.x += tempACC2.x;
          temp2.y += tempACC2.y;
          temp2.z += tempACC2.z;
          temp2.w += tempACC2.w;  
       }
       jrk_i[writeIdx] = temp2;
       
       double4 temp3 = snpBuff[0];
       for(int i=1; i < 52; i++)
        {
          double4 tempACC2 = snpBuff[i];
          temp3.x += tempACC2.x;
          temp3.y += tempACC2.y;
          temp3.z += tempACC2.z;
          temp3.w += tempACC2.w;  
        }
        snp_i[writeIdx] = temp3;

    asm("//JBC2");
#elif 0


    asm("//JB1");
        double4 tempACC = acc_i[writeIdx];
        double4 tempJRK = jrk_i[writeIdx];
        double4 tempSNP = snp_i[writeIdx];
        #pragma unroll 
//         for(int i=1; i < 52; i++)
        for(int i=1; i < 52; i++)
        {
          double4 tempACC2 = acc_i[writeIdx+i*ni_total*3];

//           printf("Thread: %d reads from: %d \t offset: %d\n", threadIdx.x, writeIdx+i*ni_total*3, ni_offset);
          tempACC.x += tempACC2.x;
          tempACC.y += tempACC2.y;
          tempACC.z += tempACC2.z;
          tempACC.w += tempACC2.w;  
        }
       for(int i=1; i < 52; i++)
        {
          double4 tempJRK2 = jrk_i[writeIdx+i*ni_total*3];
          
          tempJRK.x += tempJRK2.x;
          tempJRK.y += tempJRK2.y;
          tempJRK.z += tempJRK2.z;
          tempJRK.w += tempJRK2.w;   
      }
       for(int i=1; i < 52; i++)
        {
          double4 tempSNP2 = snp_i[writeIdx+i*ni_total*3];

          tempSNP.x += tempSNP2.x;
          tempSNP.y += tempSNP2.y;
          tempSNP.z += tempSNP2.z;
          tempSNP.w += tempSNP2.w;     
        }
        acc_i[writeIdx] = tempACC;
        jrk_i[writeIdx] = tempJRK;
        snp_i[writeIdx] = tempSNP;
        asm("//JB2");



#else
        asm("//JB1");
//         double4 tempACC = acc_i[writeIdx];
//         double4 tempJRK = jrk_i[writeIdx];
//         double4 tempSNP = snp_i[writeIdx];
//         #pragma unroll 
// //         for(int i=1; i < 52; i++)
//         for(int i=1; i < 52; i++)
//         {
//           double4 tempACC2 = acc_i[writeIdx+i*ni_total*3];
//           double4 tempJRK2 = jrk_i[writeIdx+i*ni_total*3];
//           double4 tempSNP2 = snp_i[writeIdx+i*ni_total*3];
// 
// //           printf("Thread: %d reads from: %d \t offset: %d\n", threadIdx.x, writeIdx+i*ni_total*3, ni_offset);
//           tempACC.x += tempACC2.x;
//           tempACC.y += tempACC2.y;
//           tempACC.z += tempACC2.z;
//           tempACC.w += tempACC2.w;      
// 
//           
//           tempJRK.x += tempJRK2.x;
//           tempJRK.y += tempJRK2.y;
//           tempJRK.z += tempJRK2.z;
//           tempJRK.w += tempJRK2.w;   
// 
//           
//           tempSNP.x += tempSNP2.x;
//           tempSNP.y += tempSNP2.y;
//           tempSNP.z += tempSNP2.z;
//           tempSNP.w += tempSNP2.w;     
// 
//         }
//         acc_i[writeIdx] = tempACC;
//         jrk_i[writeIdx] = tempJRK;
//         snp_i[writeIdx] = tempSNP;
        asm("//JB2");
#endif

        if(threadIdx.x == 0) retirementCountBuildNodes = 0; 
    } //if last block
  } //Numblocks > 1
#elif 1

  const int startOffset = blockIdx.x*ni_total*3;
  const int numBlocks   = gridDim.x * gridDim.y;
  acc_i = &result_i [0          + startOffset];
  jrk_i = &result_i [1*ni_total + startOffset];
  snp_i = &result_i [2*ni_total + startOffset];


  const int writeIdx = (blockIdx.x) + numBlocks*(threadIdx.x+ni_offset);

  if(ty == 0 && threadIdx.x == 0)
  {
    acc_i[writeIdx] = out2.storeAccD4();
    jrk_i[writeIdx] = out2.storeJrkD4();
    snp_i[writeIdx] = out2.storeSnpD4();
  }

  
  if (numBlocks > 1)
  {
    __shared__ bool amLast;

    // Thread 0 takes a ticket
    if(threadIdx.x == 0 && ty == 0)
    {
      unsigned int ticket = atomicInc(&retirementCountBuildNodes, numBlocks);
      //unsigned int ticket = retirementCountBuildNodes++;

      // If the ticket ID is equal to the number of blocks, we are the last block!
      amLast = (ticket == numBlocks-1);
    }
    __syncthreads();

    if( amLast && ty == 0 && threadIdx.x == 0)
    {
        // if(threadIdx.x == 0)
        // {
//           printf("\nLAST BLOCK! : %d %d\t%d\toffset: %d\n", blockIdx.x, blockDim.x, threadIdx.x, ni_offset);
        // }
        __threadfence();        //Make sure all global memory writes are completed


        acc_i = &result_i[0];
        jrk_i = &result_i[ni_total];
        snp_i = &result_i[ni_total*2];

        for(int i=1; i < numBlocks; i++)
        {
          acc_i[writeIdx].x += acc_i[writeIdx+i*ni_total*3].x;
          acc_i[writeIdx].y += acc_i[writeIdx+i*ni_total*3].y;
          acc_i[writeIdx].z += acc_i[writeIdx+i*ni_total*3].z;
          acc_i[writeIdx].w += acc_i[writeIdx+i*ni_total*3].w;       

          jrk_i[writeIdx].x += jrk_i[writeIdx+i*ni_total*3].x;
          jrk_i[writeIdx].y += jrk_i[writeIdx+i*ni_total*3].y;
          jrk_i[writeIdx].z += jrk_i[writeIdx+i*ni_total*3].z;

          snp_i[writeIdx].x += snp_i[writeIdx+i*ni_total*3].x;
          snp_i[writeIdx].y += snp_i[writeIdx+i*ni_total*3].y;
          snp_i[writeIdx].z += snp_i[writeIdx+i*ni_total*3].z;
        }

        if(threadIdx.x == 0) retirementCountBuildNodes = 0; 
    } //if last block
  } //Numblocks > 1
#endif
/*

template<typename outType, typename T,  const bool doNGB, const bool doNGBList>
__device__ __forceinline__ void body_body_interaction(
                                      outType           &outVal,
                                      inout int         *ngb_list,
                                      int               &n_ngb,
                                      particle<DSX, T>  &iParticle,
                                      particle<DSX, T>  &jParticle,
                                      const T            EPS2) 
{

  if (iParticle.pID != jParticle.pID)    /* assuming we always need ngb */
  {
    const T drx = (jParticle.posx - iParticle.posx);
    const T dry = (jParticle.posy - iParticle.posy);
    const T drz = (jParticle.posz - iParticle.posz);

    const T ds2 = drx*drx + dry*dry + drz*drz;

    if(doNGB)
    {     
      outVal.setnnb(ds2, jParticle.pID);
    }

    if(doNGBList)
    {
      #if ((NGB_PB & (NGB_PB - 1)) != 0)
        #error "NGB_PB is not a power of 2!"
      #endif

      /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
      if(iParticle.mass_h2 > ds2)
      {
        ngb_list[n_ngb & (NGB_PB-1)] = jParticle.pID;
        n_ngb++;
      }
    }


    const T inv_ds = rsqrt(ds2+EPS2);

    const T minvr1 = jParticle.mass_h2*inv_ds; 
    const T invr2  = inv_ds*inv_ds; 
    const T minvr3 = minvr1*invr2;

    const T factor1 = -1.0;
    const T factor2 = -3.0;

    // 3*4 + 3 = 15 FLOP
    outVal.accx   += minvr3 * drx;
    outVal.accy   += minvr3 * dry;
    outVal.accz   += minvr3 * drz;
    outVal.pot    += (factor1)*minvr1; //TODO make -1.0 or -1.0f

    const T dvx = jParticle.velx - iParticle.velx;
    const T dvy = jParticle.vely - iParticle.vely;
    const T dvz = jParticle.velz - iParticle.velz;

    const T drdv = (factor2) * (minvr3*invr2) * (drx*dvx + dry*dvy + drz*dvz); //TODO make -3.0 or -3.0f

    outVal.jrkx += minvr3 * dvx + drdv * drx;  
    outVal.jrky += minvr3 * dvy + drdv * dry;
    outVal.jrkz += minvr3 * dvz + drdv * drz;
  }
}


extern "C" __global__ void
//__launch_bounds__(NTHREADS)
dev_evaluate_gravity_reduce_template(
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

  extern __shared__ DS4X shared_posx[];

  __shared__ DS4X   shared_posx[256];
  __shared__ float4 shared_vel [256];
  __shared__ int    shared_id  [256];
  
  const int tx  = threadIdx.x;
  const int ty  = threadIdx.y;
  const int bx  =  blockIdx.x;
  const int Dim = blockDim.x*blockDim.y;

//   float4 *shared_vel = (float4*)&shared_posx[Dim];

  particle<DSX, float> *sharedJ = (particle<DSX, float>*)&shared_posx[0];


  const bool doNGB      = true;
  const bool doNGBList  = false;

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const float EPS2 = (float)EPS2_d;
  
  particle<DSX, float> iParticle;
  particle<DSX, float> jParticle;

  iParticle.setPosMass(pos_i[tx+ni_offset]);
  iParticle.setVel    (vel_i[tx+ni_offset]);
  iParticle.pID  =  id_i[tx+ni_offset];

  const float LARGEnum = 1.0e10f;

  devForce2<float, doNGB> out2(0);

  out2.ds2min = LARGEnum;
  out2.nnb    = -1;

  int tile  = 0;
  int ni    = bx * (nj*blockDim.y) + nj*ty;
  const int offy = blockDim.x*ty;
  for (int i = ni; i < ni+nj; i += blockDim.x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      jParticle.setPosMass(pos_j[i + tx]);
      jParticle.setVel    (vel_j[i + tx]);
      jParticle.pID  =      id_j[i + tx];
      sharedJ[addr]  = jParticle;

      shared_posx[addr].x = jParticle.posx;



    } else {
      jParticle.setPosMass(LARGEnum,LARGEnum,LARGEnum,0);      
      jParticle.setVel    (0.0,0.0,0.0);
      jParticle.pID  =  -1;
      sharedJ[addr]  = jParticle;
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim.x, blockDim.x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction <devForce2<float, doNGB>, float,  doNGB, doNGBList>(
          out2,
          local_ngb_list, n_ngb,
          iParticle, sharedJ[offy+k],
          EPS2);


    for (int k = j1; k < j; k++) 
      body_body_interaction <devForce2<float, doNGB>, float,  doNGB, doNGBList>(
          out2,
          local_ngb_list, n_ngb,
          iParticle, sharedJ[offy+k],
          EPS2);

    __syncthreads();

    tile++;
  } //end while

  //Up to here with conversion
  #if 0
  float4 *shared_acc = (float4*)&shared_posx[0];
  float4 *shared_jrk = (float4*)&shared_acc[Dim];

  const int addr = offy + tx;
  shared_acc[addr] = out2.storeAcc();
  shared_jrk[addr] = out2.storeJrk();

  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim.x; i < Dim; i += blockDim.x)
    {
      out2.accx += shared_acc[i + tx].x;
      out2.accy += shared_acc[i + tx].y;
      out2.accz += shared_acc[i + tx].z;
      out2.pot  += shared_acc[i + tx].w;

      out2.jrkx += shared_jrk[i + tx].x;
      out2.jrky += shared_jrk[i + tx].y;
      out2.jrkz += shared_jrk[i + tx].z;
    }
  }
  __syncthreads();

  //Reduce neighbours info
  if(doNGB)
  {
    int    *shared_ngb = (int*   )&shared_posx[Dim];
    int    *shared_ofs = (int*   )&shared_ngb[Dim];
    int    *shared_nid = (int*   )&shared_ofs[Dim];
    float  *shared_ds  = (float* )&shared_nid[Dim];

    shared_ngb[addr] = n_ngb;
    shared_ofs[addr] = 0;
    shared_ds [addr] = out2.ds2min;
    shared_nid[addr] = out2.nnb;

    if (ty == 0)
    {
      for (int i = blockDim.x; i < Dim; i += blockDim.x)
      {
        const int addr = i + tx;
      
        if(shared_ds [addr]  < out2.ds2min) 
        {
          out2.nnb    = shared_nid[addr];
          out2.ds2min = shared_ds [addr];
        }

        shared_ofs[addr] = min(n_ngb, NGB_PB);
        n_ngb           += shared_ngb[addr];
      }
      n_ngb  = min(n_ngb, NGB_PB);
    }
    __syncthreads();
  }
#endif

  if (ty == 0) 
  {
    int *atomicVal = ngb_list;
//     float *waitList = (float*)&shared_posx;
    if(threadIdx.x == 0)
    {
      int res          = atomicExch(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
//         waitList[blockDim.x] = (float)waitCounter;
        res = atomicExch(&atomicVal[0], 1); 
      }
    }
    __syncthreads();
    //Convert results to double and write
    ds2min_i[tx+ni_offset]   = out2.ds2min;
     double4 temp = out2.storeAccD4();
    acc_i[   tx+ni_offset].x   += temp.x;
    acc_i[   tx+ni_offset].y   += temp.y;
    acc_i[   tx+ni_offset].z   += temp.z;
    acc_i[   tx+ni_offset].w   += temp.w;
// 
     double4 jrk = out2.storeJrkD4();
    jrk_i[   tx+ni_offset].x   += jrk.x;
    jrk_i[   tx+ni_offset].y   += jrk.y;
    jrk_i[   tx+ni_offset].z   += jrk.z;
    jrk_i[   tx+ni_offset].w   += jrk.w;


//     atomicAdd((float*)&acc_i[   tx+ni_offset].x, (float)temp.x);
//     atomicAdd((float*)&acc_i[   tx+ni_offset].y, (float)temp.y);
//     atomicAdd((float*)&acc_i[   tx+ni_offset].z, (float)temp.z);
//     atomicAdd((float*)&acc_i[   tx+ni_offset].w, (float)temp.w);
// 
//     atomicAdd((float*)&jrk_i[   tx+ni_offset].x, (float)jrk.x);
//     atomicAdd((float*)&jrk_i[   tx+ni_offset].y, (float)jrk.y);
//     atomicAdd((float*)&jrk_i[   tx+ni_offset].z, (float)jrk.z);
//     atomicAdd((float*)&jrk_i[   tx+ni_offset].w, (float)jrk.w);

//     atomicAdd(&acc_i[   tx+ni_offset].x, temp.x);
//     atomicAdd(&acc_i[   tx+ni_offset].y, temp.y);
//     atomicAdd(&acc_i[   tx+ni_offset].z, temp.z);
//     atomicAdd(&acc_i[   tx+ni_offset].w, temp.w);
// 
//     atomicAdd(&jrk_i[   tx+ni_offset].x, jrk.x);
//     atomicAdd(&jrk_i[   tx+ni_offset].y, jrk.y);
//     atomicAdd(&jrk_i[   tx+ni_offset].z, jrk.z);
//     atomicAdd(&jrk_i[   tx+ni_offset].w, jrk.w);



    ngb_count_i[tx+ni_offset]   = n_ngb;

    if(threadIdx.x == 0)
    {
      atomicExch(&atomicVal[0], 0); //Release the lock
    }
  }*/


//TODO
//   //Write the neighbour list
//   {
//     int offset  = threadIdx.x * gridDim.x*NGB_PB + blockIdx.x * NGB_PB;
//     offset     += shared_ofs[addr];
//     n_ngb       = shared_ngb[addr];
//     for (int i = 0; i < n_ngb; i++) 
//       ngb_list[offset + i] = local_ngb_list[i];
//   }
}
#endif

