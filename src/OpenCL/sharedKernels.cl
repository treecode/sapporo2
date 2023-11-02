

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_amd_fp64: enable

#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)                                                                                                        
#define blockIdx_x  get_group_id(0)                                                                                                                         
#define blockIdx_y  get_group_id(1)                                                                                                                         
#define threadIdx_x get_local_id(0)                                                                                                                         
#define threadIdx_y get_local_id(1)                                                                                                                         
#define gridDim_x   get_num_groups(0)                                                                                                                       
#define gridDim_y   get_num_groups(1)                                                                                                                       
#define blockDim_x  get_local_size(0)                                                                                                                       
#define blockDim_y  get_local_size(1)  



#define NPIPES 16384


enum { GRAPE5   = 0, FOURTH, SIXTH, EIGHT};        //0, 1, 2, 3

#define NGB_PB 256

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
/*  __inline devForce() {}
  __inline devForce(const float v) : x(v), y(v), z(v), w(v) {}
  __inline float4 to_float4() const {return (float4){x,y,z,w};}
  __inline double4 to_double4() const {return (double4){x,y,z,w};}
*/
};

#endif



typedef float2 DS;  // double single;

typedef struct DS4 {
  DS x, y, z, w;
} DS4;
typedef struct DS2 {
  DS x, y;
} DS2;


__inline DS to_DS(double a) {
  DS b;
  b.x = (float)a;
  b.y = (float)(a - b.x);
  return b;
}

__inline double to_double(DS a) {
  double b;
  b = (double)((double)a.x + (double)a.y);
  return b;
}


// This function computes c = a + b.
__inline DS dsaddds(DS a, DS b) {
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
__inline DS dsadd(DS a, float b) {
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




/*

Function to predict the particles
DS version

*/

__kernel void dev_predictor(
                            int                nj,
                            double             t_i,
                            int                integrationOrder,
                            __global  double2 *t_j,
                            __global  double4 *Ppos_j,
                            __global  double4 *pos_j, 
                            __global  double4 *Pvel_j,                            
                            __global  double4 *vel_j,
                            __global  double4 *acc_j,
                            __global  double4 *jrk_j,
                            __global  double4 *Pacc_j,
                            __global  double4 *snp_j,
                            __global  double4 *crk_j)
{
//   int index = blockIdx_x * blockDim_x + threadIdx_x;
  const uint bid = blockIdx_y * gridDim_x + blockIdx_x;
  const uint tid = threadIdx_x;
  const uint index = bid * blockDim_x + tid;
  
  if(integrationOrder == GRAPE5) return;
  
  if (index < nj) {

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

    double4 snp = (double4)(0,0,0,0);
    double4 crk = (double4)(0,0,0,0);

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
  }  
}



__kernel void
dev_reset_buffers(
        const int            ni_total,
        const int            doNGB,
        const int            doNGBList,
        const int            integrationOrder,
        __global double4    *result_i, 
        __global float2     *ds_i,
        __global int        *ngb_count_i)
{
  const uint bid = blockIdx_y * gridDim_x + blockIdx_x;
  const uint tid = threadIdx_x;
  const uint idx = bid * blockDim_x + tid;


  if(idx >= ni_total) return;


  ds_i[idx] = (float2){-1, 10e10f};
  ngb_count_i[idx] = 0;
  

  double4 reset = (double4){0,0,0,0};
  result_i[idx] = reset; //Acceleration

  if(integrationOrder >= FOURTH)
    result_i[idx+ni_total] = reset; //Jrk

  if(integrationOrder >= SIXTH)
    result_i[idx+ni_total*2] = reset; //Snp
    
  if(idx == ni_total-1)
  {
    ngb_count_i[NPIPES]    = 0;
  }
}//reset buffers

/*
 * Function that moves the (changed) j-particles
 * to the correct address location.
*/
__kernel void dev_copy_particles(
                                 int nj, 
                                 int         integrationOrder,
            __global             double4   *pos_j, 
            __global             double4   *pos_j_temp,
            __global             int       *address_j,
            __global             double2   *t_j,
            __global             double4   *Ppos_j,
            __global             double4   *Pvel_j,
            __global             double4   *vel_j,
            __global             double4   *acc_j,
            __global             double4   *jrk_j,
            __global             int       *id_j,
            __global             double2   *t_j_temp,
            __global             double4   *vel_j_temp,
            __global             double4   *acc_j_temp,
            __global             double4   *jrk_j_temp,
            __global             int       *id_j_temp,
            __global             double4   *Pacc_j,
            __global             double4   *snp_j,
            __global             double4   *crk_j,
            __global             double4   *snp_j_temp,
            __global             double4   *crk_j_temp)
{
//   int index = blockIdx_x * blockDim_x + threadIdx_x;
  const uint bid = blockIdx_y * gridDim_x + blockIdx_x;
  const uint tid = threadIdx_x;
  const uint index = bid * blockDim_x + tid;

  //Copy the changed particles
  if (index < nj)
  {
    pos_j[address_j[index]] = pos_j_temp[index];
    Ppos_j[address_j[index]] = pos_j_temp[index];

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

