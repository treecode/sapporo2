/*

Sapporo 2 device kernels

Version 1.1

Template based, contains default kernels for:
GRAPE5 / Second order in single and double-single precision
GRAPE6 / Fourth order in double-single and double precision including neighbour lists
Sixt order hermite, in double precision, including neighbour lists



*/

#include <stdio.h>

#include "defines.h"

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


#if __CUDA_ARCH__ < 600
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
#endif


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

//   if(isnan(inv_ds))
//   {
//     if(threadIdx.x == 0)
//   {
//     printf("NAN : %f %f %f %f \t %f %f %f %f \n",  
//             iParticle.posx,  iParticle.posy,  iParticle.posz,  iParticle.mass_h2,
//             jpos.posx, jpos.posy, jpos.posz, EPS2);
//   }
// }

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
//      outVal.pot    += 1;

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
    if(integrationOrder > FOURTH)
    {
      iParticle.setAcc(acc_i_in[tx+ni_offset]);
    }
  }

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
          out2.snpy += shared_snp[i + tx].y;
          out2.snpz += shared_snp[i + tx].z;            
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


  if(tx+ni_offset >= ni_total) return;

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

    if(integrationOrder > GRAPE5) {
      jrk_i[   tx+ni_offset].x   += jrk.x;
      jrk_i[   tx+ni_offset].y   += jrk.y;
      jrk_i[   tx+ni_offset].z   += jrk.z;

      if(integrationOrder > FOURTH) {
        snp_i[tx+ni_offset].x += snp.x;
        snp_i[tx+ni_offset].y += snp.y;
        snp_i[tx+ni_offset].z += snp.z;
      }
    }
    
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
//
CUDA_GLOBAL ( dev_evaluate_gravity_sixth_DS, DSX, float, float3, float4, true, true, SIXTH);


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
