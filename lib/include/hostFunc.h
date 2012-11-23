#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cassert>
#include <xmmintrin.h>


// g++ hostEvaluate.cpp -O3 -mavx -I ../include -Wall
// g++ hostEvaluate.cpp -O3 -msse4 -I ../include -Wall

#if 1
  #include "SSE_AVX/SSE/sse.h"
#else
  #include "SSE_AVX/AVX/avx.h"
#endif

#if 0
typedef float real;
#else
typedef double real;
#endif

typedef SIMD::scalar<real> sreal;
typedef SIMD::vector<real> vreal;
typedef vreal REAL;

#define LDA(x) (REAL::aload(x))
#define LDU(x) (REAL::uload(x))
#define REF(x) (REAL::aref(x))


struct Force
{
  vreal acc[3];
  vreal jrk[3];
  vreal pot;
};

typedef int    _v4si  __attribute__((vector_size(16)));

struct Predictor
{
  vreal pos[3];
  vreal vel[3];
  vreal mass;
  _v4si id;
};

static inline 
vreal dot(const vreal a[3], const vreal b[3]) 
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline
vreal RSQRT(const vreal x) 
{
#if 1
  return vreal(1.0)/REAL::sqrt(x);
#else
  return REAL::rsqrt(x);
#endif
}

static long long int interCount = 0;

static inline 
void body_body_force(
    Force &f,
    const Predictor &pi, 
    const Predictor &pj, 
    const vreal eps2)
{
  
  const vreal dr[3] = {
    pj.pos[0] - pi.pos[0],
    pj.pos[1] - pi.pos[1],
    pj.pos[2] - pi.pos[2] };
  const vreal ds2 = dot(dr,dr) + eps2;
  vreal  inv_ds  = RSQRT(ds2);
  
    
  // check i-particleID against j-particleIds to build a mask  
  //Built a mask that has 0xFF.. when id's are not equal, and 0 when id's are equal
  //this should change inv_ds from a NaN to a 0 incase we take rsqrt of 0
  //VERY slow (factor 2 slow down :( )
  const SIMD::vector<float> mask =  __builtin_ia32_cmpneqps((REAL::_v4sf)pi.id, (REAL::_v4sf)pj.id);
  inv_ds                         = inv_ds & mask;
  
  
  const vreal  inv_ds2 = inv_ds  * inv_ds;
  const vreal minv_ds  = inv_ds  * pj.mass;
  const vreal minv_ds3 = inv_ds2 * minv_ds;

  f.acc[0] += minv_ds3 * dr[0];
  f.acc[1] += minv_ds3 * dr[1];
  f.acc[2] += minv_ds3 * dr[2];
  f.pot    -= minv_ds;
  
  const vreal dv[3] = {
    pj.vel[0] - pi.vel[0],
    pj.vel[1] - pi.vel[1],
    pj.vel[2] - pi.vel[2] };
  const vreal rv = dot(dr,dv);
  
  const vreal Jij = vreal(-3.0) * rv * inv_ds2 * minv_ds3;
  
  f.jrk[0] += minv_ds3*dv[0] + Jij*dr[0];
  f.jrk[1] += minv_ds3*dv[1] + Jij*dr[1];
  f.jrk[2] += minv_ds3*dv[2] + Jij*dr[2];
}


void compute_forces_jb(
    const int     ni,
    const int     nj,
    const real massi[],
    const real posxi[],
    const real posyi[],
    const real poszi[],
    const real velxi[],
    const real velyi[],
    const real velzi[],
    const int  idi  [],
    const real massj[],
    const real posxj[],
    const real posyj[],
    const real poszj[],
    const real velxj[],
    const real velyj[],
    const real velzj[],
    const int  idj  [],
    real accx[],
    real accy[],
    real accz[],
    real jrkx[],
    real jrky[],
    real jrkz[],
    real gpot[],
    const real seps2)
{
  const vreal eps2(seps2);
  
  #pragma omp parallel for
  for (int i = 0; i < ni; i+= vreal::WIDTH)
  {
    Force fi;
    Predictor pi;
    pi.mass   = LDA(massi[i]);
    pi.pos[0] = LDA(posxi[i]);
    pi.pos[1] = LDA(posyi[i]);
    pi.pos[2] = LDA(poszi[i]);
    pi.vel[0] = LDA(velxi[i]);
    pi.vel[1] = LDA(velyi[i]);
    pi.vel[2] = LDA(velzi[i]);
    
    //TODO JB Can this be done in a cleaner way?
    if(vreal::WIDTH == 4)
      pi.id = *(_v4si*)&idi[i];
    else
    {
      //Double, use same ID twice
      pi.id = __builtin_ia32_vec_set_v4si(pi.id, idi[i], 0);
      pi.id = __builtin_ia32_vec_set_v4si(pi.id, idi[i], 1);
      pi.id = __builtin_ia32_vec_set_v4si(pi.id, idi[i+1], 2);
      pi.id = __builtin_ia32_vec_set_v4si(pi.id, idi[i+1], 3);
    }
        
    fi.acc[0] = 0.0;
    fi.acc[1] = 0.0;
    fi.acc[2] = 0.0;
    fi.jrk[0] = 0.0;
    fi.jrk[1] = 0.0;
    fi.jrk[2] = 0.0;
    fi.pot    = 0.0;

    for (int j = 0; j < nj; j++)
    {
      Predictor pj;
      pj.mass   = massj[j];
      pj.pos[0] = posxj[j];
      pj.pos[1] = posyj[j];
      pj.pos[2] = poszj[j];
      pj.vel[0] = velxj[j];
      pj.vel[1] = velyj[j];
      pj.vel[2] = velzj[j];
      
      //Set all j-items to the same ID
      pj.id  =  (_v4si)_mm_set1_epi32(idj[j]);
      
      body_body_force(fi, pi, pj, eps2);
      
      
//       fprintf(stderr, "idi: %d %d %d %d and j: %d %d %d %d \n", 
//               idi[i+0],idi[i+1],idi[i+2],idi[i+3],
//               idj[j+0],idj[j+0],idj[j+0],idj[j+0]);
      
//       fprintf(stderr, "idi: %d %d %d %d and j: %d %d %d %d \n", 
//               idi[i+0],idi[i+1],idi[i+2],idi[i+3],
//               idj[j+0],idj[j+0],idj[j+0],idj[j+0]);
// 
//       fprintf(stderr, "pi.id %d %d %d %d \n", 
//               __builtin_ia32_vec_ext_v4si(pi.id, 0),
//               __builtin_ia32_vec_ext_v4si(pi.id, 1),
//               __builtin_ia32_vec_ext_v4si(pi.id, 2),
//               __builtin_ia32_vec_ext_v4si(pi.id, 3));   
//       fprintf(stderr, "pj.id %d %d %d %d \n", 
//               __builtin_ia32_vec_ext_v4si(pj.id, 0),
//               __builtin_ia32_vec_ext_v4si(pj.id, 1),
//               __builtin_ia32_vec_ext_v4si(pj.id, 2),
//               __builtin_ia32_vec_ext_v4si(pj.id, 3));        
//       
//       fprintf(stderr, "Force (%d): %f %f %f %f \n", j,
//               fi.acc[0][0],fi.acc[0][1],fi.acc[0][2],fi.acc[0][3]);
//       
//       const SIMD::_v4sf mask =  __builtin_ia32_cmpneqps((SIMD::_v4sf)pi.id, (SIMD::_v4sf)pj.id);
//       vreal test1 = (SIMD::_v4sf)pi.id;
//       vreal test2 = (SIMD::_v4sf)pj.id;
//       
//       fprintf(stderr, "test1: %f %f %f %f \n", test1[0],test1[1],test1[2],test1[3]);
//       fprintf(stderr, "test2: %f %f %f %f \n", test2[0],test2[1],test2[2],test2[3]);
//       
//       
//       fprintf(stderr, "%f %f %f %f \n", 
//               __builtin_ia32_vec_ext_v4sf(mask, 0),
//               __builtin_ia32_vec_ext_v4sf(mask, 1),
//               __builtin_ia32_vec_ext_v4sf(mask, 2),
//               __builtin_ia32_vec_ext_v4sf(mask, 3));

    }//for j
    REF(accx[i]) = fi.acc[0];
    REF(accy[i]) = fi.acc[1];
    REF(accz[i]) = fi.acc[2];
    REF(jrkx[i]) = fi.jrk[0];
    REF(jrky[i]) = fi.jrk[1];
    REF(jrkz[i]) = fi.jrk[2];
    REF(gpot[i]) = fi.pot;
  }//for i
}//compute_forces

#define FTINY 1e-10
#define FHUGE 1e+10

struct real4 {
  real x, y, z, w;
  real4() {};
  real4(const real r) : x(r), y(r), z(r), w(r) {}
  real4(const real &_x, 
      const real &_y,
      const real &_z,
      const real &_w = 0) : x(_x), y(_y), z(_z), w(_w) {};
  real abs2() const {return x*x + y*y + z*z;}
};

inline real sqr(const real &x) {return x*x;}



void forces_jb(
    const int nj,    
    const double4 posj[],
    const double4 velj[],
    const int     id_j[],
    const int ni,
    const double4 posi[],
    const double4 veli[],
    const int     id_i[],
    double4 acc[],
    double4 jrk[],
    const real eps2)   
{
  
//   const int max_nj = 65536;
//   const int max_ni = 65536;
  const int max_nj  = 50000; //65000 fast, 66000 fast, 65535 slow, 65500 slow, 65750 fast,  66750 fast
  const int max_ni  = 50000;
  
  static std::vector<real> massi(max_ni);
  static std::vector<real> posxi(max_ni);
  static std::vector<real> posyi(max_ni);
  static std::vector<real> poszi(max_ni);
  static std::vector<real> velxi(max_ni);
  static std::vector<real> velyi(max_ni);
  static std::vector<real> velzi(max_ni);
  static std::vector<int > idi  (max_ni);
  static std::vector<real> massj(max_nj);
  static std::vector<real> posxj(max_nj);
  static std::vector<real> posyj(max_nj);
  static std::vector<real> poszj(max_nj);
  static std::vector<real> velxj(max_nj);
  static std::vector<real> velyj(max_nj);
  static std::vector<real> velzj(max_nj); 
  static std::vector<int > idj  (max_nj);
  static std::vector<real> accx(max_ni);
  static std::vector<real> accy(max_ni);
  static std::vector<real> accz(max_ni);
  static std::vector<real> jrkx(max_ni);
  static std::vector<real> jrky(max_ni);
  static std::vector<real> jrkz(max_ni);
  static std::vector<real> gpot(max_ni);  
  
  
  //Walk in blocks over the i-particles
  for(int z=0; z < ni; z += max_ni)    
  {
    int cur_ni = std::min(max_ni, ni-z);

    //Copy the i-particles we are currently processing
    //and reset their acceleration/jerk
    for(int i=0; i < cur_ni; i++)
    {
      massi[i] = posi[z+i].w;
      posxi[i] = posi[z+i].x;
      posyi[i] = posi[z+i].y;
      poszi[i] = posi[z+i].z;
      velxi[i] = veli[z+i].x;
      velyi[i] = veli[z+i].y;
      velzi[i] = veli[z+i].z;    
      idi  [i] = id_i[z+i];
      
      acc[z+i].w = 0; acc[z+i].x = 0;
      acc[z+i].y = 0; acc[z+i].z = 0;
      jrk[z+i].x = 0; jrk[z+i].y = 0;
      jrk[z+i].z = 0;
    } //for i
      
    //walk in blocks over the j-particles  
    for(int y=0; y < nj; y += max_nj)    
    {
      int cur_nj = std::min(max_nj, nj-y);    

      for(int j=0; j < cur_nj; j++)
      {        
        //Copy data into the data-structures
        massj[j] = posj[y+j].w;
        posxj[j] = posj[y+j].x;
        posyj[j] = posj[y+j].y;
        poszj[j] = posj[y+j].z;
        velxj[j] = velj[y+j].x;
        velyj[j] = velj[y+j].y;
        velzj[j] = velj[y+j].z;     
        idj  [j] = id_j[y+j];
        
//         fprintf(stderr, "J value: %d || %f %f %f || %f %f %f \n", 
                
        
      }//for j 
      
      compute_forces_jb(
          cur_ni,
          cur_nj,
          &massi[0],
          &posxi[0],
          &posyi[0],
          &poszi[0],
          &velxi[0],
          &velyi[0],
          &velzi[0],
          &idi  [0],
          &massj[0],
          &posxj[0],
          &posyj[0],
          &poszj[0],
          &velxj[0],
          &velyj[0],
          &velzj[0],
          &idj  [0],
          &accx[0],
          &accy[0],
          &accz[0],
          &jrkx[0],
          &jrky[0],
          &jrkz[0],
          &gpot[0],
          eps2);      

      for (int i = 0; i < cur_ni; i++)
      {
        //Store first set of results
        acc[z+i].w += gpot[i];
        acc[z+i].x += accx[i];
        acc[z+i].y += accy[i];
        acc[z+i].z += accz[i];
        jrk[z+i].x += jrkx[i];
        jrk[z+i].y += jrky[i];
        jrk[z+i].z += jrkz[i];
      }//for i
    }//for y
  }//for z
}//forces_host


