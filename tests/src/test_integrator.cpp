#include <cstdlib>
#include <cmath>
#include "sapporohostclass.h"
#include "g6lib.h"

#include <cstdio>
#include <algorithm>
#include <vector>
#include <cassert>
//#include <xmmintrin.h>

// g++ src/test_integrator.cpp -I ../lib/ -I ../lib/include/ -I /usr/local/cuda/include/ -lsapporo -L ../lib/ -lcuda -fopenmp -o test_integrator

#define JB 1
#define real double

// typedef struct real4
// {
//   double x,y,z,w;
// } real4;

struct real4 {
  real x, y, z, w;
  real4() {};
  real4(const real r) : x(r), y(r), z(r), w(r) {}
  real4(const real &_x, 
      const real &_y,
      const real &_z,
      const real &_w = 0) : x(_x), y(_y), z(_z), w(_w) {};
  const real abs2() const {return x*x + y*y + z*z;}
};


#include <sys/time.h>
inline double get_time() {
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}


// g++ hostEvaluate.cpp -O3 -mavx -I ../include -Wall
// g++ hostEvaluate.cpp -O3 -msse4 -I ../include -Wall


#define FTINY 1e-10
#define FHUGE 1e+10

const real timestep(
    const int n,
    double (*vel)[3] ,
    double (*acc)[3] ,
    const real eta,
    const real eps2) 
{
  real dt_min = FHUGE;
  for (int i = 0; i < n; i++) {
    real absV = (vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1] + vel[i][2]*vel[i][2] );
    real absA = (acc[i][0]*acc[i][0] + acc[i][1]*acc[i][1] + acc[i][2]*acc[i][2] );
    const real vabs = std::sqrt(absV);
    const real aabs = std::sqrt(absA);

    const real s = std::sqrt(eps2);
    const real idt1 = s/(vabs + FTINY);
    const real idt2 = std::sqrt(s/(aabs + FTINY));
    dt_min = std::min(dt_min, std::min(idt1, idt2));
  }
  dt_min *= eta;

  return dt_min;
}


void forces(
    const int ni,
    const int nj,
    double (*posi)[3] ,
    double (*veli)[3] ,
    int     *ids, 
    double (*acc)[3] ,
    double (*jrk)[3] ,
    double *pot,
    double *h2,
    const real eps2,
    int npipes) 
{
  
  for(int i=0; i < ni; i+=npipes)
  {
    int ntoCalc = min(npipes, ni-i);
    
    g6calc_firsthalf(0, nj, ntoCalc, ids+i, posi+i, veli+i, acc+i, jrk+i, pot+i, eps2, h2+i);
    
    g6calc_lasthalf(0, nj, ntoCalc, ids+i, posi+i, veli+i, eps2, h2+i, acc+i, jrk+i, pot+i);        
  }
  
}



const real iterate(
    const int n,
    double (*pos)[3] ,
    double (*vel)[3] ,
    double (*acc)[3] ,
    double (*jrk)[3] ,       
    const real eta,
    const real eps2,
    const real dt,
    double *h2,
    double *pot,
    int     *ids, 
    int npipes) 
{
  std::vector<real4> acc0(n), jrk0(n);
  const real dt2 = dt*(1.0/2.0);
  const real dt3 = dt*(1.0/3.0);

  for (int i = 0; i < n; i++)
  {
    acc0[i].x = acc[i][0]; acc0[i].y = acc[i][1]; acc0[i].z = acc[i][2];
    jrk0[i].x = jrk[i][0]; jrk0[i].y = jrk[i][1]; jrk0[i].z = jrk[i][2];

    pos[i][0] += dt*(vel[i][0] + dt2*(acc[i][0] + dt3*jrk[i][0]));
    pos[i][1] += dt*(vel[i][1] + dt2*(acc[i][1] + dt3*jrk[i][1]));
    pos[i][2] += dt*(vel[i][2] + dt2*(acc[i][2] + dt3*jrk[i][2]));

    vel[i][0] += dt*(acc[i][0] + dt2*jrk[i][0]);
    vel[i][1] += dt*(acc[i][1] + dt2*jrk[i][1]);
    vel[i][2] += dt*(acc[i][2] + dt2*jrk[i][2]);
  }

#if 0
  forces_jb(n, pos, vel, n, pos, vel, acc, jrk, eps2);  
#else
  //forces(n, pos, vel, acc, jrk, eps2);
    forces(n,n, pos, vel, ids, acc, jrk, pot, h2, eps2, npipes);
#endif
  
  if (dt > 0.0)
  {
    const real h    = dt*0.5;
    const real hinv = 1.0/h;
    const real f1   = 0.5*hinv*hinv;
    const real f2   = 3.0*hinv*f1;

    const real dt2  = dt *dt * (1.0/2.0);
    const real dt3  = dt2*dt * (1.0/3.0);
    const real dt4  = dt3*dt * (1.0/4.0);
    const real dt5  = dt4*dt * (1.0/5.0);

    for (int i = 0; i < n; i++)
    {

      /* compute snp & crk */

      const real4 Am(   acc[i][0] - acc0[i].x,     acc[i][1] - acc0[i].y,     acc[i][2] - acc0[i].z);
      const real4 Jm(h*(jrk[i][0] - jrk0[i].x), h*(jrk[i][1] - jrk0[i].y), h*(jrk[i][2] - jrk0[i].z));
      const real4 Jp(h*(jrk[i][0] + jrk0[i].x), h*(jrk[i][1] + jrk0[i].y), h*(jrk[i][2] + jrk0[i].z));
      real4 snp(f1* Jm.x,         f1* Jm.y,         f1* Jm.z        );
      real4 crk(f2*(Jp.x - Am.x), f2*(Jp.y - Am.y), f2*(Jp.z - Am.z));

      snp.x -= h*crk.x;
      snp.y -= h*crk.y;
      snp.z -= h*crk.z;

      /* correct */

      pos[i][0] += dt4*snp.x + dt5*crk.x;
      pos[i][1] += dt4*snp.y + dt5*crk.y;
      pos[i][2] += dt4*snp.z + dt5*crk.z;

      vel[i][0] += dt3*snp.x + dt4*crk.x;
      vel[i][1] += dt3*snp.y + dt4*crk.y;
      vel[i][2] += dt3*snp.z + dt4*crk.z;
    }
  }

  return timestep(n, vel, acc, eta, eps2);
}



void energy(
    const int n,
    double (*pos)[3] ,
    double (*vel)[3] ,
    double (*acc)[3] ,
    double *pot, 
    double  *mass,
    real &Ekin, real &Epot) {
  Ekin = Epot = 0;
  for (int i = 0; i < n; i++) {
    Ekin += mass[i] * (vel[i][0]*vel[i][0] + vel[i][1]*vel[i][1] + vel[i][2]*vel[i][2] ) * 0.5;
    Epot += 0.5*mass[i] * pot[i];
  }
}


void integrate(
    double (*pos)[3] ,
    double (*vel)[3] ,
    double *mass,
    const real eta,
    const real eps2,
    const real t_end,
    const int n) {

 double (*acc)[3] = new double[n][3];
 double *pot  = new double[n];
 double *h2  = new double[n];
 double (*jrk)[3] = new double[n][3];
  
  int *ids = new int[n];
  
  int npipes = g6_npipes();
  
  fprintf(stderr, "Number of pipes: %d \n", npipes); 


  const double tin = get_time();
  
  double timeStart = 0;
  int clusterID = 0;
  double null3[3] = {0,0,0};
  for(int i=0; i < n; i++)
  {
    ids[i] = 3000+i;
    double null3[3] = {0,0,0};

    g6_set_j_particle_(&clusterID, &i, &i, &timeStart, &timeStart,
                       &mass[i], null3, null3,
                       null3, vel[i], pos[i]);
  }
  
  g6_set_ti(0,0);

  forces(n,n, pos, vel, ids, acc, jrk, pot, h2, eps2, npipes);

  
  const double fn = n;
  fprintf(stderr, " mean flop rate in %g sec [%g GFLOP/s]\n", get_time() - tin,
      fn*fn*60/(get_time() - tin)/1e9);
  
  

  real Epot0, Ekin0;
//   energy(n, &pos[0], &vel[0], &acc[0], Ekin0, Epot0);
  energy(n, pos, vel, acc, pot, mass, Ekin0, Epot0);  
  const real Etot0 = Epot0 + Ekin0;
  fprintf(stderr, " E: %g %g %g \n", Epot0, Ekin0, Etot0);

  /////////

  real t_global = 0;
  double t0 = 0;
  int iter = 0;
  int ntime = 10;
  real dt = 0;
  real Epot, Ekin, Etot = Etot0;
  while (t_global < t_end) {
    if (iter % ntime == 0) 
      t0 = get_time();
    
    g6_set_ti(0,t_global);    

    dt = iterate(n, &pos[0], &vel[0], &acc[0], &jrk[0], eta, eps2, dt, h2, pot, ids, npipes );
    iter++;
    

    const real Etot_pre = Etot;
    //energy(n, &pos[0], &vel[0], &acc[0], Ekin, Epot);
    energy(n, pos, vel, acc, pot, mass, Ekin, Epot);

    Etot = Ekin + Epot;
    
    //Update j-particles on device
    for(int i=0; i < n; i++)
    {
      g6_set_j_particle_(&clusterID, &i, &i, &t_global, &t_global,
                         &mass[i], null3, null3,
                         null3, vel[i], pos[i]);
    }
  
    t_global += dt;

    if (iter % 1 == 0) {
      const real Etot = Ekin + Epot;
      fprintf(stderr, "iter= %d: t= %g  dt= %g Ekin= %g  Epot= %g  Etot= %g , dE = %g d(dE)= %g \n",
          iter, t_global, dt, Ekin, Epot, Etot, (Etot - Etot0)/std::abs(Etot0),
          (Etot - Etot_pre)/std::abs(Etot_pre)   );
    }

    if (iter % ntime == 0) {
      fprintf(stderr, " mean flop rate in %g sec [%g GFLOP/s]\n", get_time() - t0,
          fn*fn*60/(get_time() - t0)/1e9*ntime);
    }

  }
};

int main(int argc, char *argv[]) {
  int nbodies = 131072;

  nbodies = 8192;
  nbodies = 1024;  
  
  if (argc > 1) nbodies = atoi(argv[1]);
  
  int n = nbodies;
  double (*pos)[3] = new double[n][3];
  double (*vel)[3] = new double[n][3];

  double *mass = new double[n];
  int    *nnb  = new int[n];
  double *h2  = new double[n];
  int    *nngb  = new int[n];
  int    *ngb_list = new int[n];
  int    *id   = new int[n];

  
  fprintf(stderr, "nbodies= %d\n", nbodies);
  const real R0 = 1;
  const real mp = 1.0/nbodies;
  for (int i = 0; i < nbodies; i++) {
    real xp, yp, zp, s2 = 2*R0;
    real vx, vy, vz;
    while (s2 > R0*R0) {
      xp = (1.0 - 2.0*drand48())*R0;
      yp = (1.0 - 2.0*drand48())*R0;
      zp = (1.0 - 2.0*drand48())*R0;
      s2 = xp*xp + yp*yp + zp*zp;
      vx = drand48() * 0.1;
      vy = drand48() * 0.1;
      vz = drand48() * 0.1;
    } 
    
    pos[i][0] = xp;  pos[i][1] = yp;
    pos[i][2] = zp;
    mass[i]   = mp;
 
    vel[i][0] = vx;  vel[i][1] = vy;
    vel[i][2] = vz;  

  }
  const real eps2 = 4.0f/nbodies;
//   const real eps2 = 0;;
  real eta  = 0.01f;

  if (argc > 2) eta *= atof(argv[2]);
  fprintf(stderr, " eta= %g \n", eta);
  fprintf(stderr, " starting ... \n");
  const real tend = 1.0;
  
  g6_open_(0);
  
  
  
  integrate(pos, vel, mass, eta, eps2, tend, n);

}
