#include "sapporohostclass.h"

struct real3 {
  double x, y, z;
};
double get_time() {
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +
          1.e-6*((double) Tvalue.tv_usec));
}


int main(int argc, char *argv[]) {
  int n = 1024;
  int ndev = 1;
  if (argc > 1) n = atoi(argv[1]);
  cerr << " n = " << n << endl;
//   if (argc > 2) ndev = atoi(argv[2]);
//   cerr << " ndev = " << ndev << endl;

  int n1 = 0;
  int n2 = n;

  double (*pos)[3] = new double[n][3];
  double (*vel)[3] = new double[n][3];
  double (*acc)[3] = new double[n][3];
  double (*jrk)[3] = new double[n][3];
  double (*snp)[3] = new double[n][3];
  double (*crk)[3] = new double[n][3];
  double *pot  = new double[n];
  double *mass = new double[n];
  int    *nnb  = new int[n];
  double *h2  = new double[n];
  int    *id   = new int[n];

  double tm = 0;
  for (int i = 0; i < n; i++) {
    pos[i][0] = drand48();
    pos[i][1] = drand48();
    pos[i][2] = drand48();

    vel[i][0] = drand48();
    vel[i][1] = drand48();
    vel[i][2] = drand48();

    h2[i] = 1.0 * pow(12.0/n, 1.0/3);
    h2[i] = h2[i]*h2[i];

    mass[i] = 1.0/n * drand48();
    tm += mass[i];
    id[i] = i + 1;
  }

  for (int i = 0; i < n; i++) {
    mass[i] *= 1.0/tm;
  }

  sapporo grav;

//   int sapporo::open(std::string kernelFile, int *devices, int nprocs = 1, int order = FOURTH)
//   grav.open("CUDA/kernels4th.ptx", cluster_id, 1, 1);
  int devices[] = {0,1,2,3,4,5};
//  grav.open("CUDA/kernels4thDP.ptx",devices , 1, 1);
  
  
  std::string kernelFile;
  
  if (argc > 2)
  {
    kernelFile.assign(argv[2]);
  }
  else
  {    
    #ifdef _OCL_
      kernelFile.assign("OpenCL/kernels6th.cl");
    #else
      kernelFile.assign("CUDA/kernels.ptx");
    #endif
  }
  
  int integrationOrder = 2;     //6th
  int integrationPrecision = 2; //Default double-precision
  int nDevices = 1;
  
 
  if (argc > 3) integrationOrder        = atoi(argv[3]);
  if (argc > 4) integrationPrecision    = atoi(argv[4]);
  if (argc > 5) nDevices                = atoi(argv[5]);  
 
  
  fprintf(stderr, "%s is using: n: %d\tDevices: %d\tIntegrationOrder: %d\tIntegrationPrecision: %d\tFile: %s\n",
          argv[0], n, nDevices, integrationOrder, integrationPrecision, kernelFile.c_str());
  
//   int sapporo::open(std::string kernelFile, int *devices, int nprocs = 1, int order = FOURTH, int precision = DEFAULT)  
  grav.open(kernelFile.c_str(),devices, nDevices, integrationOrder, integrationPrecision);
    
  
  int ipmax = grav.get_n_pipes();

  double null3[3] = {0,0,0};
  for (int i = 0; i < n; i++) {
    grav.set_j_particle(i, id[i],
			0, 0,
			mass[i],
                        null3,
                        null3,
                        null3,
			vel[i],
                        pos[i],
                        null3, null3, 0);
  }

  grav.set_time(0);

  double eps2 = 0;
  n1 = 0;
  n2 = 2*ipmax;
//   for (int i = n1; i < n2; i += ipmax) {
  for (int j = 0; j < 1; j++) {
//     int npart = min(n2 - i, ipmax);


    double tx = get_time();

    for (int i = 0; i < n; i += ipmax)
    {
      int npart = min(n - i, ipmax);


      printf("n: %d  start: %d  npart: %d \n", n, i, npart);

      grav.startGravCalc(n, npart,
                          id + i,
                          pos+i, vel+i,
                          acc+i, jrk+i, pot+i, eps2, h2, NULL);

      grav.getGravResults(n, npart,
                          id+i, pos+i, vel+i,
                          eps2, h2,
                          acc+i, jrk+i,snp+i, crk+i, pot+i, nnb+i, NULL, true);
//         for(int k=0; k < 10; k++)
//        {
//             fprintf(stderr, " %d  %f \n", i+k, pot[i+k]);
//         }

    }

    fprintf(stderr, "TIMING N: %d\tNGPU: %d\tTSEC: %lg \n",
            n, ndev, get_time() - tx);
    cerr << " done in " << get_time() - tx << " sec \n";
  }
  grav.close();



  cerr << "done!\n";
}
