/*

Sapporo 2 device kernels

Version 1.0
OpenCL Double Precision

*/


#include "OpenCL/sharedKernels.cl"



__inline void body_body_interaction(
                                    inout float2 *ds2_min,
                                    inout int   *n_ngb,
                                    inout __private int *ngb_list,
                                    inout double4 *accNew_i,
                                    inout double4 *jrkNew_i,
                                    inout double4 *snpNew_i,
                                    const double4  pos_i,
                                    const double4  vel_i,
                                    const double4  acc_i,
                                    const double4  pos_j,
                                    const double4  vel_j,
                                    const double4  acc_j,
                                    const int iID,
                                    const double  EPS2) {

  const int jID   = as_int((float)vel_j.w);
  if (iID != jID)    /* assuming we always need ngb */
  {

    const double3 dr = {pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z};
    const double ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#if 0
    if (ds2 <= pos_i.w.x && n_ngb < NGB_PB)
      ngb_list[n_ngb++] = jID;
#else

#if ((NGB_PB & (NGB_PB - 1)) != 0)
#error "NGB_PB is not a power of 2!"
#endif

    /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
    (*ds2_min) = ((*ds2_min).x < ds2) ? (*ds2_min) : (float2){ds2, as_float(jID)};

    if (ds2 <= pos_i.w)
    {
      ngb_list[(*n_ngb) & (NGB_PB-1)] = jID;
      (*n_ngb)++;
    }
#endif

    /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
//     if (ds2 <= pos_i.w.x)
//     {
//       ngb_list[(*n_ngb) & (NGB_PB-1)] = jID;
//       (*n_ngb)++;
//     }

    const double inv_ds = rsqrt(ds2+EPS2);

    const double mass    = pos_j.w;
    const double minvr1  = mass*inv_ds;
    const double inv_ds2 = inv_ds*inv_ds;                         // 1 FLOP
    const double inv_ds3 = mass * inv_ds*inv_ds2;                 // 2 FLOP

    // 3*4 + 3 = 15 FLOP
    (*accNew_i).x += inv_ds3 * dr.x;
    (*accNew_i).y += inv_ds3 * dr.y;
    (*accNew_i).z += inv_ds3 * dr.z;
    (*accNew_i).w +=  (-1.0)*minvr1; //Potential
    
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
    (*snpNew_i).x += (inv_ds3 * da.x) + alpha * jerk.x + beta * (inv_ds3 * dr.x);
    (*snpNew_i).y += (inv_ds3 * da.y) + alpha * jerk.y + beta * (inv_ds3 * dr.y);
    (*snpNew_i).z += (inv_ds3 * da.z) + alpha * jerk.z + beta * (inv_ds3 * dr.z);

    //Had to reuse jerk for snap so only add to total now
    (*jrkNew_i).x += jerk.x;
    (*jrkNew_i).y += jerk.y;
    (*jrkNew_i).z += jerk.z;

    // TOTAL 50 FLOP (or 60 FLOP if compared against GRAPE6)
  }

}

/*
 *  blockDim.x = ni
 *  gridDim.x  = 16, 32, 64, 128, etc.
 */


#define ajc(i, j) (i + blockDim_x*j)
__kernel void dev_evaluate_gravity(
                                     const          int        nj_total,
                                     const          int        nj,
                                     const          int        ni_offset,
                                     const __global double4    *pos_j,
                                     const __global double4    *pos_i,
                                     __out __global double4    *acc_i_out,
                                                      double     EPS2,
                                     const __global double4    *vel_j,
                                     const __global int        *id_j,
                                     const __global double4    *vel_i,
                                     __out __global double4    *jrk_i,
                                     __out __global int        *id_i,
                                     __out __global double     *ds2min_i,
                                     __out __global int        *ngb_count_i,
                                     __out __global int        *ngb_list,
                                    const __global double4     *acc_i_in,
                                           __global double4    *acc_j,
                                           __global double4    *snp_i,
                                           __local  double4    *shared_pos) {
#if 0
  const int tx = threadIdx_x;
  const int ty = threadIdx_y;
  const int bx =  blockIdx_x;
  const int Dim = blockDim_x*blockDim_y;

//   __local double4 *shared_vel = (__local double4*)&shared_pos[Dim];
  __local double4 *shared_vel = (__local double4*)&shared_pos[Dim];
  __local double3 *shared_acc = (__local double3*)&shared_vel[Dim];
  __local int     *shared_id  = (__local int*)&shared_acc[Dim];

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const double4 pos    = pos_i   [tx+ni_offset];
  const double4 vel    = vel_i   [tx+ni_offset];
  const double4 acc    = acc_i_in[tx+ni_offset];
  const int particleID = id_i    [tx+ni_offset];

  //Set the softening for the i-particle
  EPS2 = vel.w;

  const float LARGEnum = 1.0e10f;

  float2      ds2_min2;
  ds2_min2.x  = LARGEnum;
  ds2_min2.y  = as_float(-1);

  double4 accNew = {0.0f, 0.0f, 0.0f, 0.0f};
  double4 jrkNew = {0.0f, 0.0f, 0.0f, 0.0f};
  double4 snpNew = {0.0f, 0.0f, 0.0f, 0.0f};

  int tile       = 0;
  int ni         = bx * (nj*blockDim_y) + nj*ty;
  const int offy = blockDim_x*ty;
  for (int i = ni; i < ni+nj; i += blockDim_x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total)
    {
      shared_pos[addr]     = pos_j[i + tx];
      shared_id[addr]      = id_j[i + tx];
      shared_vel[addr]     = vel_j[i + tx];
      shared_acc[addr] = (double3){acc_j[i + tx].x, acc_j[i + tx].y, acc_j[i + tx].z};
    } else {
      shared_pos[addr] = (double4){LARGEnum,LARGEnum,LARGEnum,0};
      shared_id[addr]  = -1;
      shared_vel[addr]   = (double4){0.0, 0.0, 0.0, 0.0};
      shared_acc[addr]   = (double3){0.0, 0.0, 0.0};
    }
    __syncthreads();

    const int j  = min(nj - tile*blockDim_x, blockDim_x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++)
      body_body_interaction(&ds2_min2, &n_ngb, local_ngb_list,
          &accNew, &jrkNew, &snpNew, pos, vel, acc,
          shared_pos[offy+k], shared_vel[offy+k],
          shared_acc[offy+k],
          particleID, EPS2);

    for (int k = j1; k < j; k++)
      body_body_interaction(&ds2_min2, &n_ngb, local_ngb_list,
          &accNew, &jrkNew, &snpNew, pos, vel, acc,
          shared_pos[offy+k], shared_vel[offy+k],
          shared_acc[offy+k],
          particleID, EPS2);

    __syncthreads();

    tile++;
  } //end while

  __local double4 *shared_acc2 = (__local double4*)&shared_pos[0];
  __local double4  *shared_snp = (__local double4* )&shared_acc2[Dim];

  acc.w = -acc.w;

  const int addr = offy + tx;
  shared_acc2[addr] = accNew;
  shared_snp[addr]  = snpNew;
  __syncthreads();


  if (ty == 0) {
    for (int i = blockDim_x; i < Dim; i += blockDim_x) {
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

  __local double4 *shared_jrk = (__local double4*)&shared_pos[0];
  __local int     *shared_ngb = (__local int*   )&shared_jrk[Dim];
  __local int     *shared_ofs = (__local int*   )&shared_ngb[Dim];
  __local double  *shared_ds  = (__local double*)&shared_ofs[Dim];


//TODO re-enable this after testing
//   n_ngb = 0;


  double ds2_min = ds2_min2.x;
  jrkNew.w       = ds2_min2.y;
  shared_jrk[addr] = jrkNew;
  shared_ngb[addr] = n_ngb;
  shared_ofs[addr] = 0;
  shared_ds [addr] = ds2_min;
  __syncthreads();

  if (threadIdx_y == 0) {
    for (int i = blockDim_x; i < Dim; i += blockDim_x) {
      const int addr = i + tx;
      double4 jrk1 = shared_jrk[addr];
      double  ds1  = shared_ds [addr];

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

  if (threadIdx_y == 0) {
    //Store the results
    const int addr = bx*blockDim_x + tx;
    ds2min_i[      addr] = ds2_min;
    acc_i_out[     addr] = accNew;
    jrk_i[         addr] = jrkNew;
    snp_i[         addr] = snpNew;
    ngb_count_i[   addr] = n_ngb;
  }

  //Write the neighbour list
  {
    int offset  = threadIdx_x * gridDim_x*NGB_PB + blockIdx_x * NGB_PB;
    offset     += shared_ofs[addr];
    n_ngb       = shared_ngb[addr];
    for (int i = 0; i < n_ngb; i++) 
      ngb_list[offset + i] = local_ngb_list[i];
  }

//   {
//     //int offset  = threadIdx.x * NBLOCKS*NGB_PB + blockIdx.x * NGB_PB;
//     int offset  = threadIdx_x * gridDim_x*NGB_PB + blockIdx_x * NGB_PB;
//     offset += shared_ofs[ajc(threadIdx_x, threadIdx_y)];
// 
//     if (threadIdx_y == 0)
//       ngb_list[offset++] = n_ngb;
// 
//     n_ngb = shared_ngb[ajc(threadIdx_x, threadIdx_y)];
//     for (int i = 0; i < n_ngb; i++)
//       ngb_list[offset + i] = local_ngb_list[i];
//   }
#endif
}




#define ajc(i, j) (i + blockDim_x*j)
__kernel void dev_evaluate_gravity_sixth_double(
                                     const          int        nj_total, 
                                     const          int        nj,
                                     const          int        ni_offset,    
                                     const          int        ni_total,
                                     const __global double4    *pos_j, 
                                     const __global double4    *pos_i,
                                           __global double4    *result_i,
                                     const          double     EPS2,
                                     const __global double4    *vel_j,
                                     const __global int        *id_j,                                     
                                     const __global double4    *vel_i,
                                     __out __global int        *id_i,
                                     __out __global float2     *dsminNNB,
                                     __out __global int        *ngb_count_i,
                                     __out __global int        *ngb_list,
                                     const __global double4    *acc_i_in,
                                     const __global double4    *acc_j,                                 
                                           __local  double4    *shared_pos) {

  const int tx = threadIdx_x;
  const int ty = threadIdx_y;
  const int bx =  blockIdx_x;
  const int Dim = blockDim_x*blockDim_y;

  __local double4 *shared_vel  = (__local double4*)&shared_pos[Dim];
  __local double4 *shared_accj = (__local double4*)&shared_vel[Dim];

  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const double4 pos     = pos_i[threadIdx_x + ni_offset];
  const int particleID  = id_i [threadIdx_x + ni_offset];
  const double4 vel     = vel_i[threadIdx_x + ni_offset];
  const double4 acc_in  = acc_i_in[threadIdx_x + ni_offset];


  const float LARGEnum = 1.0e10f;

  float2      ds2_min2;
  ds2_min2.x  = LARGEnum;
  ds2_min2.y  = as_float(-1);

  double4 acc;
  acc.x = acc.y = acc.z = acc.w = 0.0;
  double3 jrkNew = {0.0, 0.0, 0.0};  
  double3 snpNew = {0.0, 0.0, 0.0};

  int tile  = 0;
  int ni    = bx * (nj*blockDim_y) + nj*ty;
  const int offy = blockDim_x*ty;
#if 1
  for (int i = ni; i < ni+nj; i += blockDim_x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      shared_pos[addr]     = pos_j[i + tx];
      shared_vel[addr]     = (double4){
                                    vel_j[i + tx].x, 
                                    vel_j[i + tx].y,
                                    vel_j[i + tx].z, 
                                    as_float(id_j[i + tx])};
      shared_accj[addr]     = acc_j[i+tx];
    } else {
      shared_pos[addr]  = (double4){LARGEnum,LARGEnum,LARGEnum,0};
      shared_vel[addr]  = (double4){0.0, 0.0, 0.0,as_float(-1)}; 
      shared_accj[addr] = (double4){0.0,0.0,0.0,0.0};
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim_x, blockDim_x);
    const int j1 = j & (-32);



#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction(&ds2_min2, &n_ngb, local_ngb_list,
          &acc, &jrkNew, &snpNew, pos, vel, acc_in,
          shared_pos[offy+k], shared_vel[offy+k], shared_accj[offy+k],
          particleID, EPS2);

    for (int k = j1; k < j; k++) 
      body_body_interaction(&ds2_min2, &n_ngb, local_ngb_list,
          &acc, &jrkNew, &snpNew, pos, vel, acc_in,
          shared_pos[offy+k], shared_vel[offy+k], shared_accj[offy+k],
          particleID, EPS2);
    __syncthreads();

    tile++;
  } //end while
#endif



  __local double4 *shared_acc = (__local double4*)&shared_pos[0];
  __local double4 *shared_jrk = (__local double4*)&shared_acc[Dim];
  __local double3 *shared_snp = (__local double3*)&shared_jrk[Dim];

  const int addr = offy + tx;

  shared_acc[addr].x = acc.x; shared_acc[addr].y = acc.y;
  shared_acc[addr].z = acc.z; shared_acc[addr].w = acc.w;
  shared_jrk[addr]   = (double4) {jrkNew.x, jrkNew.y, jrkNew.z, 0};
  shared_snp[addr]   = (double3) {snpNew.x, snpNew.y, snpNew.z};
  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim_x; i < Dim; i += blockDim_x)
    {
      const int addr = i + tx;
      double4 acc1 = shared_acc[addr];
      double4 jrk1 = shared_jrk[addr];

      acc.x += acc1.x;
      acc.y += acc1.y;
      acc.z += acc1.z;
      acc.w += acc1.w;

      jrkNew.x += jrk1.x;
      jrkNew.y += jrk1.y;
      jrkNew.z += jrk1.z;

      snpNew.x += shared_snp[i + tx].x;
      snpNew.y += shared_snp[i + tx].y;
      snpNew.z += shared_snp[i + tx].z;    
    }       
  }
  __syncthreads();

     //Reduce neighbours info
  __local int    *shared_ngb = (__local int*  )&shared_pos[0];
  __local int    *shared_ofs = (__local int*  )&shared_ngb[Dim];
  __local float  *shared_nid = (__local float*)&shared_ofs[Dim];
  __local float  *shared_ds  = (__local float*)&shared_nid[Dim];
  
  shared_ngb[addr] = n_ngb;
  shared_ofs[addr] = 0;
  shared_ds [addr] = ds2_min2.x;
  shared_nid[addr] = ds2_min2.y;
     
  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim_x; i < Dim; i += blockDim_x)
    {
      const int addr = i + tx;
      
      if(shared_ds[addr] < ds2_min2.x)
      {
        ds2_min2.x = shared_ds[addr];
        ds2_min2.y = shared_nid[addr];
      }
      
      shared_ofs[addr] = min(n_ngb, NGB_PB);
      n_ngb           += shared_ngb[addr];      
    }
      n_ngb  = min(n_ngb, NGB_PB);
  }
  __syncthreads();
  int ngbListStart = 0;
  
  __global double4 *acc_i = (__global double4*)&result_i[0];
  __global double4 *jrk_i = (__global double4*)&result_i[ni_total];
  __global double4 *snp_i = (__global double4*)&result_i[ni_total*2];
  
  if (ty == 0) 
  {
    __global int *atomicVal = &ngb_count_i[NPIPES];
    if(threadIdx_x == 0)
    {
      int res          = atomic_xchg(&atomicVal[0], 1); //If the old value (res) is 0 we can go otherwise sleep
      int waitCounter  = 0;
      while(res != 0)
      {
        //Sleep
        for(int i=0; i < (1024); i++)
        {
          waitCounter += 1;
        }
        //Test again
        shared_ds[blockDim_x] = (float)waitCounter;
        res = atomic_xchg(&atomicVal[0], 1); 
      }
    }
    __syncthreads();
    
    float2 temp2; 
    temp2 = dsminNNB[tx+ni_offset];
    if(ds2_min2.x <  temp2.y)
    {
      temp2.y = ds2_min2.x;
      temp2.x = ds2_min2.y;
      dsminNNB[tx+ni_offset] = temp2;
    }

    
    acc_i[tx+ni_offset].x += acc.x; acc_i[tx+ni_offset].y += acc.y;
    acc_i[tx+ni_offset].z += acc.z; acc_i[tx+ni_offset].w += acc.w;
    jrk_i[tx+ni_offset].x += jrkNew.x; jrk_i[tx+ni_offset].y += jrkNew.y;
    jrk_i[tx+ni_offset].z += jrkNew.z; 
    snp_i[tx+ni_offset].x += snpNew.x; snp_i[tx+ni_offset].y += snpNew.y;
    snp_i[tx+ni_offset].z += snpNew.z; 
    
    ngbListStart                = ngb_count_i[tx+ni_offset];
    ngb_count_i[tx+ni_offset]  += n_ngb;

    if(threadIdx_x == 0)
    {
      atomic_xchg(&atomicVal[0], 0); //Release the lock
    }
  }//end atomic section

  //Write the neighbour list
  {
    //Share ngbListStart with other threads in the block
    const int yBlockOffset = shared_ofs[addr];
    __syncthreads();
    if(ty == 0)
    {
      shared_ofs[threadIdx_x] = ngbListStart;
    }
    __syncthreads();
    ngbListStart    = shared_ofs[threadIdx_x];


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
  }

}



/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni
 */
__kernel void dev_reduce_forces(
                                __global double4 *acc_i_temp, 
                                __global double4 *jrk_i_temp,
                                __global double  *ds_i_temp,
                                __global int     *ngb_count_i_temp,
                                __global int     *ngb_list_i_temp,
                                __global double4 *result_i,
                                __global double  *ds_i,
                                __global int     *ngb_count_i,
                                __global int     *ngb_list,                                
                                         int     offset_ni_idx,
                                         int     ni_total,
                               __global double4 *snp_i_temp,

                               __local  double4 *shared_acc ) {

//    extern __shared__ float4 shared_acc[];
 __local  double4 *shared_jrk = (__local double4*)&shared_acc[blockDim_x];
 __local  double4 *shared_snp = (__local double4*)&shared_jrk[blockDim_x];
 __local  int    *shared_ngb  = (__local int*   )&shared_snp[blockDim_x];
 __local  int    *shared_ofs  = (__local int*   )&shared_ngb[blockDim_x];
 __local  double  *shared_ds  = (__local double* )&shared_ofs[blockDim_x];


  int index = threadIdx_x * gridDim_x + blockIdx_x;

  //Early out if we are a block for non existent particle
  if((blockIdx_x + offset_ni_idx) >= ni_total)
    return;

//   shared_acc[threadIdx.x] = acc_i[index];
//   shared_jrk[threadIdx.x] = jrk_i[index];
//   shared_ds [threadIdx.x] = vel_i[offset_ds + index].w;

  //Convert the data to floats
  shared_acc[threadIdx_x] = acc_i_temp[index];
  shared_jrk[threadIdx_x] = jrk_i_temp[index];
  shared_snp[threadIdx_x] = snp_i_temp[index];
  shared_ds [threadIdx_x] = ds_i_temp [index];

  shared_ngb[threadIdx_x] = ngb_count_i_temp[index];
  shared_ofs[threadIdx_x] = 0;

  __syncthreads();


  int n_ngb = shared_ngb[threadIdx_x];
  if (threadIdx_x == 0) {
    double4 acc0 = shared_acc[0];
    double4 jrk0 = shared_jrk[0];
    double4 snp0 = shared_snp[0];
    double   ds0 = shared_ds [0];

    for (int i = 1; i < blockDim_x; i++) {
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

      shared_ofs[i] = min(n_ngb + 1, NGB_PB);
      n_ngb += shared_ngb[i];
    }
    n_ngb = min(n_ngb, NGB_PB);

    jrk0.w = (int)(jrk0.w);

    //Store the results
    result_i    [blockIdx_x + offset_ni_idx]              = acc0;
    result_i    [blockIdx_x + offset_ni_idx + ni_total]   = jrk0;
    result_i    [blockIdx_x + offset_ni_idx + 2*ni_total] = snp0;
    ds_i        [blockIdx_x + offset_ni_idx] = ds0;
    ngb_count_i [blockIdx_x + offset_ni_idx] = n_ngb;
  }
  __syncthreads();

  //Compute the offset of where to store the data and where to read it from
  //Store is based on ni, where to read it from is based on thread/block
  int offset     = (offset_ni_idx + blockIdx_x)  * NGB_PB + shared_ofs[threadIdx_x];
  int offset_end = (offset_ni_idx + blockIdx_x)  * NGB_PB + NGB_PB;
  int ngb_index  = threadIdx_x * NGB_PB + blockIdx_x * NGB_PB*blockDim_x;


  n_ngb = shared_ngb[threadIdx_x];
  __syncthreads();
  for (int i = 0; i < n_ngb; i++)
  {
    if (offset + i < offset_end){
        ngb_list[offset + i] = ngb_list_i_temp[ngb_index + i];
    }
  }
}



__kernel void dev_evaluate_gravity_reduce()
{}

__kernel void  dev_no_predictor(int nj,
                                        double  t_i_d,
                               __global    double2 *t_j,
                               __global    double4 *Ppos_j,
                               __global    double4 *Pvel_j,
                               __global    double4 *pos_j,
                               __global    double4 *vel_j,
                               __global    double4 *acc_j,
                               __global    double4 *jrk_j,
                               __global    double4 *Pacc_j,
                               __global    double4 *snp_j,
                               __global    double4 *crk_j){
  int index = blockIdx_x * blockDim_x + threadIdx_x;

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


