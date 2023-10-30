/*

Sapporo 2 device kernels

Version 1.0
OpenCL Fourth order DoubleSingle kernels

*/


#include "sharedKernels.cl"

#define CAST 1

// __inline float2 body_body_interaction(
__inline void body_body_interaction(
                                    inout struct devForce *acc_i, 
                                    inout float4 *jrk_i,
                                    inout float2  *ds2_min,     
                                    inout int   *n_ngb,
                                    inout __private int *ngb_list,
                                    const DS4     pos_i, 
                                    const float4  vel_i,
                                    const DS4     pos_j, 
                                    const float4  vel_j,
                                    //const float  EPS2,
                                    float  EPS2,
                                    const int iID
                                    ) {

  const int jID   = as_int(pos_j.w.y);

  if (iID != jID)    /* assuming we always need ngb */
  {

    const float3 dr = {(pos_j.x.x - pos_i.x.x) + (pos_j.x.y - pos_i.x.y),
                       (pos_j.y.x - pos_i.y.x) + (pos_j.y.y - pos_i.y.y),
                       (pos_j.z.x - pos_i.z.x) + (pos_j.z.y - pos_i.z.y)};   // 3x3 = 9 FLOP
   

    const float ds2 = ((dr.x*dr.x) + dr.y*dr.y) + dr.z*dr.z;
#if 1
    (*ds2_min) = ((*ds2_min).x < ds2) ? (*ds2_min) : (float2){ds2, pos_j.w.y};


    /* WARRNING: In case of the overflow, the behaviour will be different from the original version */
    if (ds2 <= pos_i.w.x)
    {
      ngb_list[(*n_ngb) & (NGB_PB-1)] = jID;
      (*n_ngb)++;
    }

#endif

    float inv_ds = 0.0f;
    inv_ds = rsqrt(ds2+EPS2);
    //     inv_ds = (jID == iID) ? 0.0f : inv_ds; //451800

    const float mass   = pos_j.w.x;
    const float minvr1 = mass*inv_ds; 
    const float  invr2 = inv_ds*inv_ds; 
    const float minvr3 = minvr1*invr2;

    // 3*4 + 3 = 15 FLOP
    (*acc_i).x += minvr3 * dr.x;
    (*acc_i).y += minvr3 * dr.y;
    (*acc_i).z += minvr3 * dr.z;
    (*acc_i).w += (-1.0f)*minvr1;
    //(*acc_i).w += 1;
#if 1
    const float3 dv  =  {vel_j.x - vel_i.x, vel_j.y - vel_i.y, vel_j.z -  vel_i.z};
    const float drdv = (-3.0f) * (minvr3*invr2) * (((dr.x*dv.x) + dr.y*dv.y) + dr.z*dv.z);

    (*jrk_i).x += minvr3 * dv.x + drdv * dr.x;  
    (*jrk_i).y += minvr3 * dv.y + drdv * dr.y;
    (*jrk_i).z += minvr3 * dv.z + drdv * dr.z;
#endif
    // TOTAL 50 FLOP (or 60 FLOP if compared against GRAPE6)  
  }

}

__kernel
//__attribute__((reqd_work_group_size(256,1,1)))
//__attribute__((work_group_size_hint(256,1,1)))
void dev_evaluate_gravity_fourth_DS(
                                     const          int        nj_total, 
                                     const          int        nj,
                                     const          int        ni_offset,    
                                     const          int        ni_total,
                                     const __global double4    *pos_j, 
                                     const __global double4    *pos_i,
                                           __global double4    *result_i,
                                     const          double     EPS2_d,
                                     const __global double4    *vel_j,
                                     const __global int        *id_j,                                     
                                     const __global double4    *vel_i,
                                     __out __global int        *id_i,
                                     __out __global float2     *dsminNNB,
                                     __out __global int        *ngb_count_i,
                                     __out __global int        *ngb_list,
                                     const __global double4    *acc_i_in,
                                     const __global double4    *acc_j,                                 
                                           __local  DS4        *shared_pos) {

  const int tx = threadIdx_x;
  const int ty = threadIdx_y;
  const int bx =  blockIdx_x;
  const int Dim = blockDim_x*blockDim_y;

  const int gid = blockIdx_x*blockDim_x + threadIdx_x;

  __local float4 *shared_vel = (__local float4*)&shared_pos[Dim];


  int local_ngb_list[NGB_PB + 1];
  int n_ngb = 0;

  const float EPS2 = (float)EPS2_d;

  DS4 pos;
  pos.x = to_DS(pos_i[tx+ni_offset].x); 
  pos.y = to_DS(pos_i[tx+ni_offset].y);
  pos.z = to_DS(pos_i[tx+ni_offset].z);
  pos.w = to_DS(pos_i[tx+ni_offset].w);

  const int iID    = id_i[tx+ni_offset];
  const float4 vel = (float4){vel_i[tx+ni_offset].x, vel_i[tx+ni_offset].y, 
                              vel_i[tx+ni_offset].z, vel_i[tx+ni_offset].w};

  const float LARGEnum = 1.0e10f;

  float2  ds2_min2;
  ds2_min2.x = LARGEnum;
  ds2_min2.y = as_float(-1);

  struct devForce acc;
  acc.x = acc.y = acc.z = acc.w = 0.0f;
  float4   jrk = {0.0f, 0.0f, 0.0f, 0.0f};

  int tile  = 0;
  int ni    = bx * (nj*blockDim_y) + nj*ty; //Default
  const int offy = blockDim_x*ty;

  for (int i = ni; i < ni+nj; i += blockDim_x)  //Default
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      const double4 jp     = pos_j[i + tx];
      shared_pos[addr].x   = to_DS(jp.x);
      shared_pos[addr].y   = to_DS(jp.y);
      shared_pos[addr].z   = to_DS(jp.z);
      shared_pos[addr].w   = to_DS(jp.w);     
      shared_pos[addr].w.y = as_float(id_j[i + tx]);

      shared_vel[addr]     = (float4){vel_j[i + tx].x, 
                                      vel_j[i + tx].y, 
                                      vel_j[i + tx].z, 
                                      vel_j[i + tx].w};
    } else {
      shared_pos[addr].x = (float2){LARGEnum, 0.0f};
      shared_pos[addr].y = (float2){LARGEnum, 0.0f};
      shared_pos[addr].z = (float2){LARGEnum, 0.0f};
      shared_pos[addr].w = (float2){0.0f,  as_float(-1)}; 
      shared_vel[addr]   = (float4){0.0f, 0.0f, 0.0f, 0.0f};
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim_x, blockDim_x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
        body_body_interaction(
          &acc, &jrk, &ds2_min2, &n_ngb, local_ngb_list,
          pos, vel, 
          shared_pos[offy+k], 
          shared_vel[offy+k],
          EPS2, iID);

    //No unroll here, AMD does not like it....
    for (int k = j1; k < j; k++) 
        body_body_interaction(
          &acc, &jrk, &ds2_min2, &n_ngb, local_ngb_list,
          pos, vel,
          shared_pos[offy+k],
          shared_vel[offy+k],
          EPS2, iID);

    __syncthreads();

    tile++;
  } //end while

  __local float4 *shared_acc = (__local float4*)&shared_pos[0];
  __local float4 *shared_jrk = (__local float4*)&shared_acc[blockDim_x*blockDim_y];

  const int addr = offy + tx;

  shared_acc[addr].x = acc.x; shared_acc[addr].y = acc.y;
  shared_acc[addr].z = acc.z; shared_acc[addr].w = acc.w;
  shared_jrk[addr]   = jrk;
  __syncthreads();

  if (ty == 0)
  {
    for (int i = blockDim_x; i < Dim; i += blockDim_x)
    {
      const int addr = i + tx;
      float4 acc1 = shared_acc[addr];
      float4 jrk1 = shared_jrk[addr];

      acc.x += acc1.x;
      acc.y += acc1.y;
      acc.z += acc1.z;
      acc.w += acc1.w;

      jrk.x += jrk1.x;
      jrk.y += jrk1.y;
      jrk.z += jrk1.z;
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
      temp2.y = ds2_min2.x; //distance
      temp2.x = ds2_min2.y; //neighbour id
      dsminNNB[tx+ni_offset] = temp2;
    }

    
    acc_i[tx+ni_offset].x += acc.x; acc_i[tx+ni_offset].y += acc.y;
    acc_i[tx+ni_offset].z += acc.z; acc_i[tx+ni_offset].w += acc.w;
    jrk_i[tx+ni_offset].x += jrk.x; jrk_i[tx+ni_offset].y += jrk.y;
    jrk_i[tx+ni_offset].z += jrk.z; 
    
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
}//end function



