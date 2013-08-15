/*

Sapporo 2 device kernels
GRAPE5

Version 1.0
CUDA single precisin kernels

*/

#include "OpenCL/sharedKernels.cl"


__inline void body_body_interaction(inout float4     *acc_i,                                       
                                    const float4     pos_i,                                       
                                    const float4     pos_j, 
                                    const float      EPS2) {

  const float3 dr = {(pos_j.x - pos_i.x) , (pos_j.y-  pos_i.y) ,(pos_j.z - pos_i.z)};   // 3x3 = 9 FLOP

  const float ds2 = ((dr.x*dr.x + (dr.y*dr.y)) + dr.z*dr.z);

  //EPS is in GRAPE5 always non-zero, if it is zero well then behaviour is undefined
  const float inv_ds  = rsqrt(ds2 + EPS2);

  /*if((ds2 + EPS2) == 0)
    inv_ds = 0;
  */

  const float inv_ds2 = inv_ds*inv_ds;                         
  const float inv_ds3 = pos_j.w * inv_ds*inv_ds2;             //pos_j.w is mass
  
  // 3*4 + 3 = 15 FLOP
  (*acc_i).x += ((inv_ds3 * dr.x));
  (*acc_i).y += ((inv_ds3 * dr.y));
  (*acc_i).z += ((inv_ds3 * dr.z));
  
  (*acc_i).w += (pos_j.w * inv_ds);      //Potential
}


/*
 *  blockDim.x = ni
 *  gridDim.x  = 16, 32, 64, 128, etc. 
 */ 

__kernel void dev_evaluate_gravity_second_float(
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
                                           __local  float4     *shared_pos)
{
  const int tx = threadIdx_x;
  const int ty = threadIdx_y;
  const int bx =  blockIdx_x;
  const int Dim = blockDim_x*blockDim_y;
  
  const float EPS2 = (float)EPS2_d;

  float4 pos;
  pos.x = pos_i[tx+ni_offset].x; pos.y = pos_i[tx+ni_offset].y;
  pos.z = pos_i[tx+ni_offset].z; pos.w = pos_i[tx+ni_offset].w;

  const float LARGEnum = 1.0e10f;
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};


  int tile = 0;
  int ni    = bx * (nj*blockDim_y) + nj*ty;
  const int offy = blockDim_x*ty;
  for (int i = ni; i < ni+nj; i += blockDim_x)
  {
    const int addr = offy + tx;

    if (i + tx < nj_total) 
    {
      const double4 jp     = pos_j[i + tx];
      shared_pos[addr].x = jp.x;
      shared_pos[addr].y = jp.y;
      shared_pos[addr].z = jp.z;
      shared_pos[addr].w = jp.w;
    } else {
      shared_pos[addr].x = LARGEnum;
      shared_pos[addr].y = LARGEnum;
      shared_pos[addr].z = LARGEnum;
      shared_pos[addr].w = 0.0f; 
    }

    __syncthreads();

    const int j  = min(nj - tile*blockDim_x, blockDim_x);
    const int j1 = j & (-32);

#pragma unroll 32
    for (int k = 0; k < j1; k++) 
      body_body_interaction(&acc, pos, shared_pos[offy+k], EPS2);

    for (int k = j1; k < j; k++) 
      body_body_interaction(&acc, pos, shared_pos[offy+k], EPS2);

    __syncthreads();

    tile++;
  } //end while

  __local  float4 *shared_acc = (__local float4*)&shared_pos[0];  
  acc.w = -acc.w;
  
  const int addr = offy + tx;
  shared_acc[addr] = acc;
  __syncthreads();

  if (ty == 0) 
  {
    for (int i = blockDim_x; i < Dim; i += blockDim_x)
    {
      float4 acc1 = shared_acc[i + tx];
      acc.x += acc1.x;
      acc.y += acc1.y;
      acc.z += acc1.z;
      acc.w += acc1.w;
    }
  }
  __syncthreads();

  __global double4 *acc_i = (__global double4*)&result_i[0];

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
        __local float* shared_ds = (__local float*)&shared_pos[0];
        shared_ds[blockDim_x] = (float)waitCounter;
        res = atomic_xchg(&atomicVal[0], 1); 
      }
    }
    __syncthreads();
    
    acc_i[tx+ni_offset].x += acc.x; acc_i[tx+ni_offset].y += acc.y;
    acc_i[tx+ni_offset].z += acc.z; acc_i[tx+ni_offset].w += acc.w;
   
    if(threadIdx_x == 0)
    {
      atomic_xchg(&atomicVal[0], 0); //Release the lock
    }
  }//end atomic section
  

}



