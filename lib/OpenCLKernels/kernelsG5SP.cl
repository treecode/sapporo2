/*

Sapporo 2 device kernels
GRAPE5

Version 1.0
CUDA single precisin kernels

*/

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

#define inout
#define __out


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


//should make this depending on if we use Fermi or GT80/GT200
// #define ajc(i, j) (i + __mul24(blockDim.x,j))
// #define ajc(i, j) (i + blockDim.x*j)

/*
 *  blockDim.x = ni
 *  gridDim.x  = 16, 32, 64, 128, etc. 
 */ 

__kernel void dev_evaluate_gravity(
                                     const          int        nj_total, 
                                     const          int        nj,
                                     const          int        ni_offset,
                                     const __global double4    *pos_j,                                      
                                     const __global double4    *pos_i,
                                     __out __global double4    *acc_i,                                      
                                     const          double     EPS2_d,
                                           __local  float4     *shared_pos)
{
  
//   __shared__ char shared_mem[NTHREADS*(sizeof(float4))];
//   float4 *shared_pos = (float4*)&shared_mem[0];
  const int tx = threadIdx_x;
  const int ty = threadIdx_y;
  const int bx =  blockIdx_x;
  const int Dim = blockDim_x*blockDim_y;
  
  const float EPS2 = (float)EPS2_d;

  float4 pos;
  pos.x = pos_i[tx+ni_offset].x; pos.y = pos_i[tx+ni_offset].y;
  pos.z = pos_i[tx+ni_offset].z; pos.w = pos_i[tx+ni_offset].w;

  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  const float LARGEnum = 1.0e10f;

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
  
  if (threadIdx_y == 0) {
    //Convert results to double and write  
    acc_i[blockIdx_x * blockDim_x + threadIdx_x] = (double4){acc.x, acc.y, acc.z, acc.w};
  }
}



/*
 *  blockDim.x = #of block in previous kernel
 *  gridDim.x  = ni
 */ 
__kernel void dev_reduce_forces(
                                __global double4 *acc_i_temp,
                                __global double4 *acc_i,
                                         int      offset_ni_idx,
                                __local  float4  *shared_acc)
{
  
  int index = threadIdx_x * gridDim_x + blockIdx_x;

  //Convert the data to floats
  shared_acc[threadIdx_x] = (float4){acc_i_temp[index].x, acc_i_temp[index].y,
                                     acc_i_temp[index].z, acc_i_temp[index].w};
         
  __syncthreads();

  if (threadIdx_x == 0) {
    float4 acc0 = shared_acc[0];

    for (int i = 1; i < blockDim_x; i++) {
      acc0.x += shared_acc[i].x;
      acc0.y += shared_acc[i].y;
      acc0.z += shared_acc[i].z;
      acc0.w += shared_acc[i].w;
    }
    //Store the results
    acc_i[blockIdx_x+offset_ni_idx] = (double4){acc0.x, acc0.y, acc0.z, acc0.w};
  }
  __syncthreads();  
}


/*
 * Function that moves the (changed) j-particles
 * to the correct address location.
*/
__kernel void dev_copy_particles(int nj,                                                                              
                                 __global double4   *pos_j, 
                                 __global double4   *pos_j_temp,
                                 __global int       *address_j) {
  const uint bid = blockIdx_y * gridDim_x + blockIdx_x;
  const uint tid = threadIdx_x;
  const uint index = bid * blockDim_x + tid;
  //Copy the changed particles
  if (index < nj)
  {   
     pos_j[address_j[index]] = pos_j_temp[index];
  }
}

/*

Function to predict the particles
DS version

*/
__kernel void dev_predictor(int nj) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  //NOt used in GRAPE5
}

