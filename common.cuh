#include<iostream>
#include <cublas_v2.h>
#include <curand.h>

#define CHECK(...)  \
do{ \
   const cudaError_t err = __VA_ARGS__;\
   if(err!=cudaSuccess){ \
        cout<<err<<endl; \
        cout<<cudaGetErrorString(err)<<endl; \
   }  \
}while(0)

const int WARP_SIZE=32;

template<int warpSize=WARP_SIZE>
__forceinline__ __device__ float reduce_sum(float val){
    #pragma unroll
    for(int offset = warpSize>>1;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

template<int blockSize=1024>
__forceinline__ __device__ float block_reduce_sum(float val){
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x%WARP_SIZE;
    int wid = threadIdx.x/WARP_SIZE;
    val = reduce_sum(val);
    if(lane==0){
        shared[wid]=val;
    }
    __syncthreads();
    if(wid==0){
        val = (lane<blockSize>>5)?shared[lane]:0;
        val = reduce_sum(val);
    }
    return val;
}
