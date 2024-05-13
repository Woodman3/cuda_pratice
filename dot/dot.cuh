const int WARP_SIZE=32;

template<int warpSize=WARP_SIZE>
__forceinline__ __device__ float reduce(float val){
    for(int offset = warpSize>>1;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

template <int numThread=1024>
__global__ void gpu_dot(const float *a ,const float *b,float *c,int n){
    int i = blockIdx.x*numThread + threadIdx.x;
    int tx=threadIdx.x;
    float sum = i<n?(a[i]*b[i]):0.0f;
    sum =reduce(sum);
    constexpr int num_warp=numThread/WARP_SIZE;
    extern __shared__ float reduce_sum[num_warp];
    int laneid = tx%WARP_SIZE;
    int warp_id = tx/WARP_SIZE;
    if(laneid==0){
        reduce_sum[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id==0){
        sum = laneid<num_warp? reduce_sum[laneid]:0.0;
        sum = reduce(sum);
        if(laneid==0){
            atomicAdd(c,sum);
        }
    }
}


