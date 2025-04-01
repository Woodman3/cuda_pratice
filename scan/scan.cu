#include "../common.cuh"
// template<typename T,int warpSize = WARP_SIZE,class OP>
// __forceinline__ __device__ T warp_scan_up(T* input,T* output,OP op){

//     for(int offset =1;offset<warpSize,offset<<=1){
//         if(offset & )
//         val=op(val,__shfl_up_sync(0xffffffff,val,offset));
//     }
// }

// template<typaname T,int blockSize=1024,class OP>
// __forceinline__ __device__ T block_scan(int n,OP op){
//     int tid = threadIdx.x;
//     int i = ( tid << 1 ) + 1;
//     __shared__ T[WARP_SIZE];

// }

template<typename T,int threadNum = 1024>
__global__ void blocK_scan(T* input,T* output,int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;
    // int lane = i % WARP_SIZE;
    // int warp = i / WARP_SIZE;
    __shared__ T temp[threadNum<<1];
    temp[i<<1] = input[i<<1];
    temp[i<<1+1] = input[i<<1+1];
    int offset = 1;
    for(;offset<(n>>1);offset<<=1){
        int right = (i<<1+1)*offset;
        int left = (i<<1)*offset;
        if(right<n){
            temp[right] = temp[left]+temp[right];
        }
        __syncthreads();
    }
}

template<typename T>
__global__ void seq_scan(T* input,T* output,T init,int n){
    if(threadIdx.x==0){
        output[0] = init;
        for(int i=1;i<n;i++){
            output[i] =input[i-1]+output[i-1];
        }
    }
}

int main(){
    int n=2048;
    int block_size = 2048;
    int block_num = (n+block_size-1)/block_size;

    float *a,*b;
    cudaMalloc(&a,n*sizeof(float));
    cudaMalloc(&b,n*sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateUniform(gen, a,n);
    curandDestroyGenerator(gen);
    float init = 0.0;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan<<<1,1>>>(a,b,init,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);

    return 0;
}