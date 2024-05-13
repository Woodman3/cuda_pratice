#define OFFSET(row,col,R) (((row)*(R))+(col))
const int warpsize = 32;
template<int warpsize>
__forceinline__ __device__ float warp_sum(float sum){
    #pragma unroll
    for(int i=warpsize>>1;i>0;i>>=1){
        sum+=__shfl_down_sync(0xFFFFFFFF,sum,i);
    }
    return sum;
}

// Gemv function
// x: n x 1 vector
// A: m x n matrix
// y: m x 1 vector
__global__ void gpu_gemv(const float* A, const float* x, float* y, int m, int n) {
    int currentrow = blockIdx.y * blockDim.y + threadIdx.y;
    if(currentrow < m){
        float sum=0;
        int k = n/warpsize;
        int col = blockIdx.x*warpsize + threadIdx.x*k;
        #pragma unroll
        for(int i=0;i<k;i++){
            sum+=A[OFFSET(currentrow,col+i,n)]*x[col+i];
        } 
        sum=warp_sum<warpsize>(sum);
        if(threadIdx.x==0){
            y[currentrow]=sum;
        }
    }
}
