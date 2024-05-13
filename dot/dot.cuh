template <int numThread=128>
__global__ void gpu_dot(const float *a ,const float *b,float *c,int n){
    int i = blockIdx.x*numThread + threadIdx.x;
    if(i < n){
        c[i] = a[i] * b[i];
    }
}