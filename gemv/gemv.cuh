#define OFFSET(row,col,R) (((row)*(R))+(col))
// Gemv function
// x: n x 1 vector
// A: m x n matrix
// y: m x 1 vector
__global__ void gpu_gemv(const float* A, const float* x, float* y, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float temp = 0.0;
        for (int i = 0; i < n; i++) {
            temp += A[OFFSET(idx,i,n)] * x[i];
        }
        y[idx]=temp;
    }
}
