#include <stdio.h>
#include <random>

#define WARP_SIZE 32

template <unsigned int warpSize=WARP_SIZE>
__forceinline__ __device__ float warp_reduce(float val) {
    for(int offset = warpSize>>1 ; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<unsigned int threadNum = 1024>
__global__ void reduce_kernel(float* input, float* output, int size) {
    constexpr int numWarps = threadNum / WARP_SIZE;
    __shared__ float shared[numWarps];

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = (tid < size) ? input[tid] : 0.0f;
    for(int i = tid + stride; i < size; i += stride) {
        sum += input[i];
    }
    sum = warp_reduce(sum);
    __syncthreads();
    if (lane == 0) 
        shared[warp] = sum;
    __syncthreads();
    sum = (lane < numWarps) ? shared[lane] : 0.0f;
    if (warp==0)
        sum = warp_reduce(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

float reduce(float* input, int size) {
    float* deviceInput;
    float* deviceOutput;

    cudaMalloc((void**)&deviceInput, size * sizeof(float));
    cudaMalloc((void**)&deviceOutput, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(deviceInput, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = 64;
    cudaEventRecord(start);

    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);

    float output;
    cudaMemcpy(&output, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;
}

float generateTestData(float* input, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    float r=0.0;
    for (int i = 0; i < size; i++) {
        input[i] = dis(gen);
        r += input[i]; 
    }
    return r;
}
int main() {
    int size = 1<<25;
    float* input = new float[size];
    float ans = generateTestData(input, size);

    float result = reduce(input, size);

    printf("Reduced value: %f, right is %f\n", result, ans);

    return 0;
}