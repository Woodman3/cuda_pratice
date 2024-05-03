#include <stdio.h>
#include <random>

template <unsigned int warpSize=32>
__forceinline__ __device__ float warp_reduce(float val) {
    for(int offset = warpSize>>1 ; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<unsigned int threadNum = 256>
__global__ void reduce_kernel(float* input, float* output, int size) {
    __shared__ float shared[threadNum];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        shared[tid] = input[i];
    }
    else {
        shared[tid] = 0;
    }

    __syncthreads();

    for (unsigned int stride = blockDim.x>>1; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float val = shared[tid];
    val = warp_reduce(val);

    if (tid == 0) {
        atomicAdd(output, val);
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
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