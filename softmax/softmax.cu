#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA softmax kernel
__global__ void softmax(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float max_val = input[idx];
        for (int i = 0; i < size; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += expf(input[i] - max_val);
        }
        
        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

// Wrapper function for calling the CUDA softmax kernel
void cudaSoftmax(float* input, float* output, int size) {
    float* d_input;
    float* d_output;
    
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to generate random input data
void generateData(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int size = 1000;
    float* input = new float[size];
    float* output = new float[size];
    
    generateData(input, size);
    
    cudaSoftmax(input, output, size);
    
    // Print the output
    for (int i = 0; i < size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] input;
    delete[] output;
    
    return 0;
}