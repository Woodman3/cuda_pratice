#include <iostream>
#include <cstdlib>
#include <random>
#include <cublas_v2.h>
#include <curand.h>
#include "dot.cuh"

void cpu_dot(const float *a ,const float *b,float *c,int n  ){
    for(int i = 0; i < n; i++){
        *c += a[i] * b[i];
    }
}


// Wrapper function
void dot_wrapper(const float* a, const float* b, int n) {

    float *c1,*c2;
    cudaMalloc(&c1, sizeof(float));
    cudaMalloc(&c2, sizeof(float));
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    // Launch kernel
    gpu_dot<<<numBlocks, blockSize>>>(a,b, c1, n);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;


    // Start the timer


    // Perform gemv operation using CUBLAS
    float alpha = 1.0;

    float beta = 0.0;
    cublasHandle_t handle; // Declare the CUBLAS handle

    // Initialize the CUBLAS library
    cublasCreate(&handle);

    cudaEventRecord(start);
    // Perform the matrix-vector multiplication using CUBLAS
    cublasSdot(handle, n, a, 1, b, 1, c2);

    // Destroy the CUBLAS handle
    cublasDestroy(handle);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUBLAS execution time: " << milliseconds << " ms" << std::endl;
    
    float *h_c3 = new float;
    float *h_a = new float[n];
    float *h_b = new float[n];
    cudaMemcpy(h_a, a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_dot(h_a,h_b,h_c3,n);
    
    float *h_c1 = new float;
    float *h_c2 = new float;
    cudaMemcpy(h_c1, c1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2, c2, sizeof(float), cudaMemcpyDeviceToHost);
    // Check the correctness of the result
    if (abs(*h_c1 - *h_c2) > 1e-4) {
        std::cout<< h_c1 << " " << h_c2 <<" "<<h_c3<<std::endl;
        std::cout << "Result is incorrect!" << std::endl;
    }else {
        std::cout << "Result is correct!" << std::endl;
    }
    // Free memory
    delete[] h_c1;
    delete[] h_c2;
    cudaFree(c1);
    cudaFree(c2);
}

// Data generation function
void generateData(float *a,float *b, int n) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerateUniform(gen, a, n);
    curandGenerateUniform(gen, b, n);

    curandDestroyGenerator(gen);
}

int main() {
    int n = 128; // Number of columns
    float *a, *b;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));

    // Generate random data
    generateData(a,b, n);
    // cudaMemcpy(A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform gemv operation
    dot_wrapper(a, b, n);

    // Print the result

    return 0;
}