#include <iostream>
#include <cstdlib>
#include <random>
#include <cublas_v2.h>
#include <curand.h>
#include "gemv.cuh"

void cup_gemv(const float* A, const float* x, float* y, int m, int n){
    for(int i=0;i<m;i++){
        y[i]=0.0;
        for(int j=0;j<n;j++){
            y[i]+=A[OFFSET(i,j,n)]*x[j];
        }
    }
}


// Wrapper function
void dot_wrapper(const float* A, const float* x, int m, int n) {

    float *y1,*y2;
    cudaMalloc(&y1, m * sizeof(float));
    cudaMalloc(&y2, m * sizeof(float));
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);
    int blockSize = 256;
    // dim3 blockSize = {32,8,1};
    int numBlocks = (m + blockSize - 1) / blockSize;
    // dim3 numBlocks = {1, (m + blockSize.y - 1) / blockSize.y, 1};
    // Launch kernel
    gpu_gemv<<<numBlocks, blockSize>>>(A, x, y1, m, n);

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
    cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, A, n, x, 1, &beta, y2, 1);

    // Destroy the CUBLAS handle
    cublasDestroy(handle);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUBLAS execution time: " << milliseconds << " ms" << std::endl;
    
    float *h_y3 = new float[m];
    float *h_A = new float[m*n];
    float *h_x = new float[n];
    cudaMemcpy(h_A, A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cup_gemv(h_A,h_x,h_y3,m,n);
    
    float *h_y1 = new float[m];
    float *h_y2 = new float[m];
    cudaMemcpy(h_y1, y1, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2, y2, m * sizeof(float), cudaMemcpyDeviceToHost);
    // Check the correctness of the result
    bool correct = true;
    for (int i = 0; i < m; i++) {
        if (abs(h_y1[i] - h_y2[i]) > 1e-6) {
            std::cout<< h_y1[i] << " " << h_y2[i] <<" "<<h_y3[i]<<" "<<i<< std::endl;
            
            std::cout << "Result is incorrect!" << std::endl;
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "Result is correct!" << std::endl;
    // Free memory
    delete[] h_y1;
    delete[] h_y2;
    cudaFree(y1);
    cudaFree(y2);
}

// Data generation function
void generateData(float* A, float* x, int m, int n) {
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerateUniform(gen, A, m * n);
    curandGenerateUniform(gen, x, n);

    curandDestroyGenerator(gen);
}

int main() {
    int m = 4096; // Number of rows
    int n = 128; // Number of columns
    // float *h_A = new float[m * n];
    // float *h_x = new float[n];
    // for (int i=0;i<m*n;i++){
    //     h_A[i]=i;
    // }
    // for(int i=0;i<n;i++){
    //     h_x[i]=i;
    // }
    // Allocate memory for matrices and vectors
    float *A,*x ;
    cudaMalloc(&A, m * n * sizeof(float));
    cudaMalloc(&x, n * sizeof(float));

    // Generate random data
    generateData(A, x, m, n);
    // cudaMemcpy(A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform gemv operation
    dot_wrapper(A, x, m, n);

    // Print the result

    return 0;
}