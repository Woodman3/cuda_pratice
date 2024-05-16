#include <iostream>
#include <cmath>
#include"../common.cuh"

// CUDA softmax kernel
template<int NUM_THREAD=1024>
__global__ void softmax(float* input, float* output,float* total ,int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float exp_val = idx<size?expf(input[idx]):0.0f;
    float sum = block_reduce_sum(exp_val);
    if(threadIdx.x==0){
        atomicAdd(total,sum);
    }
    // __threadfence();
    // if(idx<size){
    //     output[idx] = exp_val/(*total);
    // }
}

template<int NUM_THREAD=1024>
__global__ void tempfun(float*input,float* output,float*total,int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size){
        output[idx] =expf(input[idx])/(*total);
    }
}

void cpu_softmax(float* input,float* output,int size){
    float total=0.0f;
    for(int i=0;i<size;i++){
        total+=expf(input[i]);
    }
    std::cout<<total<<std::endl;
    for(int i=0;i<size;i++){
        output[i] = expf(input[i])/total;
    }
}

// Wrapper function for calling the CUDA softmax kernel
void cudaSoftmax(float* input, float* output, int size) {
    float* d_input;
    float* d_output;
    float* d_total;
    
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    cudaMalloc((void**)&d_total, sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_total,0,sizeof(float));
    
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output,d_total, size);
    tempfun<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,d_total,size);
    
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    float* h_total = new float;
    cudaMemcpy(h_total,d_total,sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"device total = "<<*h_total<<std::endl;
    
    float* h_y1=new float[size];
    cpu_softmax(input,h_y1,size);
    float* h_y2=new float[size];
    cudaMemcpy(h_y2,d_output,size*sizeof(float),cudaMemcpyDeviceToHost);
    bool flag=true;
    for(int i=0;i<size;i++){
        if(fabs(h_y1[i]-h_y2[i])>1e-6){
            std::cout<<"Error at "<<i<<" "<<h_y1[i]<<" "<<h_y2[i]<<std::endl;
            flag =false;
            break;
        }
    }
    if(flag)
        std::cout<<"Test passed"<<std::endl;
    delete[] h_y1;
    delete[] h_y2;    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to generate random input data
void generateData(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = 1.0f + static_cast<float>(rand()) / (RAND_MAX / 1.0f);
    }
}

int main() {
    int size = 10000;
    float* input = new float[size];
    float* output = new float[size];
    
    generateData(input, size);
    
    cudaSoftmax(input, output, size);
    
    delete[] input;
    delete[] output;
    
    return 0;
}