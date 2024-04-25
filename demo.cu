#include"check.cuh"
#include<iostream>
#include<stdio.h>
#define N 10

using namespace std;
template<typename T>
__global__ void fun(T* A,T* B,T* C){
    int i = threadIdx.x;
    C[i]=A[i]+B[i];
    printf("wtf%d\n",i);

}

int main(){
    int *h_A,*h_B,*h_C;
    h_A=(int*)malloc(N*sizeof(int));
    h_B=(int*)malloc(N*sizeof(int));
    h_C=(int*)malloc(N*sizeof(int));
    int *d_A,*d_B,*d_C;
    for(int i=0;i<N;i++){
        h_A[i]=i;
        h_B[i]=i+4;
        h_C[i]=3;
    } 
    CHECK(cudaMalloc((void**)&d_A,N*sizeof(int)));
    CHECK(cudaMalloc((void**)&d_B,N*sizeof(int)));
    CHECK(cudaMalloc((void**)&d_C,N*sizeof(int)));
    CHECK(cudaMemcpy(d_A,h_A,N*sizeof(int),::cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,N*sizeof(int),::cudaMemcpyHostToDevice));
    fun<int><<<1,N>>>(d_A,d_B,d_C);
    CHECK(cudaMemcpy(h_C,d_C,N*sizeof(int),::cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    cout<<h_C[0]<<endl;
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C); 
}