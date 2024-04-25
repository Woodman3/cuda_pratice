#include<iostream>
#include<fstream>
#define OFFSET(row,col,R) ((row*R)+col)

using namespace std;

typedef double tt;

template<typename T>
__global__ void gemm(T *a,T *b,T *c,int m,int k,int n){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int bdx=blockDim.x;
    int bdy=blockDim.y;
    int x=bdx*bx+tx;
    int y=bdy*by+ty;
    if(x>=m||y>=n)
        return;
    int id=OFFSET(x,y,n);
    T count=0;
    for(int i=0;i<k;i++){
        count+=a[OFFSET(x,i,k)]*b[OFFSET(i,y,n)];
    }
    c[id]=count;

}

int main(){
    int m,k,n;
    ifstream fin;
    fin.open("./test");
    fin>>m>>k>>n;
    tt *h_a = new tt[m*k];
    tt *h_b = new tt[k*n];
    tt *h_c = new tt[m*n];
    tt *right = new tt[m*n];
    tt *d_a,*d_b,*d_c;
    for(int i=0;i<m*k;i++)
        fin>>h_a[i];
    for(int i=0;i<k*n;i++)
        fin>>h_b[i];
    for(int i=0;i<m*n;i++)
        fin>>right[i]; 
    cudaMalloc(&d_a,m*k*sizeof(tt));
    cudaMalloc(&d_b,k*n*sizeof(tt));
    cudaMalloc(&d_c,m*n*sizeof(tt));
    cudaMemcpy(d_a,h_a,m*k*sizeof(tt),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,k*n*sizeof(tt),cudaMemcpyHostToDevice);
    int blockx=16;
    int blocky=16;
    dim3 BlockRange(blockx,blocky);
    dim3 GridRange(m/blockx+1,n/blocky+1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm<tt><<<GridRange,BlockRange>>>(d_a,d_b,d_c,m,k,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << " ms" << endl;
    cudaMemcpy(h_c,d_c,m*n*sizeof(tt),cudaMemcpyDeviceToHost); 

    for(int i=0;i<m*n;i++){
        if(abs(right[i]-h_c[i])>0.01){
            cout<<"wrong"<<right[i]<<" "<<h_c[i]<<endl;
        }
    }
    cout<<"all is right!"<<endl;
    return 0;
}