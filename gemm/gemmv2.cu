#include<iostream>
#include<fstream>
#include<float.h>
#include"../check.cuh"
#define OFFSET(row,col,R) (((row)*(R))+(col))

using namespace std;

typedef double tt;

const int BN = 128;
const int BM = 128;
const int TN = 8;
const int TM = 8;   
const int STEP = 8;

template<typename T>
__global__ void gemm(T *a,T *b,T *c,int n,int k,int m){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int bdx=blockDim.x;
    int bdy=blockDim.y;
    int ulrow = bx*BN; 
    int ulcol = by*BM;
    int tid=OFFSET(tx,ty,bdy);
    __shared__ T sa[BN][STEP];
    __shared__ T sb[STEP][BM];
    T tc[TN][TM]={0.0};
    int sacol_load = (tid&1) <<2; // %1  and *4
    int sarow_load = tid>>1;// /2
    int sbrow_load = tid>>5; // /32
    int sbcol_load =(tid & 31) <<2;// %32 and *4
    int len=(k+STEP-1)/STEP;
    int tnr = BN/TN; // threadnum of row
    int tnc = BM/TM; // threadnum of col
    for(int i=0;i<len;i++){
        for(int j=0;j<4;j++){
            sa[sarow_load][sacol_load+j]=a[OFFSET(ulrow+sarow_load,sacol_load+(i*STEP)+j,k)];
            sb[sbrow_load][sbcol_load+j]=b[OFFSET(sbrow_load+(i*STEP),ulcol+sbcol_load+j,m)];
        }
        __syncthreads();
        for(int row = 0;row<TN;row++){
            for(int col = 0;col<TM;col++){
                for(int j=0;j<STEP;j++){
                    tc[row][col]+=sa[tx*TN+row][j]*sb[j][ty*TM+col];
                }
            }
        }
        __syncthreads();
    }
    for(int row = 0;row<TN;row++){
        for(int col = 0;col<TM;col++){
            c[OFFSET(ulrow+tx*TN+row,ulcol+ty*TM+col,m)]=tc[row][col];
        }
    }
}

int main(){
    int m,k,n;
    ifstream fin;
    fin.open("./test");
    fin>>m>>k>>n;
    tt *h_a = new tt[n*k];
    tt *h_b = new tt[k*m];
    tt *h_c = new tt[n*m];
    tt *right = new tt[n*m];
    tt *d_a,*d_b,*d_c;
    for(int i=0;i<n*k;i++)
        fin>>h_a[i];
    for(int i=0;i<k*m;i++)
        fin>>h_b[i];
    for(int i=0;i<n*m;i++)
        fin>>right[i]; 
    cudaMalloc(&d_a,m*k*sizeof(tt));
    cudaMalloc(&d_b,k*n*sizeof(tt));
    cudaMalloc(&d_c,m*n*sizeof(tt));
    cudaMemset(d_c,0,m*n*sizeof(tt));
    cudaMemcpy(d_a,h_a,m*k*sizeof(tt),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,k*n*sizeof(tt),cudaMemcpyHostToDevice);
    dim3 BlockRange(BN/TN,BM/TM);
    dim3 GridRange(n/BN,m/BM);
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

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            int idx = OFFSET(i,j,m);
            if(abs(right[idx]-h_c[idx])>0.01){
                printf("wrong of %d %d %f %f\n",i,j,right[idx],h_c[idx]);
                return -1;
            }
        }
    }
    cout<<"all is right!"<<endl;
    return 0;
}