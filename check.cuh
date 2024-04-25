#include<iostream>


#define CHECK(call)  \
do{ \
   const cudaError_t err = call; \
   if(err!=cudaSuccess){ \
        cout<<err<<endl; \
        cout<<cudaGetErrorString(err)<<endl; \
   }  \
}while(0)
