#include<iostream>


#define CHECK(...)  \
do{ \
   const cudaError_t err = __VA_ARGS__;\
   if(err!=cudaSuccess){ \
        cout<<err<<endl; \
        cout<<cudaGetErrorString(err)<<endl; \
   }  \
}while(0)
