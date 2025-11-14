#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

__global__ void sumVector(int* _dDataPtr1, int* _dDataPtr2){
    _dDataPtr1[threadIdx.x] += _dDataPtr2[threadIdx.x];
}

__global__ void setData(int* _dDataPtr){
    _dDataPtr[threadIdx.x] = threadIdx.x;
}

int main(void){
    int* a, * b, * c;
    int* dDataPtr1, * dDataPtr2; 
    int memSize = sizeof(int)* NUM_DATA;
    cudaError_t errorCode;
    
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    
    for (int i = 0; i < NUM_DATA; i++){
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    for (int i = 0; i < NUM_DATA; i++){
        c[i] = a[i] + b[i];
    }
    
    errorCode = cudaMalloc(&dDataPtr1, memSize);
    printf("cudaMalloc1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMalloc(&dDataPtr2, memSize);
    printf("cudaMalloc2 - %s\n", cudaGetErrorName(errorCode));
    
    errorCode = cudaMemset(dDataPtr1,0, memSize);
    printf("cudaMemset1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMemset(dDataPtr2,0, memSize);
    printf("cudaMemset2 - %s\n", cudaGetErrorName(errorCode));
    
    errorCode = cudaMemcpy(dDataPtr1, a, memSize, cudaMemcpyHostToDevice);
    printf("cudaMemcpy1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMemcpy(dDataPtr2, b, memSize, cudaMemcpyHostToDevice);
    printf("cudaMemcpy2 - %s\n", cudaGetErrorName(errorCode));
    
    delete[] b;

    sumVector<<<1,NUM_DATA>>>(dDataPtr1, dDataPtr2);
    errorCode = cudaGetLastError();
    printf("kernel launch - %s", cudaGetErrorName(errorCode));

    //cuda memcpy는 호스트 함수임!!
    errorCode = cudaMemcpy(a, dDataPtr1, memSize, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy3 - %s\n", cudaGetErrorName(errorCode));

    cudaFree(dDataPtr1); cudaFree(dDataPtr2);

    bool results = true;
    for (int i = 0; i < NUM_DATA; i++){
        if(c[i] != a[i]){
            results = false;
            printf("Calculating False - Data not matched");
            break;
        }
    }

    if(results){
        printf("Work done");
    }
    delete[] a,delete[] c;
}