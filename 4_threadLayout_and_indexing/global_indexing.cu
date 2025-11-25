#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define NUM_DATA 1024 * 1024

__global__ void sumVector(int* _dDataPtr1, int* _dDataPtr2, int size){
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < size)
        _dDataPtr1[tID] += _dDataPtr2[tID];
}

__global__ void setData(int* _dDataPtr, int size){
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < size)
        _dDataPtr[tID] = tID;
}

int main(void){
    int* a, * b, * c;
    int* dDataPtr1, * dDataPtr2; 
    int memSize = sizeof(int)* NUM_DATA;
    DS_timer timer(5);
    timer.setTimerName(0, (char*) "CUDA Time");
    timer.setTimerName(1, (char*) "Computation (Kernel)");
    timer.setTimerName(2, (char*) "Data transport : Host -> Device");
    timer.setTimerName(3, (char*) "Data transport : Device -> Host");
    timer.setTimerName(4, (char*) "vector add cal on host");
    timer.initTimers();

    cudaError_t errorCode;
    
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    
    for (int i = 0; i < NUM_DATA; i++){
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++){
        c[i] = a[i] + b[i];
    }
    timer.offTimer(4);

    errorCode = cudaMalloc(&dDataPtr1, memSize);
    printf("cudaMalloc1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMalloc(&dDataPtr2, memSize);
    printf("cudaMalloc2 - %s\n", cudaGetErrorName(errorCode));
    
    errorCode = cudaMemset(dDataPtr1,0, memSize);
    printf("cudaMemset1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMemset(dDataPtr2,0, memSize);
    printf("cudaMemset2 - %s\n", cudaGetErrorName(errorCode));
    
    timer.onTimer(0);

    timer.onTimer(2);
    errorCode = cudaMemcpy(dDataPtr1, a, memSize, cudaMemcpyHostToDevice);
    printf("cudaMemcpy1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMemcpy(dDataPtr2, b, memSize, cudaMemcpyHostToDevice);
    printf("cudaMemcpy2 - %s\n", cudaGetErrorName(errorCode));
    timer.offTimer(2);
    
    timer.onTimer(1);
    dim3 dimGrid(ceil ((float) NUM_DATA /256) , 1,1);
    dim3 dimBlock(256,1,1);
    sumVector<<<dimGrid,dimBlock>>>(dDataPtr1, dDataPtr2, NUM_DATA);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    timer.onTimer(3);
    //cuda memcpy는 호스트 함수임!!
    errorCode = cudaMemcpy(a, dDataPtr1, memSize, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy3 - %s\n", cudaGetErrorName(errorCode));
    timer.offTimer(3);
    timer.offTimer(0);

    cudaFree(dDataPtr1); cudaFree(dDataPtr2);

    bool results = true;
    for (int i = 0; i < NUM_DATA; i++){
        if(c[i] != a[i]){
            results = false;
            printf("Calculating False - Data not matched\n");
            break;
        }
    }

    if(results){
        printf("Work done");
    }

    timer.printTimer(); 

    delete[] a; delete[] b; delete[] c;
}