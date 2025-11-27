#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW 1048
#define COL 1048
#define NUM_DATA COL * ROW  // 4,194,304

__global__ void matrixAddition (int* matrix1, int* matrix2, int* matrix3){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < ROW && col < COL){
        int idx = row * COL + col;
        matrix3[idx] = matrix1[idx] + matrix2[idx];
    }
}
int main(){
    int* a, *b, *c, *d;
    int* dDataPtr1, * dDataPtr2, * dDataPtr3;
    int memSize = sizeof(int) * NUM_DATA;
    
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
    d = new int[NUM_DATA]; memset(d, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++){
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++){
        c[i] = a[i] + b[i];
    }
    timer.offTimer(4);

    // 2D Grid, 2D Block 로 해결하기
    dim3 blockSize (16,16,1); // 블록당 256 개 쓰레드 할당
    dim3 gridSize ((ROW + blockSize.x -1 ) / blockSize.x,
                    (COL + blockSize.y -1 ) / blockSize.y);


    errorCode = cudaMalloc(&dDataPtr1, memSize);
    printf("cudaMalloc1 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMalloc(&dDataPtr2, memSize);
    printf("cudaMalloc2 - %s\n", cudaGetErrorName(errorCode));
    errorCode = cudaMalloc(&dDataPtr3, memSize);
    printf("cudaMalloc3 - %s\n", cudaGetErrorName(errorCode));

    timer.onTimer(0); // cuda time
    cudaMemset(dDataPtr3, 0, memSize);
    timer.onTimer(2); // data transport
    cudaMemcpy(dDataPtr1, a, memSize, cudaMemcpyHostToDevice);  
    cudaMemcpy(dDataPtr2, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);
 
    
    timer.onTimer(1);
    matrixAddition<<<gridSize, blockSize>>> (dDataPtr1,dDataPtr2, dDataPtr3);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    timer.onTimer(3); // Device to host data transport
    cudaMemcpy(d, dDataPtr3, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    timer.offTimer(0);

    cudaFree(dDataPtr1); cudaFree(dDataPtr2); cudaFree(dDataPtr3);

    bool results = true;
    for (int i = 0; i < NUM_DATA; i++){
        if(c[i] != d[i]){
            results = false;
            printf("Calculating False - Data not matched\n");
            break;
        }
    }

    if(results){
        printf("CUDA Success");
    }

    timer.printTimer();

    delete[] a; delete[] b; delete[] c; delete[] d;

    return 0;
}
