#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printData(int* _dDataPtr){
    printf("%d", _dDataPtr[threadIdx.x]);
}

__global__ void setdata(int* _dDataPtr){
    _dDataPtr[threadIdx.x] += 1;
}

int main(){
    int data[10] = {0};
    for (int i = 0; i < 10; i++) data[i] = 1;

    int* dDataPtr;
    cudaMalloc(&dDataPtr, sizeof(int)*10);
    cudaMemset(dDataPtr, 0, sizeof(int)*10);

    printf("Device Data : ");
    printData<<<1,10>>>(dDataPtr);

    cudaMemcpy(dDataPtr, data, sizeof(int)*10 , cudaMemcpyHostToDevice);
    printf("\nDevice -> Host: ");
    for (int i = 0; i < 10; i++) printf("%d", data[i]);

    cudaFree(dDataPtr);
}
