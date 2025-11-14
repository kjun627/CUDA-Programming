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
    printf("\nHost -> Device: ");
    for (int i = 0; i < 10; i++) printf("%d", data[i]);

    printf("\nreceive Data from Host: ");
    printData<<<1,10>>>(dDataPtr);
    cudaDeviceSynchronize(); // 이거 없이 실행하면 cuda 커널 실행하고 결과를 기다리지 않고 바로 다음 cmd 실행함.
    // 동기화 해줘야한다. -> 근데 이게 맞나..?
    
    printf("\nChange Device Data: ");
    setdata<<<1,10>>>(dDataPtr);
    printData<<<1,10>>>(dDataPtr);

    cudaMemcpy(data,dDataPtr, sizeof(int)*10, cudaMemcpyDeviceToHost);
    printf("\nDevice -> Host: ");
    for (int i = 0; i < 10; i++) printf("%d", data[i]);
    cudaFree(dDataPtr);
}
