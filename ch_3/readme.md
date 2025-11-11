# Chaper 3

## 3.1 CUDA Program sturcture and processing
1. 호스트 메모리 -> 디바이스 메모리 데이터 복사 (GPU에서 처리할 데이터)
2. GPU 연산 (커널 호출 및 데이터 처리)
3. 디바이스 메모리 -> 호스트 메모리 데이터 복사 (연산 결과)

## 3.2 CUDA Basic memory API

### 디바이스 메모리 할당 - cudaMalloc()
- ptr void** 형으로 디바이스 메모리 공간의 시작 주소를 담을 포인터 변수의 주소 (포인터의 포인터)
- size 는 할당할 공간의 크기 (Byte)

```C
#include "cuda_runtime.h"
#include "device_launch_paramters.h"

int main(void){
    int *dDataPtr;
    cudaMalloc(&dDataPrt, szieof(int)*32);
}
```
위와 같은 방식으로 디바이스 메모리를 할당. 할당된 메모리 공간의 시작 주소가 dDataPtr 에 저장됨.-> dDataPtr로 디바이스 메모리 접근 가능. But 호스트 코드에서 dData Ptr 로 디바이스 메모리에 직접 접근 X

**Notice:** CUDA Program 에서는 호스트 메모리 영역과 디바이스 영역의 구분을 위해서 , 디바이스 메모리 영역 변수는 이름 앞에 d 를 붙인다고 함. 

### 디바이스 메모리 해제 - cudaFree()
C 언어에서의 Free() 와 동일한 개념.
```c
cudaFree(dDataPrt);
```
### 디바이스 메모리 초기화 - cudaMemset()
cudaMalloc() 으로 메모리 할당  시 garbage value 존재.
c 에서 memset() 함수와 동일한 역할.
```c
cudaError_t cudaMemset (void* ptr, int value, size_t size)

```
### 에러 코드 확인 - cudaGetErrorname()
CUDA API의 return 값이 대부분 enum 임.(에러 코드 수 많음) cudaSuccess가 아닌 경우에 cudaGetErrorName()으로 에러 종류 확인하는 것이 좋다고 함.
```c
__host__ __device__ const char* cudaGetErrorName (cudaError_t error)
```
**호스트 디바이스 코드에서 모두 사용 가능**
