## Thread Layout 설계

스레드 레이아웃은 그리드아 블록에 따라 결정되는데, 그리드와 블록의 크기는 데이터의 크기, 커널 성능 특성, GPU 자원 제한도 고려해야한다고 한다.

## Thread Indexing
블록이 가질 수 있는 최대 스레드 수는 1024개인데, 만약 백터의 길이가 이것보다 길면 블록이 여러개 필요하다.
이렇게 되면 각 블록마다 처리해야하는 벡터의 원소를 알맞게 매핑해줘야한다. 이러한 부분을 보고 이전에 작성한 커널을
다시 살펴보자.

```c
__global__ void sumVector(int* _dDataPtr1, int* _dDataPtr2){
        _dDataPtr1[threadIdx.x] += _dDataPtr2[threadIdx.x];
    }
```
위 코드는 이전에 작성한 벡터 합 구하는 코드의 CUDA kernel 인데 thread index를 기반으로 작업하는 것을 볼 수 있다.
이렇게 됐을 때 문제가 뭐냐면 만약 1024개의 thread 개수가 넘어가게 되면, 글로벌 인덱스 기반으로 커널을 작동한게 아니어서
의도하지 않은 결과물이 나온다. 예를 들면 1025 차원의 벡터일 경우, 1025번째 원소는 블록 2번의 쓰레드일텐데, 그 쓰레드는 1번 블록의 1번 쓰레드와 같은 메모리를
접근하게 된다.

따라서 다음과 같은 일반화된 인덱싱이 필요하다.

*N 번째 블록의 M 번째 쓰레드가 접근해야할 벡터의 원소*  
**vector[blockIdx.x * blockDim.x + threadIdx.x]**

내가 몇번째 블록에 있고 (blockIdx.x), 블록당 할당 된 쓰레드의 개수를 고려할 때(blockDim.x) 할당되는 쓰레드 인덱스는 위와 같은 것이다.

## global indexing kernel example
```c
__global__ void sumVector(int* _dDataPtr1, int* _dDataPtr2){
        int tID = blockIdx.x * blockDim.x + threadIdx.x;
        _dDataPtr1[tID] += _dDataPtr2[tID];
    }
```

글로벌 인덱싱을 기반으로 커널을 수정하면 위 코드와 같이 수정할 수 있다. 근데 추가적으로 발생하는 문제가 있는데,
1025 개의 쓰레드가 필요할 경우 1025번째 쓰레드는 두번째 블록에 할당될 것이다. 하지만 쿠다의 실행단위가 블록임(정확히는 wrap 이지만)으로
사용하지 않는 다른 블록들 1023개도 해당 커널의 동작을 수행하게 되는데, 이것에 대한 예외 처리가 필요하다.

```c
__global__ void vecAdd(int* _dDataPtr1, int* _dDataPtr2, int _size){
    int tID = blockIdx. * blockDim.x + threadIdx.x;
    if (tID < _size) {
        _dDataPtr1[tID] += _dDataPtr2[tID];
    }
}
```

## indexing 연습

### 블록 내에서의 스레드 global Index  

하나의 블록안에서의 global index  
1D 일 때 = threadIdx.x  
2D 일 때 = blockDim.x * threadIdx.y + threadIdx.x  
3D 일 때 = blockDim.x * blockDim.y * threadIdx.z + (2차원 블록 안에서의 tid)

### global thread Index in Grid
NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z (블록 내 스레드 총 개수)   
1D Grid = blockIdx.x * (NUM_THREAD_IN_BLOCK) + TID_IN_BLOCK
2D Grid = gridDim.x * NUM_THREAD_IN_BLOCK * blockIdx.y + 1D Graid
3D Grid = blockIdx.x * gridDim.y * gridIm.x * NUM_THREAD_IN_BLOCk + 2D_GRID_TID

위 인덱스 들은 매번 정의하기 힘드니 헤더파일로 사용하는 것을 추천함.

### 2차원 데이터에 대한 indexing
이전까지의 방법은 1차원 데이터에 대한 인덱싱이었다. 가장 간단한 방ㅂ버은 2차원 스레드를 사용해 각 스레드가
행렬의 원소를 할당 받게 하면 된다.

```c
col = threadIdx.x
row = threadIdx.y

index (row, col) = row * (length(row)) + col
index (row, col) = row * blockDim.x + col
                 = threadIdx.y * blockDim.x + threadIdx.x
```

### Large Scale Matrix calculation
이전 설명에서는 Thread의 global index를 1D, 2D 에서만 설명헀다. 다만 이전의 설명 방법과 별개로 
Thread layout,indexing은 데이터, 알고리즘에 따라 설계 방법이 달라질 수 있다.

근데 만약 행렬의 크기가 1024x1024보다 크면 어떻게 해야할까?

(2D Grid, 2D block), (1D Grid, 1D Block), (2D Grid, 1D Block)

이렇게 나누어서 한번 해결해보려한다.