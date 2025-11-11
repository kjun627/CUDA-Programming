cmake_minimum_required(VERSION 3.18)
project(CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 4070 ti super -> (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(main ch_2/main.cu)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -g
        --expt-relaxed-constexpr
    >
)