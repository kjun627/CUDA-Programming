cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME cuda_programming)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

file(GLOB CH2_FILES "ch_2/*.cu")
foreach(file ${CH2_FILES})
    get_filename_component(filename ${file} NAME_WE)
    add_executable(ch2_${filename} ${file})
    target_compile_options(ch2_${filename} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -g
            --expt-relaxed-constexpr
        >
    )
endforeach()

file(GLOB CH3_FILES "ch_3/*.cu")
foreach(file ${CH3_FILES})
    get_filename_component(filename ${file} NAME_WE)
    add_executable(ch3_${filename} ${file})
    target_compile_options(ch3_${filename} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -g
            --expt-relaxed-constexpr
        >
    )
endforeach()