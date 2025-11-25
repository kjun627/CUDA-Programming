cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME cuda_programming)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

include(Dependency.cmake)

add_library(ds_timer_lib STATIC
    src/DS_timer.cpp)

target_include_directories(ds_timer_lib PUBLIC
    include)

function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -g
            --expt-relaxed-constexpr
        >
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${DEP_INCLUDE_DIR}
    )
    target_link_directories(${target_name} PRIVATE ${DEP_LIB_DIR})
    target_link_libraries(${target_name} PRIVATE 
        ${DEP_LIBS}
        ds_timer_lib
    )
    
    target_compile_definitions(${target_name} PRIVATE
        WINDOW_NAME="${WINDOW_NAME}"
        WINDOW_WIDTH=${WINDOW_WIDTH}
        WINDOW_HEIGHT=${WINDOW_HEIGHT}
    )
    
    add_dependencies(${target_name} ${DEP_LIST})
endfunction()

file(GLOB CH2_FILES "1_introduction/*.cu")
foreach(file ${CH2_FILES})
    get_filename_component(filename ${file} NAME_WE)
    add_cuda_executable(ch2_${filename} ${file})
endforeach()

file(GLOB CH3_FILES "2_CUDA_Workflow/*.cu")
foreach(file ${CH3_FILES})
    get_filename_component(filename ${file} NAME_WE)
    add_cuda_executable(ch3_${filename} ${file})
endforeach()

file(GLOB CH4_FILES "3_Thread_hierachy/*.cu")
foreach(file ${CH4_FILES})
    get_filename_component(filename ${file} NAME_WE)
    add_cuda_executable(ch4_${filename} ${file})
endforeach()