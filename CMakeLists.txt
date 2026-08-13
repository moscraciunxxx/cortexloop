cmake_minimum_required(VERSION 3.16)
project(cortexloop C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CORTEXLOOP_WITH_KLEIDIAI "Fetch and link Arm KleidiAI INT4 micro-kernels" ON)

add_library(cortexloop SHARED
    native/src/util.c
    native/src/scalar.c
    native/src/image.c
    native/src/gemm.c
    native/src/cnn.c
    native/src/api.c
    native/src/kleidiai_backend.cpp
)

target_include_directories(cortexloop
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/native/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/native/src
)

set_source_files_properties(native/src/scalar.c PROPERTIES
    COMPILE_FLAGS "-O2 -fno-vectorize -fno-slp-vectorize"
)

if(APPLE)
    target_compile_options(cortexloop PRIVATE -O3 -mcpu=native)
else()
    target_compile_options(cortexloop PRIVATE -O3 -march=armv8.6-a+dotprod+i8mm)
endif()

if(NOT APPLE)
    target_link_libraries(cortexloop PUBLIC m)
endif()

if(CORTEXLOOP_WITH_KLEIDIAI)
    include(FetchContent)
    set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        kleidiai
        GIT_REPOSITORY https://github.com/ARM-software/kleidiai.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(kleidiai)
    target_link_libraries(cortexloop PRIVATE kleidiai)
    target_compile_definitions(cortexloop PRIVATE CORTEXLOOP_WITH_KLEIDIAI=1)
    if(APPLE)
        target_compile_options(cortexloop PRIVATE -march=armv8.6-a+dotprod+i8mm)
    endif()
endif()

add_executable(cortexloop-bench native/src/cli.c)
target_link_libraries(cortexloop-bench PRIVATE cortexloop)
target_include_directories(cortexloop-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native/include)

install(TARGETS cortexloop cortexloop-bench)
