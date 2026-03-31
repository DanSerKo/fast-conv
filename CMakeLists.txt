cmake_minimum_required(VERSION 3.10)
project(TBN_Test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -msse4.2 -mavx2")

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

set(SOURCE_FILES 
    gemm/baseline.cpp 
    gemm/coded.cpp
    gemm/cach.cpp 
    gemm/simd.cpp 
    gemm/micro_kernel.cpp
    gemm/V14opt.cpp
    gemm/final.cpp
    gemm/best.cpp

    im2col/base.cpp
    im2col/naive.cpp
    im2col/p-im2col.cpp
    im2col/parallel.cpp

    util/encoder.cpp
)

set(INCLUDE_DIRS util gemm im2col)

add_executable(run_tests test.cpp ${SOURCE_FILES})
target_include_directories(run_tests PRIVATE ${INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
target_link_libraries(run_tests GTest::GTest GTest::Main Threads::Threads)

add_executable(run_benchmarks benchmark.cpp ${SOURCE_FILES})
target_include_directories(run_benchmarks PRIVATE ${INCLUDE_DIRS})
target_link_libraries(run_benchmarks benchmark::benchmark Threads::Threads)

add_executable(run_benchmarks_conv benchmarks_conv.cpp ${SOURCE_FILES})
target_include_directories(run_benchmarks_conv PRIVATE ${INCLUDE_DIRS})
target_link_libraries(run_benchmarks_conv benchmark::benchmark Threads::Threads)