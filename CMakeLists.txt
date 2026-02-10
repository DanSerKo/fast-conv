cmake_minimum_required(VERSION 3.10)
project(TBN_Test)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

set(SOURCE_FILES 
    gemm/baseline.cpp 
    gemm/coded.cpp 
    util/encoder.cpp
)

add_executable(run_tests test.cpp ${SOURCE_FILES})
target_include_directories(run_tests PRIVATE 
    ${GTEST_INCLUDE_DIRS}
    util
    gemm
)
target_link_libraries(run_tests GTest::GTest GTest::Main pthread)


add_executable(run_benchmarks benchmark.cpp ${SOURCE_FILES})
target_include_directories(run_benchmarks PRIVATE 
    util
    gemm
)
target_link_libraries(run_benchmarks benchmark::benchmark pthread)