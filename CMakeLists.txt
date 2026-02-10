cmake_minimum_required(VERSION 3.10)
project(TBN_Test)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)

set(SOURCE_FILES 
    gemm/baseline.cpp 
    gemm/coded.cpp 
    gemm/util/encoder.cpp
)
add_executable(run_tests test.cpp ${SOURCE_FILES})

target_include_directories(run_tests PRIVATE 
    ${GTEST_INCLUDE_DIRS}
    gemm/util
    gemm
)

target_link_libraries(run_tests GTest::GTest GTest::Main pthread)