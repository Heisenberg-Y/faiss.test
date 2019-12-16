cmake_minimum_required(VERSION 3.14)
project(faiss_test)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/heisenberg/Projects/faiss")
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/hdf5/include)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/hdf5/lib)

add_executable(detail_debug detail_debug.cpp)
add_executable(test_ivf test_ivf.cpp)
add_executable(test_cpu test_cpu.cpp)
add_executable(test_column test_column.cpp)
add_executable(test_column_flat test_column_flat.cpp)
add_executable(faiss_benchmark faiss_benchmark.cpp)
add_executable(test_avx_512 test_avx_512.cpp)
add_executable(get_cpu_flags get_cpu_flags.cpp)

set(libs
        /home/heisenberg/Projects/faiss/libfaiss.a
        gomp
        cudart
        cublas
        pthread
        openblas
        hdf5)

target_link_libraries(detail_debug ${libs})
target_link_libraries(test_ivf ${libs})
target_link_libraries(test_cpu ${libs})
target_link_libraries(test_column ${libs})
target_link_libraries(test_column_flat ${libs})
target_link_libraries(faiss_benchmark ${libs})
target_link_libraries(test_avx_512 ${libs})
target_link_libraries(get_cpu_flags ${libs})