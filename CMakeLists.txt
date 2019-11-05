cmake_minimum_required(VERSION 3.14)
project(faiss_test)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/heisenberg/Projects/Sub.Projects/faiss")
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/hdf5/include)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/hdf5/lib)

add_executable(detail_debug detail_debug.cpp)
add_executable(test_ivf test_ivf.cpp)

set(libs
	/home/heisenberg/Projects/Sub.Projects/faiss/libfaiss.a
        gomp
        cudart
        cublas
        pthread
        openblas
        hdf5)

target_link_libraries(detail_debug ${libs})
target_link_libraries(test_ivf ${libs})
