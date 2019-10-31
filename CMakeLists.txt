cmake_minimum_required(VERSION 3.14)
project(faiss_test)

set(CMAKE_CXX_STANDARD 14)

include_directories("/home/heisenberg/Projects/faiss")
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/hdf5/include)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/hdf5/lib)

add_executable(1-Flat 1-Flat.cpp)
add_executable(2-IVFFlat 2-IVFFlat.cpp)
add_executable(3-IVFPQ 3-IVFPQ.cpp)
add_executable(4-GPU 4-GPU.cpp)
add_executable(5-GPU 5-GPU.cpp)
add_executable(7-GPU 7-GPU.cpp)
add_executable(X-IVFSQ X-IVFSQ.cpp)
add_executable(my_test my_test.cpp)

set(libs
        /home/heisenberg/Projects/faiss/libfaiss.a
        gomp
        cudart
        cublas
        pthread
        openblas
        hdf5)

target_link_libraries(my_test ${libs})
target_link_libraries(1-Flat ${libs})
target_link_libraries(2-IVFFlat ${libs})
target_link_libraries(3-IVFPQ ${libs})
target_link_libraries(4-GPU ${libs})
target_link_libraries(5-GPU ${libs})
target_link_libraries(7-GPU ${libs})
target_link_libraries(X-IVFSQ ${libs})