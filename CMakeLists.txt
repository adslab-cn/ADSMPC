cmake_minimum_required(VERSION 3.17)
project(MY-EzPC)

if(APPLE)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
# find_package(CUDAToolkit REQUIRED)

add_subdirectory(crypto/cryptoTools)
# add_subdirectory(crypto/bitpack)
add_subdirectory(crypto/FSS)
add_subdirectory(crypto/SCI)
# add_subdirectory(../SCI crypto/sci)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    neural_networks/src/datasets/mnist.cpp
    neural_networks/src/train.cpp
    neural_networks/src/random.cpp
    neural_networks/src/backend/cleartext.cpp
    # neural_networks/src/backend/baseline_cleartext.cpp
    # neural_networks/src/backend/float.cpp
    neural_networks/src/layers/softmax.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/neural_networks/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads FSS cryptoTools SCI-FloatML) # CUDA::cudart


# orca-sytorch example

add_executable(CNN 
    tests/CNN.cpp
)
target_link_libraries (CNN ${PROJECT_NAME})

add_executable(GCN 
    tests/GCN.cpp
)
target_link_libraries (GCN ${PROJECT_NAME})

add_executable(test
    tests/test_dataset.cpp
)
target_link_libraries (test ${PROJECT_NAME})
