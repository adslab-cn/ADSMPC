cmake_minimum_required(VERSION 3.17)
project(ADSMPC)
set(CMAKE_BUILD_TYPE Debug)
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
add_subdirectory(crypto/bitpack)
add_subdirectory(crypto/FSS)
add_subdirectory(crypto/SCI)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    nn/datasets/mnist.cpp
    # nn/train.cpp
    nn/random.cpp
    nn/backend/cleartext.cpp
    nn/layers/softmax.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nn/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads FSS cryptoTools SCI-FloatML) # CUDA::cudart

#################### orca-sytorch example #####################

add_executable(CNN 
    tests/inference/CNN.cpp
)
target_link_libraries (CNN ${PROJECT_NAME})

add_executable(GCN 
    tests/inference/GCN.cpp
)
target_link_libraries (GCN ${PROJECT_NAME})

# add_executable(test
#     tests/inference/test_dataset.cpp
# )
# target_link_libraries (test ${PROJECT_NAME})


#################### benchmark #####################

add_executable(
    dcf
    tests/bench/dcf.cpp
)
target_link_libraries(dcf ${PROJECT_NAME})


add_executable(
    dpf
    tests/bench/dpf.cpp
)
target_link_libraries(dpf ${PROJECT_NAME})

# add_executable(
#     dmpftest
#     tests/bench/bs-dmpf.cpp
# )
# target_link_libraries(dmpftest ${PROJECT_NAME})

# add_executable(
#     sloth_relu
#     tests/bench/sloth_relu.cpp
# )

# target_link_libraries(sloth_relu ${PROJECT_NAME})

# add_executable(
#     relu
#     tests/bench/relu.cpp
# )

# target_link_libraries(relu ${PROJECT_NAME})

# add_executable(
#     softmax
#     tests/bench/softmax.cpp
# )

# target_link_libraries(softmax ${PROJECT_NAME})


add_executable(test_route
    tests/bench/routing.cpp
)
target_link_libraries (test_route ${PROJECT_NAME})

add_executable(test_graph_update
    tests/bench/graph_update_test.cpp
)
target_link_libraries (test_graph_update ${PROJECT_NAME})
