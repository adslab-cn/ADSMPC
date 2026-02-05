cmake_minimum_required(VERSION 3.17)
project(sytorch)

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

add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/bitpack)
add_subdirectory(ext/FSS)
add_subdirectory(ext/sci)
# add_subdirectory(../SCI ext/sci)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    src/sytorch/random.cpp
    src/sytorch/backend/cleartext.cpp
    src/sytorch/backend/baseline_cleartext.cpp
    src/sytorch/backend/float.cpp
    src/sytorch/softmax.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads FSS cryptoTools SCI-FloatML) # CUDA::cudart


# === Benchmarks ===
add_executable(
    dpf
    tests/bench/dpf.cpp
)
target_link_libraries(dpf ${PROJECT_NAME})


# add_executable(
#     sloth_drelu
#     tests/sloth_drelu.cpp
# )

# target_link_libraries(sloth_drelu ${PROJECT_NAME})

add_executable(
    mp_drelu
    tests/bench/mp_drelu.cpp
)

target_link_libraries(mp_drelu ${PROJECT_NAME})


add_executable(
    mp_relu
    tests/bench/mp_relu.cpp
)

target_link_libraries(mp_relu ${PROJECT_NAME})

# add_executable(
#     fastsecnet_relu
#     tests/bench/fastsecnet_relu.cpp
# )

# target_link_libraries(fastsecnet_relu ${PROJECT_NAME})
add_executable(
    mp_oblivGCNconv
    tests/bench/mp_oblivGCNconv.cpp
)

target_link_libraries(mp_oblivGCNconv ${PROJECT_NAME})

add_executable(
    mp_gcnconv
    tests/bench/mp_gcnconv.cpp
)

target_link_libraries(mp_gcnconv ${PROJECT_NAME})

add_executable(
    mp_matmul
    tests/bench/mp_matmul.cpp
)

target_link_libraries(mp_matmul ${PROJECT_NAME})