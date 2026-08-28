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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
# find_package(CUDAToolkit REQUIRED)

add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/bitpack)
add_subdirectory(ext/FSS)
# add_subdirectory(ext/sci)
# add_subdirectory(../SCI ext/sci)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    src/random.cpp
    src/backend/cleartext.cpp
    src/backend/baseline_cleartext.cpp
    src/backend/float.cpp
    # src/softmax.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads FSS cryptoTools) # SCI-FloatML CUDA::cudart


# === Benchmarks ===
add_executable(
    comparison_primitive
    tests/bench/comparison_primitive.cpp
)
target_link_libraries(comparison_primitive ${PROJECT_NAME})

add_executable(dual_fss_malicious_comparison tests/bench/dual_fss_malicious_comparison.cpp)
target_link_libraries(dual_fss_malicious_comparison ${PROJECT_NAME})


add_executable(
    GTDCF-ReLU
    tests/bench/GTDCF-ReLU.cpp
)
target_link_libraries(GTDCF-ReLU ${PROJECT_NAME})

add_executable(
    Grotto-ReLU
    tests/bench/Grotto-ReLU.cpp
)
target_link_libraries(Grotto-ReLU ${PROJECT_NAME})

add_executable(
    graphiti_test
    tests/bench/graphiti_test.cpp
)
target_link_libraries(graphiti_test ${PROJECT_NAME})

add_executable(softmax_comparison tests/bench/softmax_comparison.cpp)
target_link_libraries(softmax_comparison ${PROJECT_NAME})

add_executable(relu_comparison tests/bench/relu_comparison.cpp)
target_link_libraries(relu_comparison ${PROJECT_NAME})


# === E2E ===
add_executable(
    BPGNN
    tests/e2e/BPGNN.cpp
)
target_link_libraries(BPGNN ${PROJECT_NAME})

add_executable(BPGNN_Cora tests/e2e/BPGNN_Cora.cpp)
target_link_libraries(BPGNN_Cora ${PROJECT_NAME})

add_executable(BPGNN_Citeseer tests/e2e/BPGNN_Citeseer.cpp)
target_link_libraries(BPGNN_Citeseer ${PROJECT_NAME})

add_executable(BPGNN_Pubmed tests/e2e/BPGNN_Pubmed.cpp)
target_link_libraries(BPGNN_Pubmed ${PROJECT_NAME})

add_executable(CryptGNN_Cora tests/e2e/CryptGNN_Cora.cpp)
target_link_libraries(CryptGNN_Cora ${PROJECT_NAME})

add_executable(CryptGNN_Citeseer tests/e2e/CryptGNN_Citeseer.cpp)
target_link_libraries(CryptGNN_Citeseer ${PROJECT_NAME})

add_executable(CryptGNN_Pubmed tests/e2e/CryptGNN_Pubmed.cpp)
target_link_libraries(CryptGNN_Pubmed ${PROJECT_NAME})

add_executable(
    OblivGNN
    tests/e2e/OblivGNN.cpp
)
target_link_libraries(OblivGNN ${PROJECT_NAME})

# add_executable(
#     test_gtdpf_comm
#     tests/bench/test_gtdpf_comm.cpp
# )
# target_link_libraries(test_gtdpf_comm ${PROJECT_NAME})

# add_executable(
#     dcf
#     tests/bench/dcf.cpp
# )
# target_link_libraries(dcf ${PROJECT_NAME})

# add_executable(
#     mp_gtdpf_arith_test
#     tests/bench/mp_gtdpf_arith_test.cpp
# )
# target_link_libraries(mp_gtdpf_arith_test ${PROJECT_NAME})


# add_executable(
#     test_gtdpf_payload
#     tests/bench/test_gtdpf_payload.cpp
# )
# target_link_libraries(test_gtdpf_payload ${PROJECT_NAME})

# add_executable(
#     GTDPF_payload
#     tests/bench/GTDPF_payload.cpp
# )
# target_link_libraries(GTDPF_payload ${PROJECT_NAME})

# add_executable(
#     sloth_drelu
#     tests/sloth_drelu.cpp
# )

# target_link_libraries(sloth_drelu ${PROJECT_NAME})

# add_executable(
#     mp_drelu
#     tests/bench/mp_drelu.cpp
# )

# target_link_libraries(mp_drelu ${PROJECT_NAME})


# add_executable(
#     mp_relu
#     tests/bench/mp_relu.cpp
# )

# target_link_libraries(mp_relu ${PROJECT_NAME})

# add_executable(
#     mp_fastsecnet_relu
#     tests/bench/mp_fastsecnet_relu.cpp
# )
# target_link_libraries(mp_fastsecnet_relu ${PROJECT_NAME})


# add_executable(
#     mp_gt_relu
#     tests/bench/mp_gt_relu.cpp
# )
# target_link_libraries(mp_gt_relu ${PROJECT_NAME})

# add_executable(
#     mp_softmax
#     tests/bench/mp_softmax.cpp
# )
# target_link_libraries(mp_softmax ${PROJECT_NAME})

# add_executable(
#     test_softmax_opt
#     tests/bench/test_softmax_opt.cpp
# )
# target_link_libraries(test_softmax_opt ${PROJECT_NAME})

# add_executable(
#     test_softmax_logic
#     tests/bench/test_softmax_logic.cpp
# )
# target_link_libraries(test_softmax_logic ${PROJECT_NAME})


# add_executable(
#     mp_oblivGCNconv
#     tests/bench/mp_oblivGCNconv.cpp
# )

# target_link_libraries(mp_oblivGCNconv ${PROJECT_NAME})

# add_executable(
#     mp_gcnconv
#     tests/bench/mp_gcnconv.cpp
# )

# target_link_libraries(mp_gcnconv ${PROJECT_NAME})

# add_executable(
#     mp_matmul
#     tests/bench/mp_matmul.cpp
# )

# target_link_libraries(mp_matmul ${PROJECT_NAME})


# === E2E ===
add_executable(
    CNN
    tests/e2e/CNN.cpp
)
target_link_libraries(CNN ${PROJECT_NAME})
