cmake_minimum_required(VERSION 3.16)
project(FSS)

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

add_library(${PROJECT_NAME}
    src/aux_parameter/config.cpp
    src/aux_parameter/comms.cpp
    src/aux_parameter/input_prng.cpp
    src/aux_parameter/prng.cpp
    src/aux_parameter/stats.cpp
    src/aux_parameter/utils.cpp
    src/aux_parameter/bitpack.cpp
    src/protocol/and.cpp
    src/api/api.cpp
    src/protocol/conv.cpp
    src/primitives/dcf.cpp
    src/primitives/dpf.cpp
    src/protocol/float.cpp
    src/protocol/mic.cpp
    src/protocol/msnzb.cpp
    src/protocol/mult.cpp
    src/protocol/pubdiv.cpp
    src/protocol/pubcmp.cpp
    src/protocol/clip.cpp
    src/protocol/lut.cpp
    # src/protocol/relutruncate.cpp
    src/protocol/relu.cpp
    # src/protocol/reluextend.cpp
    src/protocol/fixtobfloat16.cpp
    src/protocol/wrap.cpp
    src/protocol/signextend.cpp
    src/protocol/select.cpp
    src/protocol/taylor.cpp
    src/protocol/truncate.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads cryptoTools)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
