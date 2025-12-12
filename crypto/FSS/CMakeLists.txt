cmake_minimum_required(VERSION 3.16)
project(FSS)

if(APPLE) # APPLE系统会添加下面操作
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

find_package(OpenSSL REQUIRED)

add_library(${PROJECT_NAME}
    aux_parameter/config.cpp
    aux_parameter/comms.cpp
    aux_parameter/input_prng.cpp
    aux_parameter/prng.cpp
    aux_parameter/stats.cpp
    aux_parameter/utils.cpp

    api/api.cpp

    protocol/and.cpp
    protocol/conv.cpp
    primitives/dcf.cpp
    primitives/dpf.cpp
    primitives/dmpf.cpp

    # src/primitives/2pc_dcf.cpp
    # src/primitives/2pc_idpf.cpp
    
    protocol/float.cpp
    protocol/mic.cpp
    # protocol/msnzb.cpp
    protocol/mult.cpp
    protocol/pubdiv.cpp
    # protocol/pubcmp.cpp
    # protocol/clip.cpp
    # src/protocol/relutruncate.cpp
    protocol/relu.cpp
    protocol/lut.cpp
    # src/protocol/reluextend.cpp
    protocol/signextend.cpp
    protocol/select.cpp
    # protocol/fixtobfloat16.cpp
    # protocol/wrap.cpp
    # protocol/taylor.cpp
    # protocol/truncate.cpp
    protocol/dpfsort.cpp
    protocol/graphupdate.cpp
    protocol/relufastsecnet.cpp
    protocol/ars.cpp
    protocol/wrap.cpp
    
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads cryptoTools bitpack)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
