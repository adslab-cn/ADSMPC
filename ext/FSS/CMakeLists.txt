cmake_minimum_required(VERSION 3.16)
project(FSS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME}
    src/FSS/config.cpp
    src/FSS/comms.cpp
    src/FSS/input_prng.cpp
    src/FSS/prng.cpp
    src/FSS/stats.cpp
    src/FSS/utils.cpp
    and.cpp
    api.cpp
    conv.cpp
    dcf.cpp
    mult.cpp
    pubdiv.cpp
    relu.cpp
    signextend.cpp
    pubcmp.cpp
    clip.cpp
    dpf.cpp
    lut.cpp
    select.cpp
    fixtobfloat16.cpp
    wrap.cpp
    float.cpp
    taylor.cpp
    mic.cpp
    msnzb.cpp
    truncate.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads cryptoTools bitpack)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)