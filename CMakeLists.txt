cmake_minimum_required(VERSION 3.17)
project(ADSMPC_C LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Existing code uses GNU extensions such as variable-length arrays.
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

option(ADSMPC_ENABLE_OT "Build OT with externally supplied SCI dependencies" OFF)
option(ADSMPC_ENABLE_SECFLOAT "Build softmax with externally supplied SecFloat dependencies" OFF)
set(ADSMPC_SCI_INCLUDE_DIR "" CACHE PATH "SCI include root containing utils/emp-tool.h")
set(ADSMPC_SECFLOAT_INCLUDE_DIR "" CACHE PATH "SecFloat include root containing library_float.h")

add_library(adsm_build_options INTERFACE)
target_compile_features(adsm_build_options INTERFACE cxx_std_17)
target_link_libraries(adsm_build_options INTERFACE OpenMP::OpenMP_CXX Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(adsm_build_options INTERFACE
        -maes -march=native
        -Wno-write-strings -Wno-unused-result -Wno-ignored-attributes
        -Wno-deprecated-declarations)
endif()

# Project-owned includes use <crypto/...> or <network/include/...>.
# Do not add every leaf directory: names such as assert.h and utils.h collide.
add_library(adsm_headers INTERFACE)
target_include_directories(adsm_headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

add_subdirectory(crypto)
add_subdirectory(network)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
