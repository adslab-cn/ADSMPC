add_library(cryptoTools STATIC
    cryptoTools/Common/Defines.cpp
    cryptoTools/Common/Log.cpp
    cryptoTools/Crypto/AES.cpp
    cryptoTools/Crypto/PRNG.cpp)
target_include_directories(cryptoTools PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(cryptoTools PUBLIC adsm_build_options)
add_subdirectory(bitpack)
