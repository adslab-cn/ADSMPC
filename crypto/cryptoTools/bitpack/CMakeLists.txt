add_library(bitpack STATIC src/bitpack/bitpack.cpp)
target_include_directories(bitpack PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(bitpack PUBLIC adsm_build_options)
if(BUILD_TESTING)
    add_executable(bitpack_test tests/test.cpp)
    target_link_libraries(bitpack_test PRIVATE bitpack)
    add_test(NAME bitpack COMMAND bitpack_test)
endif()
