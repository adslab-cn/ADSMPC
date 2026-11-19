add_subdirectory(cryptoTools)

# All crypto implementation sources share one archive; directory CMake files
# own their source lists without creating circular static-library dependencies.
add_library(FSS STATIC)
target_link_libraries(FSS PUBLIC
    adsm_headers adsm_build_options Eigen3::Eigen cryptoTools bitpack)
add_subdirectory(commom)
add_subdirectory(primitives)
add_subdirectory(protocol)
