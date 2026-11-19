add_library(sytorch STATIC)
target_link_libraries(sytorch PUBLIC FSS Eigen3::Eigen adsm_headers adsm_build_options)
add_subdirectory(src)
