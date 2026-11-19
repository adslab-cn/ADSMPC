foreach(example IN ITEMS BPGNN CNN)
    add_executable(${example} "${example}.cpp")
    target_link_libraries(${example} PRIVATE sytorch)
endforeach()
# These require explicit party startup; do not register unattended training runs.

# Self-contained three-process TCP integration and failure tests.
add_executable(rss_comms_test rss_comms_test.cpp)
target_link_libraries(rss_comms_test PRIVATE FSS)
add_test(NAME rss_comms COMMAND rss_comms_test)
set_tests_properties(rss_comms PROPERTIES TIMEOUT 180)
