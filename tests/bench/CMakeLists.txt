foreach(benchmark IN ITEMS
    comparison_primitive quaddpf_test octdpf_test academic_benchmark
    verdpf_test mp_dpf_verdpf_test shark_ifss dif GTDCF-ReLU)
    add_executable(${benchmark} "${benchmark}.cpp")
    target_link_libraries(${benchmark} PRIVATE sytorch)
endforeach()
# Benchmarks may be long-running; building them does not auto-run experiments.

# Fast local RSS correctness checks; no networking or global PRNG initialization.
add_executable(rss_group_element_test rss_group_element_test.cpp)
target_link_libraries(rss_group_element_test PRIVATE FSS)
add_test(NAME rss_group_element COMMAND rss_group_element_test)
