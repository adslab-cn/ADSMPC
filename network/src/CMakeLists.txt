target_sources(sytorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/random.cpp")
add_subdirectory(backend)

if(ADSMPC_ENABLE_SECFLOAT)
    if(NOT EXISTS "${ADSMPC_SECFLOAT_INCLUDE_DIR}/library_float.h")
        message(FATAL_ERROR
            "ADSMPC_ENABLE_SECFLOAT requires external library_float.h. "
            "Set ADSMPC_SECFLOAT_INCLUDE_DIR to a compatible SecFloat include root; "
            "the SecFloat dependency is not bundled or downloaded.")
    endif()
    if(NOT TARGET SCI-FloatML)
        message(FATAL_ERROR
            "ADSMPC_ENABLE_SECFLOAT requires the external SCI-FloatML CMake target "
            "provided by an enclosing project.")
    endif()
    target_sources(sytorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp")
    target_include_directories(sytorch PRIVATE "${ADSMPC_SECFLOAT_INCLUDE_DIR}")
    target_link_libraries(sytorch PUBLIC SCI-FloatML)
else()
    message(STATUS "SecFloat softmax disabled (external library_float.h / SCI-FloatML required)")
endif()
