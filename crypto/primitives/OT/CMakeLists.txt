if(NOT ADSMPC_ENABLE_OT)
    message(STATUS "OT disabled (enable ADSMPC_ENABLE_OT only with SCI dependencies)")
    return()
endif()

# SCI-utils must be provided by an enclosing project; this repository does not
# bundle SCI and never downloads it implicitly.
if(NOT TARGET SCI-utils)
    message(FATAL_ERROR
        "ADSMPC_ENABLE_OT requires the external SCI-utils CMake target and "
        "SCI utils/emp-tool.h. Define SCI-utils in an enclosing project and set "
        "ADSMPC_SCI_INCLUDE_DIR to the compatible SCI include root. "
        "No SCI dependencies are bundled or downloaded.")
endif()
if(NOT EXISTS "${ADSMPC_SCI_INCLUDE_DIR}/utils/emp-tool.h")
    message(FATAL_ERROR
        "ADSMPC_ENABLE_OT requires ADSMPC_SCI_INCLUDE_DIR/utils/emp-tool.h; "
        "set ADSMPC_SCI_INCLUDE_DIR to the compatible SCI include root.")
endif()
add_library(SCI-OTPrimitive INTERFACE)
target_include_directories(SCI-OTPrimitive INTERFACE
    "$<BUILD_INTERFACE:${ADSMPC_SCI_INCLUDE_DIR}>")
target_link_libraries(SCI-OTPrimitive INTERFACE SCI-utils adsm_headers adsm_build_options)
