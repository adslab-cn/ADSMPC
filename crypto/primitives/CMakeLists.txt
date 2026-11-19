# ASS and RSS are header-only and use the parent FSS target's dependencies.
target_sources(FSS PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ASS/group_element.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/RSS/group_element.h")
add_subdirectory(FSS)
add_subdirectory(OT)
