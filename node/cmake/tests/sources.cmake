include("${CMAKE_CURRENT_LIST_DIR}/sources/accel.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/compute.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/core.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/runtime.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/task.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/host.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/net.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/reactor.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sources/replay.cmake")

# Source lists are the admission authority. The glob validates closure during
# configuration. The development proof normalizes source-topology changes, so
# additions, removals, rename/copy endpoints, type changes, conflicts, and
# ignored paths force this check without paying for a CMake interpreter and
# recursive glob on every no-op Ninja invocation. Ninja owns contents.
get_cmake_property(rund_node_test_source_variables VARIABLES)
set(rund_node_registered_test_sources
  tests/contract/dispatch.cpp
  tests/contract/main.cpp)
foreach(variable IN LISTS rund_node_test_source_variables)
  if(NOT variable MATCHES "^NODE_TEST_[A-Z0-9_]+_SOURCES$")
    continue()
  endif()
  foreach(source IN LISTS ${variable})
    if(IS_ABSOLUTE "${source}")
      get_filename_component(path "${source}" ABSOLUTE)
      file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${path}")
    else()
      set(relative "${source}")
      get_filename_component(
        path "${CMAKE_CURRENT_SOURCE_DIR}/${source}" ABSOLUTE)
    endif()
    if(NOT relative MATCHES "^tests/contract/.*\\.(cpp|mm)$" OR
       NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
      message(FATAL_ERROR
        "Invalid Node contract source in ${variable}: ${source}")
    endif()
    list(APPEND rund_node_registered_test_sources "${relative}")
  endforeach()
endforeach()
list(REMOVE_DUPLICATES rund_node_registered_test_sources)
list(SORT rund_node_registered_test_sources)

file(GLOB_RECURSE rund_node_live_test_sources
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/contract/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/contract/*.mm")
list(SORT rund_node_live_test_sources)
if(NOT "${rund_node_registered_test_sources}" STREQUAL
       "${rund_node_live_test_sources}")
  set(missing ${rund_node_live_test_sources})
  set(extra ${rund_node_registered_test_sources})
  foreach(source IN LISTS rund_node_registered_test_sources)
    list(REMOVE_ITEM missing "${source}")
  endforeach()
  foreach(source IN LISTS rund_node_live_test_sources)
    list(REMOVE_ITEM extra "${source}")
  endforeach()
  message(FATAL_ERROR
    "Node contract source registry is not closed; missing=${missing}; extra=${extra}")
endif()
