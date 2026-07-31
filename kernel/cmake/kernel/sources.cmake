include(${CMAKE_CURRENT_LIST_DIR}/dispatch.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/program.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/reduction.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/schedule.cmake)

set(KERNEL_COMPONENTS
  CORE
  COMPUTE
  DISPATCH
  PROGRAM
  REDUCTION
  SCHEDULE
)

set(KERNEL_SOURCES
  ${KERNEL_CORE_SOURCES}
  ${KERNEL_COMPUTE_SOURCES}
  ${KERNEL_DISPATCH_SOURCES}
  ${KERNEL_PROGRAM_SOURCES}
  ${KERNEL_REDUCTION_SOURCES}
  ${KERNEL_SCHEDULE_SOURCES}
)

list(LENGTH KERNEL_CORE_SOURCES kernel_core_source_count)
if(NOT kernel_core_source_count EQUAL 1)
  message(FATAL_ERROR
    "Kernel CORE closure must own exactly one translation unit")
endif()

set(kernel_registered_sources)
foreach(component IN LISTS KERNEL_COMPONENTS)
  set(component_sources ${KERNEL_${component}_SOURCES})
  if(NOT component_sources)
    message(FATAL_ERROR "Kernel component ${component} has no sources")
  endif()

  foreach(source IN LISTS component_sources)
    if(NOT source MATCHES "^src/[a-z0-9/]+[.]cpp$")
      message(FATAL_ERROR
        "Kernel component ${component} has non-canonical source ${source}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}" OR
       IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
      message(FATAL_ERROR
        "Kernel component ${component} has missing source ${source}")
    endif()

    list(FIND kernel_registered_sources "${source}" duplicate_source)
    if(NOT duplicate_source EQUAL -1)
      message(FATAL_ERROR
        "Kernel source has two component owners: ${source}")
    endif()
    list(APPEND kernel_registered_sources "${source}")
  endforeach()
endforeach()

set(kernel_sorted_registered_sources ${kernel_registered_sources})
list(SORT kernel_sorted_registered_sources)
file(GLOB_RECURSE kernel_discovered_sources
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(SORT kernel_discovered_sources)
if(NOT "${kernel_sorted_registered_sources}" STREQUAL
       "${kernel_discovered_sources}")
  message(FATAL_ERROR
    "Kernel component registry does not exactly own kernel/src: "
    "registered=${kernel_sorted_registered_sources}; "
    "discovered=${kernel_discovered_sources}")
endif()

# Timed source policy is derived from semantic paths in the one source
# registry. There is no second list of admitted translation units.
set(kernel_hot_source_count 0)
set(kernel_hot_frame_source_count 0)
foreach(source IN LISTS KERNEL_SOURCES)
  if(source MATCHES "^src/dispatch/(kernel/(backend|execute/(bridge|plan|schedule)|telemetry/(base/plan|worker)|validation/plan)|orchestrator/run)[.]cpp$")
    math(EXPR kernel_hot_frame_source_count
      "${kernel_hot_frame_source_count} + 1")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS
        "-O3" "-fomit-frame-pointer")
    endif()
  elseif(source MATCHES "^src/program/model(/tile)?[.]cpp$" OR
         source MATCHES "^src/schedule/workspace/(capacity/proof|placement/telemetry|telemetry/record)[.]cpp$")
    math(EXPR kernel_hot_source_count "${kernel_hot_source_count} + 1")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS "-O3")
    endif()
  endif()
endforeach()

if(NOT kernel_hot_frame_source_count EQUAL 8 OR
   NOT kernel_hot_source_count EQUAL 5)
  message(FATAL_ERROR
    "Kernel timed policy lost its 8 frame-omitting and 5 hot owners")
endif()
