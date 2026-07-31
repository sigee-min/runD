if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(runD_kernel LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Kernel benchmark build type" FORCE)
  endif()

  enable_testing()
endif()

option(
  RUND_TEST_KERNEL
  "Register kernel contract CTests and create the test executables"
  OFF)

if(NOT TARGET math32)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../math32" "${CMAKE_CURRENT_BINARY_DIR}/math32")
endif()
if(NOT TARGET math64)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../math64" "${CMAKE_CURRENT_BINARY_DIR}/math64")
endif()

include(cmake/kernel/sources.cmake)
if(RUND_TEST_KERNEL)
  if(NOT TARGET rund-test-assertion)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/root/test/assert.cmake")
    rund_test_assertion()
  endif()
  include(cmake/tests/sources.cmake)
endif()

function(rund_kernel_object_context target)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_link_libraries(${target} PRIVATE math32 math64)
endfunction()

function(rund_kernel_archive_context target)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_link_libraries(${target} PUBLIC math32 math64)
endfunction()

foreach(component IN LISTS KERNEL_COMPONENTS)
  string(TOLOWER "${component}" component_name)
  set(object_target "kernel-object-${component_name}")
  add_library(${object_target} OBJECT ${KERNEL_${component}_SOURCES})
  set_target_properties(${object_target} PROPERTIES
    EXCLUDE_FROM_ALL TRUE
    RUND_KERNEL_COMPONENT "${component}")
  rund_kernel_object_context(${object_target})
  get_target_property(object_sources ${object_target} SOURCES)
  if(NOT "${object_sources}" STREQUAL "${KERNEL_${component}_SOURCES}")
    message(FATAL_ERROR
      "Kernel component ${component} has non-canonical object inputs")
  endif()
  set(KERNEL_OBJECT_${component}_TARGET "${object_target}")
endforeach()

add_library(kernel STATIC)
set(kernel_aggregate_objects)
set(kernel_aggregate_object_targets)
foreach(component IN LISTS KERNEL_COMPONENTS)
  set(object_input "$<TARGET_OBJECTS:${KERNEL_OBJECT_${component}_TARGET}>")
  target_sources(kernel PRIVATE "${object_input}")
  list(APPEND kernel_aggregate_objects "${object_input}")
  list(APPEND kernel_aggregate_object_targets
    "${KERNEL_OBJECT_${component}_TARGET}")
endforeach()
rund_kernel_archive_context(kernel)
set_target_properties(kernel PROPERTIES
  RUND_KERNEL_COMPONENTS "${KERNEL_COMPONENTS}"
  RUND_KERNEL_OBJECT_TARGETS "${kernel_aggregate_object_targets}"
  RUND_KERNEL_SOURCE_MEMBERS "${KERNEL_SOURCES}")
get_target_property(kernel_sources kernel SOURCES)
if(NOT "${kernel_sources}" STREQUAL "${kernel_aggregate_objects}")
  message(FATAL_ERROR
    "Installed Kernel archive is not the complete component object union")
endif()

function(rund_kernel_internal_closure target)
  if(NOT ARGN)
    message(FATAL_ERROR "Kernel closure ${target} has no components")
  endif()
  set(components ${ARGN})
  add_library(${target} STATIC EXCLUDE_FROM_ALL)
  set(object_inputs)
  set(object_targets)
  set(source_members)
  set(seen_components)
  foreach(component IN LISTS components)
    list(FIND KERNEL_COMPONENTS "${component}" component_index)
    if(component_index EQUAL -1)
      message(FATAL_ERROR
        "Kernel closure ${target} has unknown component ${component}")
    endif()
    list(FIND seen_components "${component}" repeated_component)
    if(NOT repeated_component EQUAL -1)
      message(FATAL_ERROR
        "Kernel closure ${target} repeats component ${component}")
    endif()
    list(APPEND seen_components "${component}")
    set(object_target "${KERNEL_OBJECT_${component}_TARGET}")
    if(NOT object_target)
      message(FATAL_ERROR
        "Kernel closure ${target} has no object owner for ${component}")
    endif()
    if(NOT TARGET ${object_target})
      message(FATAL_ERROR
        "Kernel closure ${target} has missing object owner ${object_target}")
    endif()
    set(object_input "$<TARGET_OBJECTS:${object_target}>")
    target_sources(${target} PRIVATE "${object_input}")
    list(APPEND object_inputs "${object_input}")
    list(APPEND object_targets "${object_target}")
    list(APPEND source_members ${KERNEL_${component}_SOURCES})
  endforeach()
  rund_kernel_archive_context(${target})
  set_target_properties(${target} PROPERTIES
    RUND_KERNEL_COMPONENTS "${components}"
    RUND_KERNEL_OBJECT_TARGETS "${object_targets}"
    RUND_KERNEL_SOURCE_MEMBERS "${source_members}")
  get_target_property(closure_sources ${target} SOURCES)
  if(NOT "${closure_sources}" STREQUAL "${object_inputs}")
    message(FATAL_ERROR
      "Kernel closure ${target} has non-canonical object membership")
  endif()
endfunction()

rund_kernel_internal_closure(kernel-closure-core CORE)
rund_kernel_internal_closure(kernel-closure-compute COMPUTE)
rund_kernel_internal_closure(kernel-closure-dispatch DISPATCH)
rund_kernel_internal_closure(kernel-closure-execution
  CORE PROGRAM REDUCTION SCHEDULE)

# PROGRAM invokes prepared DISPATCH execution, while DISPATCH records into
# CORE/SCHEDULE state. Model that real static-archive SCC so one-pass linkers
# rescan the two disjoint object owners instead of depending on linker order.
target_link_libraries(kernel-closure-dispatch
  PUBLIC kernel-closure-execution)
target_link_libraries(kernel-closure-execution
  PUBLIC kernel-closure-dispatch)
set_target_properties(
  kernel-closure-dispatch
  kernel-closure-execution
  PROPERTIES LINK_INTERFACE_MULTIPLICITY 2)

function(rund_kernel_require_closure target)
  set(expected_components ${ARGN})
  get_target_property(actual_components ${target} RUND_KERNEL_COMPONENTS)
  if(NOT "${actual_components}" STREQUAL "${expected_components}")
    message(FATAL_ERROR
      "Kernel closure ${target} has non-canonical components")
  endif()
  set(expected_objects)
  set(expected_sources)
  foreach(component IN LISTS expected_components)
    list(APPEND expected_objects "${KERNEL_OBJECT_${component}_TARGET}")
    list(APPEND expected_sources ${KERNEL_${component}_SOURCES})
  endforeach()
  get_target_property(actual_objects ${target} RUND_KERNEL_OBJECT_TARGETS)
  get_target_property(actual_sources ${target} RUND_KERNEL_SOURCE_MEMBERS)
  if(NOT "${actual_objects}" STREQUAL "${expected_objects}" OR
     NOT "${actual_sources}" STREQUAL "${expected_sources}")
    message(FATAL_ERROR
      "Kernel closure ${target} has non-canonical source ownership")
  endif()
endfunction()

rund_kernel_require_closure(kernel-closure-core CORE)
rund_kernel_require_closure(kernel-closure-compute COMPUTE)
rund_kernel_require_closure(kernel-closure-dispatch DISPATCH)
rund_kernel_require_closure(kernel-closure-execution
  CORE PROGRAM REDUCTION SCHEDULE)

set(kernel_internal_view_objects)
set(kernel_internal_view_sources)
foreach(view IN ITEMS compute dispatch execution)
  get_target_property(view_objects
    kernel-closure-${view} RUND_KERNEL_OBJECT_TARGETS)
  get_target_property(view_sources
    kernel-closure-${view} RUND_KERNEL_SOURCE_MEMBERS)
  foreach(object_target IN LISTS view_objects)
    list(FIND kernel_internal_view_objects "${object_target}" overlapping_object)
    if(NOT overlapping_object EQUAL -1)
      message(FATAL_ERROR
        "Kernel internal views overlap at object owner ${object_target}")
    endif()
    list(APPEND kernel_internal_view_objects "${object_target}")
  endforeach()
  foreach(source IN LISTS view_sources)
    list(FIND kernel_internal_view_sources "${source}" overlapping_source)
    if(NOT overlapping_source EQUAL -1)
      message(FATAL_ERROR
        "Kernel internal views overlap at ${source}")
    endif()
    list(APPEND kernel_internal_view_sources "${source}")
  endforeach()
endforeach()
set(kernel_expected_internal_view_objects
  ${KERNEL_OBJECT_COMPUTE_TARGET}
  ${KERNEL_OBJECT_DISPATCH_TARGET}
  ${KERNEL_OBJECT_CORE_TARGET}
  ${KERNEL_OBJECT_PROGRAM_TARGET}
  ${KERNEL_OBJECT_REDUCTION_TARGET}
  ${KERNEL_OBJECT_SCHEDULE_TARGET})
set(kernel_expected_internal_view_sources
  ${KERNEL_COMPUTE_SOURCES}
  ${KERNEL_DISPATCH_SOURCES}
  ${KERNEL_CORE_SOURCES}
  ${KERNEL_PROGRAM_SOURCES}
  ${KERNEL_REDUCTION_SOURCES}
  ${KERNEL_SCHEDULE_SOURCES})
if(NOT "${kernel_internal_view_objects}" STREQUAL
       "${kernel_expected_internal_view_objects}" OR
   NOT "${kernel_internal_view_sources}" STREQUAL
       "${kernel_expected_internal_view_sources}")
  message(FATAL_ERROR
    "Kernel internal view union is not canonical")
endif()

get_target_property(kernel_public_components kernel RUND_KERNEL_COMPONENTS)
get_target_property(kernel_public_objects kernel RUND_KERNEL_OBJECT_TARGETS)
get_target_property(kernel_public_sources kernel RUND_KERNEL_SOURCE_MEMBERS)
if(NOT "${kernel_public_components}" STREQUAL "${KERNEL_COMPONENTS}" OR
   NOT "${kernel_public_objects}" STREQUAL
       "${kernel_aggregate_object_targets}" OR
   NOT "${kernel_public_sources}" STREQUAL "${KERNEL_SOURCES}")
  message(FATAL_ERROR
    "Installed Kernel archive is not the canonical public union")
endif()

if(RUND_TEST_KERNEL)
  add_library(kernel-test-allocation OBJECT
    tests/contract/support/allocation.cpp)
  set_target_properties(kernel-test-allocation PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(kernel-test-allocation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  target_compile_features(kernel-test-allocation PRIVATE cxx_std_20)

  function(rund_kernel_test name target label link_target)
    if(NOT TARGET ${link_target})
      message(FATAL_ERROR
        "Kernel contract ${name} has missing link owner ${link_target}")
    endif()
    add_executable(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    target_include_directories(${target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
      "${CMAKE_CURRENT_SOURCE_DIR}/../tools"
    )
    target_link_libraries(${target} PRIVATE
      ${link_target}
      rund-test-assertion)
    set_target_properties(${target} PROPERTIES
      RUND_KERNEL_LINK_OWNER "${link_target}")
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${name} PROPERTIES
      LABELS "rund_kernel;rund_contract;rund_owner_local;${label}"
      TIMEOUT 900)
    if(COMMAND rund_test_route)
      rund_test_route(${name} TARGETS ${target})
    endif()
  endfunction()

  rund_kernel_test(kernel.core kernel-core
    rund_kernel_core kernel-closure-core ${KERNEL_CORE_TEST_SOURCES})
  rund_kernel_test(kernel.execution kernel-execution
    rund_kernel_execution kernel ${KERNEL_EXECUTION_TEST_SOURCES})
  rund_kernel_test(kernel.program kernel-program
    rund_kernel_program kernel
    ${KERNEL_PROGRAM_TEST_SOURCES}
    $<TARGET_OBJECTS:kernel-test-allocation>)
  rund_kernel_test(kernel.compute kernel-compute
    rund_kernel_compute kernel
    ${KERNEL_COMPUTE_CORE_TEST_SOURCES}
    $<TARGET_OBJECTS:kernel-test-allocation>)
  rund_kernel_test(kernel.compute.model kernel-compute-model
    rund_kernel_compute_model kernel
    ${KERNEL_COMPUTE_MODEL_TEST_SOURCES}
    $<TARGET_OBJECTS:kernel-test-allocation>)
  rund_kernel_test(kernel.reduction kernel-reduction
    rund_kernel_reduction kernel ${KERNEL_REDUCTION_TEST_SOURCES})
endif()
