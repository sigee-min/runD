set(RUND_NODE_TEST_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/contract/cases.def")
include("${CMAKE_CURRENT_LIST_DIR}/../registry.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../node/contract/routes.cmake")
rund_node_registry_load("${RUND_NODE_TEST_FILE}")

function(rund_node_case_value out values name)
  list(FIND NODE_TEST_CASES "${name}" index)
  if(index EQUAL -1)
    set(${out} "" PARENT_SCOPE)
    return()
  endif()
  list(GET ${values} ${index} value)
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_symbol out name)
  rund_node_case_value(value NODE_TEST_CASE_SYMBOLS "${name}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_owner out name)
  rund_node_case_value(value NODE_TEST_CASE_OWNERS "${name}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_group out name)
  rund_node_case_value(value NODE_TEST_CASE_GROUPS "${name}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_suite out name)
  rund_node_case_value(value NODE_TEST_CASE_SUITES "${name}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_link_profile out name)
  rund_node_case_value(value NODE_TEST_CASE_LINK_PROFILES "${name}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_case_effective_link_profile out name)
  rund_node_case_route(
    unused_target unused_resource value "${name}"
    "${RUND_NODE_FOCUSED_BACKEND}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_cases_with_tag out tag)
  rund_node_require_test_tag("${tag}")
  set(matches)
  list(LENGTH NODE_TEST_CASES case_count)
  math(EXPR last_case "${case_count} - 1")
  foreach(index RANGE 0 ${last_case})
    list(GET NODE_TEST_CASES ${index} name)
    rund_node_registry_case_has_tag(has_tag "${name}" "${tag}")
    if(has_tag)
      list(APPEND matches "${name}")
    endif()
  endforeach()
  set(${out} ${matches} PARENT_SCOPE)
endfunction()

function(rund_node_require_group_link_profiles group expected_root)
  set(expected_profiles ${ARGN})
  if(NOT expected_profiles)
    message(FATAL_ERROR "Node test group ${group} has no expected profiles")
  endif()
  list(LENGTH expected_profiles expected_count)
  list(REMOVE_DUPLICATES expected_profiles)
  list(LENGTH expected_profiles unique_expected_count)
  if(NOT expected_count EQUAL unique_expected_count)
    message(FATAL_ERROR
      "Node test group ${group} repeats an expected link profile")
  endif()
  foreach(expected IN LISTS expected_profiles)
    list(FIND RUND_NODE_LINK_PROFILES "${expected}" expected_index)
    if(expected_index EQUAL -1)
      message(FATAL_ERROR
        "Node test group ${group} requires unknown link profile ${expected}")
    endif()
  endforeach()

  set(actual_profiles)
  list(LENGTH NODE_TEST_CASES case_count)
  math(EXPR last_case "${case_count} - 1")
  foreach(index RANGE 0 ${last_case})
    list(GET NODE_TEST_CASE_GROUPS ${index} case_group)
    if(NOT case_group STREQUAL group)
      continue()
    endif()
    list(GET NODE_TEST_CASE_LINK_PROFILES ${index} actual)
    list(APPEND actual_profiles "${actual}")
  endforeach()
  if(NOT actual_profiles)
    message(FATAL_ERROR "Node test group ${group} has no registry cases")
  endif()
  list(REMOVE_DUPLICATES actual_profiles)
  list(SORT actual_profiles)
  list(SORT expected_profiles)
  if(NOT "${actual_profiles}" STREQUAL "${expected_profiles}")
    message(FATAL_ERROR
      "Node test group ${group} has profiles ${actual_profiles}; expected ${expected_profiles}")
  endif()
  rund_node_link_profiles_target(unused_target actual_root ${actual_profiles})
  if(NOT actual_root STREQUAL expected_root)
    message(FATAL_ERROR
      "Node test group ${group} closes at ${actual_root}; expected ${expected_root}")
  endif()
endfunction()

set(RUND_NODE_FOCUSED_CASE "${RUND_NODE_FOCUSED_CASE}" CACHE STRING
  "Internal exact Node contract case retained across CMake regeneration")
mark_as_advanced(RUND_NODE_FOCUSED_CASE)
set(rund_node_requested_focus "${RUND_NODE_FOCUSED_CASE}")
set(RUND_NODE_FOCUSED_CASE "")
if(NOT rund_node_requested_focus STREQUAL "")
  list(FIND NODE_TEST_CASES "${rund_node_requested_focus}" focus_index)
  if(NOT focus_index EQUAL -1)
    set(RUND_NODE_FOCUSED_CASE "${rund_node_requested_focus}")
  endif()
endif()

set(RUND_NODE_FOCUSED_BACKEND "${RUND_NODE_FOCUSED_BACKEND}" CACHE STRING
  "Internal exact Node contract backend retained across CMake regeneration")
mark_as_advanced(RUND_NODE_FOCUSED_BACKEND)
if(NOT RUND_NODE_FOCUSED_BACKEND STREQUAL "")
  if(NOT RUND_NODE_FOCUSED_CASE)
    message(FATAL_ERROR
      "A focused Node backend requires an exact registered case")
  endif()
  # The shared route owner validates both registry capability and the profile
  # projection before any test target is materialized.
  rund_node_case_route(
    unused_target unused_resource unused_profile
    "${RUND_NODE_FOCUSED_CASE}" "${RUND_NODE_FOCUSED_BACKEND}")
endif()

set(RUND_NODE_FOCUSED_CASES)
if(RUND_NODE_FOCUSED_CASE)
  rund_node_focus_cases(
    RUND_NODE_FOCUSED_CASES
    "${RUND_NODE_FOCUSED_CASE}" "${RUND_NODE_FOCUSED_BACKEND}")
  list(GET RUND_NODE_FOCUSED_CASES 0 rund_node_focus_anchor)
  if(NOT rund_node_focus_anchor STREQUAL RUND_NODE_FOCUSED_CASE)
    message(FATAL_ERROR
      "Focused Node profile must use canonical anchor ${rund_node_focus_anchor}, not ${RUND_NODE_FOCUSED_CASE}")
  endif()
endif()

set(RUND_NODE_TEST_TAG "${RUND_NODE_TEST_TAG}" CACHE STRING
  "Internal Node verification tag used to derive focused owner targets")
mark_as_advanced(RUND_NODE_TEST_TAG)
