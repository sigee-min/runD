function(rund_node_add_test_target target case_file)
  cmake_parse_arguments(RUND_TARGET "" "" "SOURCES;LINK_PROFILES" ${ARGN})
  if(RUND_TARGET_UNPARSED_ARGUMENTS OR NOT RUND_TARGET_SOURCES OR
     NOT RUND_TARGET_LINK_PROFILES)
    message(FATAL_ERROR "Incomplete Node test target definition: ${target}")
  endif()
  rund_node_link_profiles_target(
    link_target link_root ${RUND_TARGET_LINK_PROFILES})
  rund_node_prepare_shared(
    "${target}" "${link_root}" "${link_target}" ${RUND_TARGET_SOURCES})
  rund_node_materialize_support(
    target_sources shared_objects "${link_root}" ${RUND_TARGET_SOURCES})
  rund_node_runner_object(runner_object)
  rund_node_dispatch_object(dispatch_object "${target}" "${case_file}")
  add_executable(${target}
    ${dispatch_object}
    ${runner_object}
    ${target_sources})
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  rund_node_compile_context(
    ${target} CASE PLATFORM LINK_ROOT "${link_root}")
  if(link_target)
    set(link_owner "${link_target}")
  else()
    set(link_owner "-")
  endif()
  set_target_properties(${target} PROPERTIES
    RUND_NODE_REQUIRED_SCC_ROOT "${link_root}"
    RUND_NODE_REQUIRED_SCC_TARGET "${link_owner}")
  target_link_libraries(${target} PRIVATE rund-test-assertion)
  if(link_target)
    target_link_libraries(${target} PRIVATE
      ${link_target}
      math32
      math64)
  endif()
  foreach(object_target IN LISTS shared_objects)
    rund_node_track_shared("${object_target}" "${target}")
  endforeach()
endfunction()

function(rund_node_subset_test_target target)
  cmake_parse_arguments(RUND_SUBSET "" "" "CASES" ${ARGN})
  if(RUND_SUBSET_UNPARSED_ARGUMENTS OR NOT RUND_SUBSET_CASES)
    message(FATAL_ERROR "Incomplete Node test subset: ${target}")
  endif()
  rund_node_subset_sources(sources ${RUND_SUBSET_CASES})
  rund_node_subset_case_file(case_file "${target}" ${RUND_SUBSET_CASES})
  set(link_profiles)
  foreach(name IN LISTS RUND_SUBSET_CASES)
    rund_node_case_link_profile(profile "${name}")
    list(APPEND link_profiles "${profile}")
  endforeach()
  list(REMOVE_DUPLICATES link_profiles)
  rund_node_add_test_target(${target} "${case_file}"
    SOURCES ${sources}
    LINK_PROFILES ${link_profiles})
endfunction()

function(rund_node_test target)
  rund_node_focused_sources(sources "${target}" ${ARGN})
  if(RUND_NODE_FOCUSED_CASE AND NOT sources)
    return()
  endif()

  if(RUND_NODE_FOCUSED_CASE)
    get_property(case_file GLOBAL PROPERTY RUND_NODE_FOCUSED_CASE_FILE)
  else()
    rund_node_target_case_file(case_file "${target}")
  endif()

  if(RUND_NODE_FOCUSED_CASE)
    rund_node_case_effective_link_profile(
      link_profile "${RUND_NODE_FOCUSED_CASE}")
    set(link_profiles "${link_profile}")
  else()
    rund_node_target_link_profiles(link_profiles "${target}")
  endif()
  rund_node_add_test_target(${target} "${case_file}"
    SOURCES ${sources}
    LINK_PROFILES ${link_profiles})
endfunction()
