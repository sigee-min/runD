set(RUND_NODE_TEST_SUPPORT_ROWS
  "tests/contract/runtime/product/support.cpp|node-test-support-runtime-product"
  "tests/contract/compute/allocation.cpp|node-test-support-compute-allocation"
  "tests/contract/runtime/task/coroutine/allocation.cpp|node-test-support-task-allocation")

function(rund_node_support_target out relative)
  string(SHA256 source_key "${relative}")
  get_property(object_target GLOBAL PROPERTY
    "RUND_NODE_TEST_SUPPORT_TARGET_${source_key}")
  set(${out} "${object_target}" PARENT_SCOPE)
endfunction()

function(rund_node_validate_support_rows)
  set(sources)
  set(targets)
  foreach(row IN LISTS RUND_NODE_TEST_SUPPORT_ROWS)
    string(REPLACE "|" ";" fields "${row}")
    list(LENGTH fields field_count)
    if(NOT field_count EQUAL 2)
      message(FATAL_ERROR "Malformed Node test support row: ${row}")
    endif()
    list(GET fields 0 source)
    list(GET fields 1 target)
    if(NOT source MATCHES
        "^tests/contract/[A-Za-z0-9_./-]+[.]cpp$" OR
       NOT target MATCHES "^[A-Za-z0-9_.+-]+$")
      message(FATAL_ERROR "Invalid Node test support row: ${row}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
      message(FATAL_ERROR "Missing Node test support source: ${source}")
    endif()
    list(FIND sources "${source}" duplicate_source)
    list(FIND targets "${target}" duplicate_target)
    if(NOT duplicate_source EQUAL -1 OR NOT duplicate_target EQUAL -1)
      message(FATAL_ERROR "Duplicate Node test support row: ${row}")
    endif()
    list(FIND NODE_TEST_CASE_OWNERS "${source}" case_owner)
    if(NOT case_owner EQUAL -1)
      message(FATAL_ERROR
        "Node test support must remain owner-neutral: ${source}")
    endif()
    list(APPEND sources "${source}")
    list(APPEND targets "${target}")
    string(SHA256 source_key "${source}")
    set_property(GLOBAL PROPERTY
      "RUND_NODE_TEST_SUPPORT_TARGET_${source_key}" "${target}")
  endforeach()
endfunction()

function(rund_node_shared_target out relative link_root)
  string(SHA256 source_key "${relative}|${link_root}")
  get_property(object_target GLOBAL PROPERTY
    "RUND_NODE_TEST_SHARED_TARGET_${source_key}")
  set(${out} "${object_target}" PARENT_SCOPE)
endfunction()

function(rund_node_track_shared object_target consumer)
  set_property(GLOBAL APPEND PROPERTY RUND_NODE_TEST_SHARED_PAIRS
    "${object_target}|${consumer}")
  set_property(GLOBAL APPEND PROPERTY
    "RUND_NODE_TEST_SHARED_CONSUMERS_${object_target}" "${consumer}")
endfunction()

function(rund_node_validate_shared_contexts)
  get_property(objects GLOBAL PROPERTY RUND_NODE_TEST_SHARED_OBJECTS)
  list(REMOVE_DUPLICATES objects)
  set(properties
    COMPILE_DEFINITIONS
    INCLUDE_DIRECTORIES
    SYSTEM_INCLUDE_DIRECTORIES
    COMPILE_OPTIONS
    COMPILE_FEATURES
    LINK_LIBRARIES
    CXX_STANDARD
    CXX_STANDARD_REQUIRED
    CXX_EXTENSIONS
    POSITION_INDEPENDENT_CODE
    INTERPROCEDURAL_OPTIMIZATION
    INTERPROCEDURAL_OPTIMIZATION_DEBUG
    INTERPROCEDURAL_OPTIMIZATION_RELEASE
    COMPILE_WARNING_AS_ERROR
    PRECOMPILE_HEADERS
    DISABLE_PRECOMPILE_HEADERS
    RUND_NODE_COMPILE_CONTEXT)

  get_property(pairs GLOBAL PROPERTY RUND_NODE_TEST_SHARED_PAIRS)
  foreach(pair IN LISTS pairs)
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 object_target)
    list(GET fields 1 consumer)
    if(NOT TARGET ${object_target} OR NOT TARGET ${consumer})
      message(FATAL_ERROR
        "Node shared compile context has a missing target: ${pair}")
    endif()
    foreach(property IN LISTS properties)
      get_target_property(object_value ${object_target} ${property})
      get_target_property(consumer_value ${consumer} ${property})
      if(object_value MATCHES "-NOTFOUND$")
        set(object_value)
      endif()
      if(consumer_value MATCHES "-NOTFOUND$")
        set(consumer_value)
      endif()
      if(NOT "${object_value}" STREQUAL "${consumer_value}")
        message(FATAL_ERROR
          "Node shared compile context diverged for ${object_target} and ${consumer}: ${property}")
      endif()
    endforeach()
  endforeach()

  foreach(object_target IN LISTS objects)
    get_property(consumers GLOBAL PROPERTY
      "RUND_NODE_TEST_SHARED_CONSUMERS_${object_target}")
    list(REMOVE_DUPLICATES consumers)
    list(LENGTH consumers consumer_count)
    if(NOT consumer_count EQUAL 2)
      message(FATAL_ERROR
        "Node shared object ${object_target} has ${consumer_count} consumers, expected 2")
    endif()
  endforeach()
endfunction()

function(rund_node_prepare_shared target link_root link_target)
  if(RUND_NODE_TEST_TAG STREQUAL "")
    return()
  endif()

  rund_node_tag_cohorts(cohorts "${RUND_NODE_TEST_TAG}")
  foreach(cohort IN LISTS cohorts)
    string(REPLACE "|" ";" fields "${cohort}")
    list(GET fields 0 profile)
    list(GET fields 1 resource)
    list(GET fields 2 encoded_cases)
    string(REPLACE "," ";" cases "${encoded_cases}")

    rund_node_link_profiles_target(
      cohort_link_target cohort_root "${profile}")
    if(NOT cohort_root STREQUAL link_root)
      continue()
    endif()
    if(NOT "${cohort_link_target}" STREQUAL "${link_target}")
      message(FATAL_ERROR
        "Node tagged cohort ${profile} changed its link target")
    endif()

    set(route_target)
    foreach(name IN LISTS cases)
      rund_node_case_group(group "${name}")
      rund_node_route(case_target case_resource "${group}")
      if(NOT case_resource STREQUAL resource)
        message(FATAL_ERROR
          "Node tagged cohort ${profile} changed its resource")
      endif()
      if(route_target AND NOT case_target STREQUAL route_target)
        set(route_target)
        break()
      endif()
      set(route_target "${case_target}")
    endforeach()
    if(NOT route_target STREQUAL target)
      continue()
    endif()

    rund_node_subset_sources(cohort_sources ${cases})
    rund_node_source_index(primary_sources shared_source_count ${ARGN})
    set(shared_sources)
    set(shared_relative)
    foreach(source IN LISTS cohort_sources)
      rund_node_source_relative(relative "${source}")
      list(FIND primary_sources "${relative}" primary_index)
      if(primary_index EQUAL -1)
        message(FATAL_ERROR
          "Node tagged source ${relative} is absent from ${target}")
      endif()
      rund_node_support_target(support_target "${relative}")
      if(support_target)
        continue()
      endif()
      list(APPEND shared_sources "${source}")
      list(APPEND shared_relative "${relative}")
    endforeach()
    if(NOT shared_sources)
      continue()
    endif()

    string(REPLACE ":" "." tag_key "${RUND_NODE_TEST_TAG}")
    set(object_target "node-test-${tag_key}-${profile}-shared")
    if(TARGET ${object_target})
      message(FATAL_ERROR
        "Node shared object owner already exists: ${object_target}")
    endif()
    add_library(${object_target} OBJECT ${shared_sources})
    set_target_properties(${object_target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    rund_node_compile_context(
      ${object_target} CASE PLATFORM LINK_ROOT "${link_root}")
    target_link_libraries(${object_target} PRIVATE rund-test-assertion)
    if(cohort_link_target)
      target_link_libraries(${object_target} PRIVATE
        ${cohort_link_target}
        math32
        math64)
    endif()

    foreach(relative IN LISTS shared_relative)
      rund_node_shared_target(
        existing_target "${relative}" "${link_root}")
      if(existing_target)
        message(FATAL_ERROR
          "Node shared source ${relative} already belongs to ${existing_target}")
      endif()
      string(SHA256 source_key "${relative}|${link_root}")
      set_property(GLOBAL PROPERTY
        "RUND_NODE_TEST_SHARED_TARGET_${source_key}" "${object_target}")
    endforeach()
    set_property(GLOBAL APPEND PROPERTY
      RUND_NODE_TEST_SHARED_OBJECTS "${object_target}")

    get_property(validation_scheduled GLOBAL PROPERTY
      RUND_NODE_TEST_SHARED_VALIDATION SET)
    if(NOT validation_scheduled)
      set_property(GLOBAL PROPERTY
        RUND_NODE_TEST_SHARED_VALIDATION TRUE)
      cmake_language(DEFER CALL rund_node_validate_shared_contexts)
    endif()
  endforeach()
endfunction()

function(rund_node_materialize_support out shared_out link_root)
  set(materialized)
  set(shared_objects)
  foreach(source IN LISTS ARGN)
    rund_node_source_relative(relative "${source}")
    rund_node_support_target(object_target "${relative}")
    if(object_target)
      if(NOT TARGET ${object_target})
        add_library(${object_target} OBJECT "${source}")
        set_target_properties(${object_target} PROPERTIES
          EXCLUDE_FROM_ALL TRUE)
        rund_node_compile_context(${object_target} PLATFORM MATH)
      endif()
      list(APPEND materialized "$<TARGET_OBJECTS:${object_target}>")
      continue()
    endif()

    rund_node_shared_target(
      object_target "${relative}" "${link_root}")
    if(object_target)
      list(APPEND materialized "$<TARGET_OBJECTS:${object_target}>")
      list(APPEND shared_objects "${object_target}")
    else()
      list(APPEND materialized "${source}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES materialized)
  list(REMOVE_DUPLICATES shared_objects)
  set(${out} ${materialized} PARENT_SCOPE)
  set(${shared_out} ${shared_objects} PARENT_SCOPE)
endfunction()
