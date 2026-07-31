include_guard(GLOBAL)

set(RUND_NODE_FOCUSED_BACKENDS cpu metal vulkan)

# Focused native execution is a physical projection of the canonical product
# component graph.  This table is the sole backend-to-component authority;
# source ownership and the full product closure remain unchanged.
set(RUND_NODE_FOCUSED_NATIVE_COMPONENT_ROWS
  "cpu|-"
  "metal|ACCEL_METAL"
  "vulkan|ACCEL_VULKAN")

set(rund_node_projected_backends)
set(rund_node_projected_components)
foreach(row IN LISTS RUND_NODE_FOCUSED_NATIVE_COMPONENT_ROWS)
  string(REPLACE "|" ";" fields "${row}")
  list(LENGTH fields field_count)
  if(NOT field_count EQUAL 2)
    message(FATAL_ERROR "Malformed focused native component row: ${row}")
  endif()
  list(GET fields 0 backend)
  list(GET fields 1 component)
  list(FIND RUND_NODE_FOCUSED_BACKENDS "${backend}" backend_index)
  list(FIND rund_node_projected_backends "${backend}" duplicate_backend)
  list(FIND rund_node_projected_components "${component}" duplicate_component)
  if(backend_index EQUAL -1 OR NOT duplicate_backend EQUAL -1 OR
     NOT component MATCHES "^(-|ACCEL_(METAL|VULKAN))$" OR
     (NOT component STREQUAL "-" AND NOT duplicate_component EQUAL -1))
    message(FATAL_ERROR "Invalid focused native component row: ${row}")
  endif()
  list(APPEND rund_node_projected_backends "${backend}")
  if(NOT component STREQUAL "-")
    list(APPEND rund_node_projected_components "${component}")
  endif()
endforeach()
set(rund_node_expected_backends ${RUND_NODE_FOCUSED_BACKENDS})
list(SORT rund_node_projected_backends)
list(SORT rund_node_expected_backends)
list(SORT rund_node_projected_components)
if(NOT "${rund_node_projected_backends}" STREQUAL
   "${rund_node_expected_backends}" OR
   NOT "${rund_node_projected_components}" STREQUAL
   "ACCEL_METAL;ACCEL_VULKAN")
  message(FATAL_ERROR "Focused native component projection is not closed")
endif()

function(rund_node_project_native_components out backend)
  if(backend STREQUAL "")
    set(components)
    foreach(row IN LISTS RUND_NODE_FOCUSED_NATIVE_COMPONENT_ROWS)
      string(REPLACE "|" ";" fields "${row}")
      list(GET fields 1 component)
      if(NOT component STREQUAL "-")
        list(APPEND components "${component}")
      endif()
    endforeach()
    set(${out} ${components} PARENT_SCOPE)
    return()
  endif()
  list(FIND RUND_NODE_FOCUSED_BACKENDS "${backend}" backend_index)
  if(backend_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node focused backend: ${backend}")
  endif()
  set(component)
  set(found FALSE)
  foreach(row IN LISTS RUND_NODE_FOCUSED_NATIVE_COMPONENT_ROWS)
    string(REPLACE "|" ";" fields "${row}")
    list(LENGTH fields field_count)
    if(NOT field_count EQUAL 2)
      message(FATAL_ERROR "Malformed focused native component row: ${row}")
    endif()
    list(GET fields 0 candidate)
    if(candidate STREQUAL backend)
      list(GET fields 1 component)
      set(found TRUE)
      break()
    endif()
  endforeach()
  if(NOT found)
    message(FATAL_ERROR
      "Focused backend ${backend} has no native component projection")
  endif()
  if(component STREQUAL "-")
    set(component)
  endif()
  set(${out} ${component} PARENT_SCOPE)
endfunction()

# Test rows select a profile by name.  This table is the sole owner that maps
# that registered test identity to its production SCC root, supported focused
# backend projection, focus materialization scope, and admitted owner bound. A
# dash means that the profile has no backend projection. `case` retains one
# exact owner; `profile` retains a bounded cohort sharing one
# target/resource/profile closure so switching its cases cannot relink an
# identical product executable.
set(RUND_NODE_LINK_PROFILE_ROWS
  "header|-|-|case|1"
  "numeric|NUMERIC|-|case|1"
  "cpu-simd|CPU_SIMD|-|case|1"
  "cpu-accel|CPU_ACCEL|-|case|1"
  "accel|ACCEL_EXECUTION|-|case|1"
  "cpu-compute|CPU_COMPUTE|-|case|1"
  "compute|COMPUTE_EXECUTION|cpu-compute|case|1"
  "runtime|RUNTIME_BASE|-|case|1"
  "cpu-product|CPU_RUNTIME_PRODUCT|-|case|1"
  "pipeline-product|RUNTIME_PRODUCT|-|case|1"
  "product|RUNTIME_PRODUCT|-|profile|2")

set(RUND_NODE_LINK_PROFILES)
set(RUND_NODE_PROFILE_ROOTS)
set(RUND_NODE_PROFILE_CPU_PROJECTIONS)
set(RUND_NODE_PROFILE_FOCUS_SCOPES)
set(RUND_NODE_PROFILE_FOCUS_BOUNDS)
foreach(row IN LISTS RUND_NODE_LINK_PROFILE_ROWS)
  string(REPLACE "|" ";" fields "${row}")
  list(LENGTH fields field_count)
  if(NOT field_count EQUAL 5)
    message(FATAL_ERROR "Malformed Node link profile row: ${row}")
  endif()
  list(GET fields 0 profile)
  list(GET fields 1 root)
  list(GET fields 2 cpu_projection)
  list(GET fields 3 focus_scope)
  list(GET fields 4 focus_bound)
  list(FIND RUND_NODE_LINK_PROFILES "${profile}" duplicate_profile)
  if(NOT duplicate_profile EQUAL -1)
    message(FATAL_ERROR "Duplicate Node link profile: ${profile}")
  endif()
  if(NOT profile MATCHES "^[a-z][a-z0-9-]*$" OR
     NOT root MATCHES "^(-|[A-Z][A-Z0-9_]*)$" OR
     NOT cpu_projection MATCHES "^(-|[a-z][a-z0-9-]*)$" OR
     NOT focus_scope MATCHES "^(case|profile)$" OR
     NOT focus_bound MATCHES "^[1-9][0-9]*$" OR
     (focus_scope STREQUAL "case" AND NOT focus_bound EQUAL 1))
    message(FATAL_ERROR "Malformed Node link profile: ${row}")
  endif()
  list(APPEND RUND_NODE_LINK_PROFILES "${profile}")
  list(APPEND RUND_NODE_PROFILE_ROOTS "${root}")
  list(APPEND RUND_NODE_PROFILE_CPU_PROJECTIONS "${cpu_projection}")
  list(APPEND RUND_NODE_PROFILE_FOCUS_SCOPES "${focus_scope}")
  list(APPEND RUND_NODE_PROFILE_FOCUS_BOUNDS "${focus_bound}")
endforeach()

function(rund_node_profile_index out profile)
  list(FIND RUND_NODE_LINK_PROFILES "${profile}" index)
  if(index EQUAL -1)
    message(FATAL_ERROR "Unknown Node link profile: ${profile}")
  endif()
  set(${out} "${index}" PARENT_SCOPE)
endfunction()

function(rund_node_profile_value out values profile)
  rund_node_profile_index(index "${profile}")
  list(GET ${values} ${index} value)
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_profile_focus_scope out profile)
  rund_node_profile_value(
    value RUND_NODE_PROFILE_FOCUS_SCOPES "${profile}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(rund_node_profile_focus_bound out profile)
  rund_node_profile_value(
    value RUND_NODE_PROFILE_FOCUS_BOUNDS "${profile}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

list(LENGTH RUND_NODE_LINK_PROFILES profile_count)
math(EXPR last_profile "${profile_count} - 1")
foreach(profile_index RANGE 0 ${last_profile})
  list(GET RUND_NODE_LINK_PROFILES ${profile_index} profile)
  list(GET RUND_NODE_PROFILE_CPU_PROJECTIONS
    ${profile_index} cpu_projection)
  if(NOT cpu_projection STREQUAL "-")
    list(FIND RUND_NODE_LINK_PROFILES "${cpu_projection}" projection_index)
    if(projection_index EQUAL -1 OR cpu_projection STREQUAL profile)
      message(FATAL_ERROR
        "Node link profile ${profile} has invalid CPU projection ${cpu_projection}")
    endif()
  endif()
endforeach()

function(rund_node_project_profile out profile backend)
  if(backend STREQUAL "")
    set(${out} "${profile}" PARENT_SCOPE)
    return()
  endif()
  list(FIND RUND_NODE_FOCUSED_BACKENDS "${backend}" backend_index)
  if(backend_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node focused backend: ${backend}")
  endif()
  if(NOT backend STREQUAL "cpu")
    set(${out} "${profile}" PARENT_SCOPE)
    return()
  endif()

  rund_node_profile_value(
    projected RUND_NODE_PROFILE_CPU_PROJECTIONS "${profile}")
  if(projected STREQUAL "-")
    message(FATAL_ERROR
      "Node link profile ${profile} does not support backend ${backend}")
  endif()
  set(${out} "${projected}" PARENT_SCOPE)
endfunction()
