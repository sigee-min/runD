include_guard(GLOBAL)

# A group owns execution meaning.  This table is the sole group-to-process and
# group-to-resource route authority shared by project configuration and the
# canonical parser that publishes the sealed exact-case catalog.
set(RUND_NODE_TEST_ROUTES
  "core|node-core|-"
  "compute|node-compute|-"
  "compute.accel|node-compute-accel|rund_accel"
  "accel.surface|accel-surface|-"
  "accel.cpu|accel-cpu|-"
  "accel.device|accel-device|rund_accel"
  "runtime|node-runtime|-"
  "runtime.compute|node-runtime-compute|-"
  "runtime.compute.accel|node-runtime-accel|rund_accel"
  "runtime.task|node-runtime|-"
  "runtime.task.host|node-runtime|-"
  "runtime.task.net|node-runtime|-"
  "runtime.task.reactor|node-runtime|-"
  "runtime.task.replay|node-runtime|-")

set(RUND_NODE_ROUTE_GROUPS)
set(RUND_NODE_ROUTE_TARGETS)
set(RUND_NODE_ROUTE_RESOURCES)
foreach(route IN LISTS RUND_NODE_TEST_ROUTES)
  string(REPLACE "|" ";" fields "${route}")
  list(LENGTH fields field_count)
  if(NOT field_count EQUAL 3)
    message(FATAL_ERROR "Malformed Node test route: ${route}")
  endif()
  list(GET fields 0 group)
  list(GET fields 1 target)
  list(GET fields 2 resource)
  if(NOT group MATCHES "^[A-Za-z][A-Za-z0-9_.-]*$" OR
     NOT target MATCHES "^[A-Za-z][A-Za-z0-9_.-]*$" OR
     NOT resource MATCHES "^(-|[A-Za-z][A-Za-z0-9_.-]*)$")
    message(FATAL_ERROR "Malformed Node test route field: ${route}")
  endif()
  list(FIND RUND_NODE_ROUTE_GROUPS "${group}" duplicate_group)
  if(NOT duplicate_group EQUAL -1)
    message(FATAL_ERROR "Duplicate Node test route group: ${group}")
  endif()
  list(APPEND RUND_NODE_ROUTE_GROUPS "${group}")
  list(APPEND RUND_NODE_ROUTE_TARGETS "${target}")
  list(APPEND RUND_NODE_ROUTE_RESOURCES "${resource}")
endforeach()

function(rund_node_require_execution_routes host_group accel_group owner)
  list(FIND RUND_NODE_ROUTE_GROUPS "${host_group}" host_index)
  list(FIND RUND_NODE_ROUTE_GROUPS "${accel_group}" accel_index)
  if(host_index EQUAL -1 OR accel_index EQUAL -1)
    message(FATAL_ERROR "Node ${owner} test routes are incomplete")
  endif()

  list(GET RUND_NODE_ROUTE_TARGETS ${host_index} host_target)
  list(GET RUND_NODE_ROUTE_TARGETS ${accel_index} accel_target)
  list(GET RUND_NODE_ROUTE_RESOURCES ${host_index} host_resource)
  list(GET RUND_NODE_ROUTE_RESOURCES ${accel_index} accel_resource)
  if(host_target STREQUAL accel_target)
    message(FATAL_ERROR
      "Node ${owner} host and accelerator groups require distinct test targets")
  endif()
  if(NOT host_resource STREQUAL "-" OR
     NOT accel_resource STREQUAL "rund_accel")
    message(FATAL_ERROR
      "Node ${owner} test routes lost their host/accelerator resource boundary")
  endif()
endfunction()

rund_node_require_execution_routes("compute" "compute.accel" "Compute")
rund_node_require_execution_routes(
  "runtime.compute" "runtime.compute.accel" "Runtime Compute")

function(rund_node_route_index out group)
  list(FIND RUND_NODE_ROUTE_GROUPS "${group}" route_index)
  if(route_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node test route group: ${group}")
  endif()
  set(${out} "${route_index}" PARENT_SCOPE)
endfunction()

function(rund_node_route out_target out_resource group)
  rund_node_route_index(route_index "${group}")
  list(GET RUND_NODE_ROUTE_TARGETS ${route_index} target)
  list(GET RUND_NODE_ROUTE_RESOURCES ${route_index} resource)
  set(${out_target} "${target}" PARENT_SCOPE)
  set(${out_resource} "${resource}" PARENT_SCOPE)
endfunction()

# Registry tags admit an execution selector.  The profile table owns the
# resulting link projection; this function is the one effective case-route
# authority used by query, focused configuration, and generated indexes.
function(rund_node_case_route out_target out_resource out_profile name backend)
  list(FIND NODE_TEST_CASES "${name}" case_index)
  if(case_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node test case: ${name}")
  endif()
  list(GET NODE_TEST_CASE_GROUPS ${case_index} group)
  list(GET NODE_TEST_CASE_LINK_PROFILES ${case_index} profile)
  rund_node_route(target resource "${group}")

  if(NOT backend STREQUAL "")
    list(FIND RUND_NODE_FOCUSED_BACKENDS "${backend}" backend_index)
    if(backend_index EQUAL -1)
      message(FATAL_ERROR "Unknown Node focused backend: ${backend}")
    endif()
    rund_node_registry_case_has_tag(
      supports_backend "${name}" backend)
    if(NOT supports_backend)
      message(FATAL_ERROR
        "Node test case ${name} does not support --backend ${backend}")
    endif()
    rund_node_project_profile(profile "${profile}" "${backend}")
    if(backend STREQUAL "cpu")
      set(resource "-")
    endif()
  endif()

  set(${out_target} "${target}" PARENT_SCOPE)
  set(${out_resource} "${resource}" PARENT_SCOPE)
  set(${out_profile} "${profile}" PARENT_SCOPE)
endfunction()

# Focus materialization is derived from the effective route, not from a second
# case list.  A profile-scoped focus is valid only when every selected case
# shares the same executable, resource, and source-suite boundary. Registry
# order supplies the stable anchor stored in CMake and the authenticated case
# catalog.
function(rund_node_focus_cases out name backend)
  rund_node_case_route(
    selected_target selected_resource selected_profile "${name}" "${backend}")
  list(FIND NODE_TEST_CASES "${name}" selected_index)
  list(GET NODE_TEST_CASE_SUITES ${selected_index} selected_suite)
  rund_node_profile_focus_scope(scope "${selected_profile}")
  if(scope STREQUAL "case")
    set(${out} "${name}" PARENT_SCOPE)
    return()
  endif()
  set(cases)
  foreach(candidate IN LISTS NODE_TEST_CASES)
    if(NOT backend STREQUAL "")
      rund_node_registry_case_has_tag(
        supports_backend "${candidate}" backend)
      if(NOT supports_backend)
        continue()
      endif()
    endif()
    rund_node_case_route(
      candidate_target candidate_resource candidate_profile
      "${candidate}" "${backend}")
    if(NOT candidate_profile STREQUAL selected_profile)
      continue()
    endif()
    rund_node_registry_case_has_tag(
      candidate_supports_backend "${candidate}" backend)
    if(candidate_supports_backend)
      message(FATAL_ERROR
        "Profile-scoped focus does not admit backend-selectable cases: ${candidate}")
    endif()
    if(NOT candidate_target STREQUAL selected_target OR
       NOT candidate_resource STREQUAL selected_resource)
      message(FATAL_ERROR
        "Node focus profile ${selected_profile} crosses target/resource boundaries")
    endif()
    list(FIND NODE_TEST_CASES "${candidate}" candidate_index)
    list(GET NODE_TEST_CASE_SUITES ${candidate_index} candidate_suite)
    if(NOT candidate_suite STREQUAL selected_suite)
      message(FATAL_ERROR
        "Node focus profile ${selected_profile} crosses source-suite boundaries")
    endif()
    list(APPEND cases "${candidate}")
  endforeach()
  if(NOT cases)
    message(FATAL_ERROR
      "Node focus profile ${selected_profile} has no registered cases")
  endif()
  rund_node_profile_focus_bound(bound "${selected_profile}")
  list(LENGTH cases case_count)
  if(case_count GREATER bound)
    message(FATAL_ERROR
      "Node focus profile ${selected_profile} has ${case_count} cases; bound is ${bound}")
  endif()
  set(${out} ${cases} PARENT_SCOPE)
endfunction()

function(rund_node_focus_anchor out name backend)
  rund_node_focus_cases(cases "${name}" "${backend}")
  list(GET cases 0 anchor)
  set(${out} "${anchor}" PARENT_SCOPE)
endfunction()
