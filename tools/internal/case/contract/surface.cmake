function(rund_case_contract_surface source_root)
  # Compare every live product identity in one interpreter. The registry stays
  # the only case and capability list; this contract checks its projections.
  rund_case_contract_invoke(
    direct_list_result direct_list direct_list_error
    "${source_root}" -D LIST_CASES=ON)
  if(NOT direct_list_result EQUAL 0)
    string(STRIP "${direct_list}\n${direct_list_error}" detail)
    message(FATAL_ERROR "Canonical case list failed: ${detail}")
  endif()
  string(STRIP "${direct_list}" direct_list)
  rund_case_contract_catalog(cached_list "${source_root}" list - -)
  if(NOT cached_list STREQUAL direct_list)
    message(FATAL_ERROR
      "Authenticated case catalog disagrees with the canonical full list")
  endif()
  rund_case_contract_case(
    "${source_root}" "^not-an-exact-case$" "" "")
  rund_case_contract_query_failure(
    "${source_root}" compute.backend cpu
    "does not support --backend cpu" "an untagged Compute case")
  rund_case_contract_query_failure(
    "${source_root}" compute.backend metal
    "does not support --backend metal" "an untagged Compute case")

  include("${source_root}/node/cmake/node/profiles.cmake")
  include("${source_root}/node/cmake/node/contract/routes.cmake")
  include("${source_root}/node/cmake/tests/registry.cmake")
  rund_node_registry_load(
    "${source_root}/node/tests/contract/cases.def")

  set(runtime_base_groups
    runtime
    runtime.task
    runtime.task.host
    runtime.task.net
    runtime.task.reactor
    runtime.task.replay)
  foreach(group IN LISTS runtime_base_groups)
    rund_node_route(target resource "${group}")
    if(NOT target STREQUAL "node-runtime" OR NOT resource STREQUAL "-")
      message(FATAL_ERROR
        "Runtime-base group ${group} does not use canonical node-runtime: ${target}, ${resource}")
    endif()
  endforeach()
  list(LENGTH runtime_base_groups runtime_base_group_count)
  math(EXPR last_runtime_base_group "${runtime_base_group_count} - 1")
  foreach(index RANGE 0 ${last_runtime_base_group})
    set(runtime_base_group_count_${index} 0)
  endforeach()
  list(LENGTH NODE_TEST_CASES node_case_count)
  math(EXPR last_node_case "${node_case_count} - 1")
  foreach(index RANGE 0 ${last_node_case})
    list(GET NODE_TEST_CASE_GROUPS ${index} group)
    list(FIND runtime_base_groups "${group}" runtime_group_index)
    if(runtime_group_index EQUAL -1)
      continue()
    endif()
    set(count_var "runtime_base_group_count_${runtime_group_index}")
    math(EXPR ${count_var} "${${count_var}} + 1")
    list(GET NODE_TEST_CASES ${index} name)
    list(GET NODE_TEST_CASE_LINK_PROFILES ${index} profile)
    if(NOT profile STREQUAL "runtime")
      message(FATAL_ERROR
        "Runtime-base case ${name} has non-runtime profile ${profile}")
    endif()
  endforeach()
  foreach(index RANGE 0 ${last_runtime_base_group})
    set(count_var "runtime_base_group_count_${index}")
    if(${${count_var}} LESS 1)
      list(GET runtime_base_groups ${index} group)
      message(FATAL_ERROR
        "Runtime-base group ${group} has no registry cases")
    endif()
  endforeach()

  if(NOT "${RUND_NODE_FOCUSED_BACKENDS}" STREQUAL
      "cpu;metal;vulkan")
    message(FATAL_ERROR
      "Focused backend product surface drifted: ${RUND_NODE_FOCUSED_BACKENDS}")
  endif()
  rund_node_project_native_components(full_native "")
  rund_node_project_native_components(cpu_native cpu)
  rund_node_project_native_components(metal_native metal)
  rund_node_project_native_components(vulkan_native vulkan)
  if(NOT "${full_native}" STREQUAL "ACCEL_METAL;ACCEL_VULKAN" OR
     NOT "${cpu_native}" STREQUAL "" OR
     NOT "${metal_native}" STREQUAL "ACCEL_METAL" OR
     NOT "${vulkan_native}" STREQUAL "ACCEL_VULKAN")
    message(FATAL_ERROR "Focused native component projection drifted")
  endif()

  set(backend_cases)
  foreach(name IN LISTS NODE_TEST_CASES)
    rund_node_registry_case_has_tag(supports_backend "${name}" backend)
    if(supports_backend)
      list(APPEND backend_cases "${name}")
    endif()
  endforeach()
  if(NOT backend_cases)
    message(FATAL_ERROR "Backend selector has no registered cases")
  endif()

  # The tag is the admission authority. Check its universal route law instead
  # of keeping a second hand-written list that becomes stale on admission.
  foreach(name IN LISTS backend_cases)
    foreach(backend IN ITEMS metal vulkan)
      rund_node_case_route(
        selected_target selected_resource selected_profile
        "${name}" "${backend}")
      if(NOT selected_target STREQUAL "node-compute-accel" OR
         NOT selected_resource STREQUAL "rund_accel" OR
         NOT selected_profile STREQUAL "compute")
        message(FATAL_ERROR
          "Backend ${backend} changed the ${name} accelerator route: "
          "${selected_target}, ${selected_resource}, ${selected_profile}")
      endif()
    endforeach()
    rund_node_case_route(
      cpu_target cpu_resource cpu_profile "${name}" cpu)
    if(NOT cpu_target STREQUAL "node-compute-accel" OR
       NOT cpu_resource STREQUAL "-" OR
       NOT cpu_profile STREQUAL "cpu-compute")
      message(FATAL_ERROR
        "CPU selector lost the ${name} isolated closure: "
        "${cpu_target}, ${cpu_resource}, ${cpu_profile}")
    endif()
  endforeach()
endfunction()
