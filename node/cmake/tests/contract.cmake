include("${CMAKE_CURRENT_LIST_DIR}/contract/target.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/contract/cases.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/node/contract/groups.cmake")
rund_node_validate_support_rows()
rund_node_validate_companion_rows()

if(RUND_TEST_NODE)
  rund_node_case_link_profile(
    rund_node_numeric_evidence_profile "evidence.numeric")
  if(NOT rund_node_numeric_evidence_profile STREQUAL "numeric")
    message(FATAL_ERROR
      "evidence.numeric must not depend on the Node product link profile")
  endif()
  set(compute_sources)
  rund_node_select_suite_sources(compute_sources cases/compute/sdk.def
    ${NODE_TEST_COMPUTE_COMMON_SOURCES}
    ${NODE_TEST_COMPUTE_SOURCES})
  set(compute_accel_sources)
  rund_node_select_suite_sources(compute_accel_sources cases/compute/accel.def
    ${NODE_TEST_COMPUTE_COMMON_SOURCES}
    ${NODE_TEST_COMPUTE_ACCEL_SOURCES})
  rund_node_select_suite_sources(compute_accel_sources
    cases/compute/expressions.def
    ${NODE_TEST_COMPUTE_EXPRESSIONS_SOURCES})
  rund_node_select_suite_sources(compute_accel_sources
    cases/compute/execution.def
    ${NODE_TEST_COMPUTE_COMMON_SOURCES}
    ${NODE_TEST_COMPUTE_EXECUTION_SOURCES})
  rund_node_require_suite_partitions(compute)
  rund_node_require_suite_partitions(compute.accel)
  rund_node_test(node-compute ${compute_sources})
  rund_node_test(node-compute-accel ${compute_accel_sources})
  rund_node_test(node-core ${NODE_TEST_CORE_TEST_SOURCES})

  # These six registry groups retain independent CTest processes but share
  # one Runtime-base dispatcher and link owner.
  set(runtime_sources)
  rund_node_select_suite_sources(runtime_sources cases/runtime/core.def
    ${NODE_TEST_RUNTIME_CORE_SOURCES})
  rund_node_require_suite_partitions(runtime)
  rund_node_require_group_link_profiles(runtime RUNTIME_BASE runtime)
  include("${CMAKE_CURRENT_LIST_DIR}/contract/task.cmake")
  list(APPEND runtime_sources ${task_sources})
  set(host_sources)
  rund_node_select_suite_sources(host_sources
    cases/runtime/task/host.def
    ${NODE_TEST_HOST_TEST_SOURCES})
  rund_node_require_suite_partitions(runtime.task.host)
  rund_node_require_group_link_profiles(runtime.task.host RUNTIME_BASE runtime)
  list(APPEND runtime_sources ${host_sources})
  include("${CMAKE_CURRENT_LIST_DIR}/contract/net.cmake")
  list(APPEND runtime_sources ${net_sources})
  set(reactor_sources)
  rund_node_select_suite_sources(reactor_sources
    cases/runtime/task/reactor.def
    ${NODE_TEST_REACTOR_TEST_SOURCES})
  rund_node_require_suite_partitions(runtime.task.reactor)
  rund_node_require_group_link_profiles(
    runtime.task.reactor RUNTIME_BASE runtime)
  list(APPEND runtime_sources ${reactor_sources})
  include("${CMAKE_CURRENT_LIST_DIR}/contract/replay.cmake")
  list(APPEND runtime_sources ${replay_sources})
  list(REMOVE_DUPLICATES runtime_sources)
  rund_node_test(node-runtime ${runtime_sources})
  set(runtime_compute_sources)
  rund_node_select_suite_sources(runtime_compute_sources
    cases/runtime/compute.def
    ${NODE_TEST_RUNTIME_COMPUTE_SOURCES})
  rund_node_select_suite_sources(runtime_compute_sources
    cases/runtime/telemetry.def
    ${NODE_TEST_RUNTIME_TELEMETRY_SOURCES})
  rund_node_require_suite_partitions(runtime.compute)
  rund_node_require_group_link_profiles(
    runtime.compute CPU_RUNTIME_PRODUCT cpu-product)
  rund_node_test(node-runtime-compute ${runtime_compute_sources})
  set(runtime_accel_sources)
  rund_node_select_suite_sources(runtime_accel_sources
    cases/runtime/accel.def
    ${NODE_TEST_RUNTIME_ACCEL_SOURCES})
  rund_node_require_suite_partitions(runtime.compute.accel)
  rund_node_require_group_link_profiles(
    runtime.compute.accel RUNTIME_PRODUCT pipeline-product product)
  rund_node_test(node-runtime-accel ${runtime_accel_sources})
  if(TARGET node-compute-accel)
    target_compile_definitions(node-compute-accel PRIVATE
      RUND_NODE_OPEN_PROBE=1)
  endif()
  if(NOT RUND_NODE_FOCUSED_CASE)
    rund_node_require_target_scc(node-compute CPU_COMPUTE)
    rund_node_require_target_scc(node-compute-accel COMPUTE_EXECUTION)
    rund_node_require_target_scc(node-runtime RUNTIME_BASE)
    rund_node_require_target_scc(node-runtime-compute CPU_RUNTIME_PRODUCT)
    rund_node_require_target_scc(node-runtime-accel RUNTIME_PRODUCT)
  endif()
endif()

if(RUND_TEST_ACCEL)
  include("${CMAKE_CURRENT_LIST_DIR}/contract/accel.cmake")
endif()

if(NOT RUND_NODE_TEST_TAG STREQUAL "")
  rund_node_tagged_groups("${RUND_NODE_TEST_TAG}")
endif()
rund_node_groups()
