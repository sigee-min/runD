rund_node_test(accel-surface
  ${NODE_TEST_ACCEL_SURFACE_TEST_SOURCES})
if(TARGET accel-surface)
  if(NOT TARGET kernel-closure-compute)
    message(FATAL_ERROR
      "Accel graph-factory contract requires the Kernel Compute closure")
  endif()
  # Accel factories are inline assembly helpers, but graph signatures have one
  # compiled Kernel owner.  Link that exact owner without admitting any Node
  # execution SCC or recompiling a mirror signature implementation.
  target_link_libraries(accel-surface PRIVATE kernel-closure-compute)
endif()

set(accel_cpu_sources)
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/vector.def
  ${NODE_TEST_ACCEL_CPU_SIMD_VECTOR_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/dsl/basic.def
  ${NODE_TEST_ACCEL_CPU_SIMD_DSL_BASIC_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/dsl/geometry.def
  ${NODE_TEST_ACCEL_CPU_SIMD_DSL_GEOMETRY_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/dsl/stats.def
  ${NODE_TEST_ACCEL_CPU_SIMD_DSL_STATS_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/dsl/matrix.def
  ${NODE_TEST_ACCEL_CPU_SIMD_DSL_MATRIX_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/ops.def
  ${NODE_TEST_ACCEL_CPU_SIMD_OPS_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/cpu/simd/backend.def
  ${NODE_TEST_ACCEL_CPU_SIMD_BACKEND_TEST_SOURCES})
rund_node_select_suite_sources(accel_cpu_sources
  cases/accel/kernel/cpu.def
  ${NODE_TEST_ACCEL_CPU_KERNEL_TEST_SOURCES})
rund_node_require_suite_partitions(accel.cpu)
list(REMOVE_DUPLICATES accel_cpu_sources)
rund_node_test(accel-cpu ${accel_cpu_sources})

set(accel_device_sources)
rund_node_select_suite_sources(accel_device_sources
  cases/accel/kernel/core.def
  ${NODE_TEST_ACCEL_KERNEL_CORE_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/kernel/numeric.def
  ${NODE_TEST_ACCEL_KERNEL_NUMERIC_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/backend/core.def
  ${NODE_TEST_ACCEL_BACKEND_CORE_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/backend/runtime.def
  ${NODE_TEST_ACCEL_BACKEND_RUNTIME_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/backend/window.def
  ${NODE_TEST_ACCEL_BACKEND_WINDOW_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/backend/pick.def
  ${NODE_TEST_ACCEL_BACKEND_PICK_TEST_SOURCES})
rund_node_select_suite_sources(accel_device_sources
  cases/accel/kernel/fusion.def
  ${NODE_TEST_ACCEL_KERNEL_FUSION_TEST_SOURCES})
rund_node_require_suite_partitions(accel.device)
list(REMOVE_DUPLICATES accel_device_sources)
rund_node_test(accel-device ${accel_device_sources})

rund_node_require_group_link_profiles(accel.surface - header)
rund_node_require_group_link_profiles(
  accel.cpu CPU_ACCEL cpu-simd cpu-accel)
rund_node_require_group_link_profiles(accel.device ACCEL_EXECUTION accel)
if(NOT RUND_NODE_FOCUSED_CASE)
  rund_node_require_target_scc(accel-surface -)
  rund_node_require_target_scc(accel-cpu CPU_ACCEL)
  rund_node_require_target_scc(accel-device ACCEL_EXECUTION)
endif()
