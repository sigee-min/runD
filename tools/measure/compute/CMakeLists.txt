if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(runD_compute_measure LANGUAGES CXX)
  include("${CMAKE_CURRENT_LIST_DIR}/../package/installed.cmake")
  add_executable(
    runD-compute-measure
    main.cpp
    suite/core.hpp
    suite/output.cpp
    suite/output.hpp
    suite/parse.cpp
    suite/reference.cpp
    suite/reference.hpp
    suite/capture.hpp
    suite/bench.hpp
    suite/bulk.hpp
    suite/warm.hpp
    suite/product.cpp
    suite/product/bounded.hpp
    suite/product/exact.hpp
    suite/product/model.hpp
    suite/product/oracle.hpp
    suite/product/report.hpp)
  target_link_libraries(runD-compute-measure PRIVATE runD::sdk)
else()
  add_executable(
    runD-compute-measure EXCLUDE_FROM_ALL
    main.cpp
    suite/core.hpp
    suite/output.cpp
    suite/output.hpp
    suite/parse.cpp
    suite/reference.cpp
    suite/reference.hpp
    suite/capture.hpp
    suite/bench.hpp
    suite/bulk.hpp
    suite/warm.hpp
    suite/product.cpp
    suite/product/bounded.hpp
    suite/product/exact.hpp
    suite/product/model.hpp
    suite/product/oracle.hpp
    suite/product/report.hpp)
  target_link_libraries(runD-compute-measure PRIVATE node)

  add_executable(
    runD-compute-focus EXCLUDE_FROM_ALL
    main.cpp
    "${CMAKE_SOURCE_DIR}/node/tests/contract/compute/allocation.cpp"
    pipeline/checkpoint.cpp
    pipeline/nested/contract.cpp
    pipeline/nested/entry.cpp
    pipeline/nested/execution.cpp
    pipeline/nested/model.cpp
    pipeline/nested/model.hpp
    pipeline/nested/prepare.cpp
    pipeline/nested/report.cpp
    pipeline/nested/sampling.cpp
    pipeline/nested/schema.cpp
    pipeline/prepare/contract.cpp
    pipeline/prepare/contract.hpp
    pipeline/prepare/model.hpp
    pipeline/prepare/program.hpp
    pipeline/prepare/report.cpp
    pipeline/prepare/report.hpp
    pipeline/prepare/report/backend.cpp
    pipeline/prepare/report/backend.hpp
    pipeline/prepare/report/columns.cpp
    pipeline/prepare/report/columns.hpp
    pipeline/prepare/report/failure.cpp
    pipeline/prepare/report/failure.hpp
    pipeline/prepare/report/format.cpp
    pipeline/prepare/report/format.hpp
    pipeline/prepare/report/memory.cpp
    pipeline/prepare/report/memory.hpp
    pipeline/prepare/report/plan.cpp
    pipeline/prepare/report/plan.hpp
    pipeline/prepare/report/row.cpp
    pipeline/prepare/report/row.hpp
    pipeline/prepare/run.cpp
    pipeline/profile.cpp
    pipeline/recurrence.cpp
    pipeline/run.cpp
    suite/batch.cpp
    suite/bulk.cpp
    suite/capture.hpp
    suite/bench.hpp
    suite/bulk.hpp
    suite/core.hpp
    suite/output.cpp
    suite/output.hpp
    suite/parse.cpp
    suite/reference.cpp
    suite/reference.hpp
    suite/resident.cpp
    suite/sparse.cpp
    suite/warm.hpp
    virtual/backing.cpp
    virtual/crossover.hpp
    virtual/crossover/aggregate.cpp
    virtual/crossover/aggregate/local.hpp
    virtual/crossover/aggregate/parse.cpp
    virtual/crossover/aggregate/report.cpp
    virtual/crossover/evidence.cpp
    virtual/crossover/internal.hpp
    virtual/crossover/prepare.cpp
    virtual/crossover/report.cpp
    virtual/crossover/run.cpp
    virtual/crossover/schema.hpp
    virtual/graph_pointwise.hpp
    virtual/graph_pointwise/internal.hpp
    virtual/graph_pointwise/oracle.cpp
    virtual/graph_pointwise/report.cpp
    virtual/graph_pointwise/report/local.hpp
    virtual/graph_pointwise/report/csv.cpp
    virtual/graph_pointwise/report/diagnostic.cpp
    virtual/graph_pointwise/run.cpp
    virtual/graph_pointwise/run/evidence.cpp
    virtual/graph_pointwise/run/fixture.cpp
    virtual/graph_pointwise/run/io.cpp
    virtual/graph_pointwise/run/local.hpp
    virtual/graph_residency.hpp
    virtual/graph_residency/internal.hpp
    virtual/graph_residency/oracle.cpp
    virtual/graph_residency/report.cpp
    virtual/graph_residency/run.cpp
    virtual/graph_residency/run/evidence.cpp
    virtual/graph_residency/run/fixture.cpp
    virtual/graph_residency/run/timing.cpp
    virtual/model.cpp
    virtual/model.hpp
    virtual/oracle.cpp
    virtual/oracle.hpp
    virtual/product_route.cpp
    virtual/product_route.hpp
    virtual/product_route/observer.cpp
    virtual/product_route/report.cpp
    virtual/product_route/report.hpp
    virtual/product_route/validation.cpp
    virtual/report.cpp
    virtual/report.hpp
    virtual/residency.cpp
    virtual/residency/backing.cpp
    virtual/residency/backing.hpp
    virtual/window/local.hpp
    virtual/window/support.cpp
    virtual/window/prepare.cpp
    virtual/window/measure.cpp
    virtual/window/entry.cpp
    virtual/window/report.cpp
    virtual/window.hpp)
  target_compile_definitions(runD-compute-focus PRIVATE RUND_COMPUTE_FOCUS=1)
  target_include_directories(runD-compute-focus PRIVATE
    "${CMAKE_SOURCE_DIR}/accel/include"
    "${CMAKE_SOURCE_DIR}/node"
    "${CMAKE_SOURCE_DIR}/node/tests/contract/compute")
  target_link_libraries(runD-compute-focus PRIVATE
    node-closure-compute-execution)
  # The focused executable links the internal SCC archive directly. Native
  # framework/loader edges are private to the nested ACCEL_EXECUTION archive
  # and therefore do not survive the second static-archive boundary. Project
  # the same canonical focused backend row explicitly so a GPU measurement
  # cannot silently become the CPU-only closure and report AdapterUnavailable.
  get_target_property(compute_focus_native_dependencies
    node-closure-accel-execution RUND_NODE_EXTERNAL_LINKS)
  if(compute_focus_native_dependencies MATCHES "-NOTFOUND$")
    message(FATAL_ERROR
      "Compute focus measurement lost the Accel native link authority")
  endif()
  if(compute_focus_native_dependencies)
    # FindVulkan's imported target is directory-scoped to node/.  The focus
    # executable is a sibling target, so preserve the same loader authority by
    # resolving only that imported edge to FindVulkan's canonical cache path.
    set(compute_focus_resolved_native_dependencies)
    foreach(native_dependency IN LISTS compute_focus_native_dependencies)
      if(native_dependency STREQUAL "Vulkan::Vulkan")
        if(NOT Vulkan_LIBRARY)
          message(FATAL_ERROR
            "Compute focus measurement cannot resolve the Vulkan loader")
        endif()
        list(APPEND compute_focus_resolved_native_dependencies
          "${Vulkan_LIBRARY}")
      else()
        list(APPEND compute_focus_resolved_native_dependencies
          "${native_dependency}")
      endif()
    endforeach()
    target_link_libraries(runD-compute-focus PRIVATE
      ${compute_focus_resolved_native_dependencies})
  endif()
  target_compile_features(runD-compute-focus PRIVATE cxx_std_20)

  # Current-source natural route matrix.  This is a standalone diagnostic
  # surface: it is deliberately excluded from the default build and observes
  # the public VirtualPipeline without selecting a route on behalf of it.
  add_executable(
    runD-compute-route-matrix EXCLUDE_FROM_ALL
    virtual/route_matrix/main.cpp
    virtual/route_matrix/internal.hpp
    virtual/route_matrix/run.cpp
    virtual/route_matrix/run/evidence.cpp
    virtual/route_matrix/run/prepare.cpp
    virtual/route_matrix/run/sampling.cpp
    virtual/route_matrix/oracle.cpp
    virtual/route_matrix/oracle/backing.cpp
    virtual/route_matrix/oracle/cohort.cpp
    virtual/route_matrix/oracle/mismatch.cpp
    virtual/route_matrix/oracle/route.cpp
    virtual/route_matrix/oracle/support.cpp
    virtual/route_matrix/report/raw.cpp
    virtual/route_matrix/report/consensus.cpp
    virtual/route_matrix/observer/local.hpp
    virtual/route_matrix/observer/evidence.cpp
    virtual/route_matrix/observer/hooks.cpp
    virtual/route_matrix/observer/lifecycle.cpp
    virtual/backing.cpp
    virtual/backing.hpp)
  target_compile_definitions(runD-compute-route-matrix PRIVATE
    RUND_COMPUTE_FOCUS=1)
  target_include_directories(runD-compute-route-matrix PRIVATE
    "${CMAKE_SOURCE_DIR}/accel/include"
    "${CMAKE_SOURCE_DIR}/node"
    "${CMAKE_SOURCE_DIR}/node/tests/contract/compute")
  target_link_libraries(runD-compute-route-matrix PRIVATE
    node-closure-compute-execution)
  get_target_property(route_matrix_native_dependencies
    node-closure-accel-execution RUND_NODE_EXTERNAL_LINKS)
  if(route_matrix_native_dependencies MATCHES "-NOTFOUND$")
    message(FATAL_ERROR
      "Compute route matrix lost the Accel native link authority")
  endif()
  if(route_matrix_native_dependencies)
    set(route_matrix_resolved_native_dependencies)
    foreach(native_dependency IN LISTS route_matrix_native_dependencies)
      if(native_dependency STREQUAL "Vulkan::Vulkan")
        if(NOT Vulkan_LIBRARY)
          message(FATAL_ERROR
            "Compute route matrix cannot resolve the Vulkan loader")
        endif()
        list(APPEND route_matrix_resolved_native_dependencies
          "${Vulkan_LIBRARY}")
      else()
        list(APPEND route_matrix_resolved_native_dependencies
          "${native_dependency}")
      endif()
    endforeach()
    target_link_libraries(runD-compute-route-matrix PRIVATE
      ${route_matrix_resolved_native_dependencies})
  endif()
  target_compile_features(runD-compute-route-matrix PRIVATE cxx_std_20)

  # Explicit native scale contract. It is deliberately not registered with
  # the default CTest graph: encoding and executing 65,537 Metal ICB commands
  # is a release-evidence boundary, while accel.kernel-core retains the
  # portable allocation-free size-class contract.
  if(APPLE AND RUND_NODE_METAL_FRAMEWORK AND
      RUND_NODE_FOUNDATION_FRAMEWORK)
    enable_language(OBJCXX)
    set(metal_icb_boundary_source pipeline/metal/icb/boundary.mm)
    add_executable(runD-metal-icb-boundary EXCLUDE_FROM_ALL
      ${metal_icb_boundary_source})
    set_source_files_properties(${metal_icb_boundary_source} PROPERTIES
      LANGUAGE OBJCXX
      COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(runD-metal-icb-boundary PRIVATE
      RUND_NODE_HAVE_METAL_SDK=1)
    target_compile_features(runD-metal-icb-boundary PRIVATE cxx_std_20)
    target_include_directories(runD-metal-icb-boundary PRIVATE
      "${CMAKE_SOURCE_DIR}/node"
      "${CMAKE_SOURCE_DIR}/node/include"
      "${CMAKE_SOURCE_DIR}/accel/include")
    target_link_libraries(runD-metal-icb-boundary PRIVATE
      "${RUND_NODE_METAL_FRAMEWORK}"
      "${RUND_NODE_FOUNDATION_FRAMEWORK}")
  endif()
endif()
target_compile_features(runD-compute-measure PRIVATE cxx_std_20)
