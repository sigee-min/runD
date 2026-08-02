if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(runD_compute_measure LANGUAGES CXX)
  include("${CMAKE_CURRENT_LIST_DIR}/../package/installed.cmake")
  add_executable(
    runD-compute-measure
    main.cpp
    suite/batch.cpp
    suite/family.cpp
    suite/host.cpp
    suite/mixed.cpp
    suite/resident.cpp
    suite/sparse.cpp)
  target_link_libraries(runD-compute-measure PRIVATE runD::sdk)
else()
  add_executable(
    runD-compute-measure EXCLUDE_FROM_ALL
    main.cpp
    suite/batch.cpp
    suite/family.cpp
    suite/host.cpp
    suite/mixed.cpp
    suite/resident.cpp
    suite/sparse.cpp)
  target_link_libraries(runD-compute-measure PRIVATE node)

  add_executable(
    runD-compute-focus EXCLUDE_FROM_ALL
    main.cpp
    "${CMAKE_SOURCE_DIR}/node/tests/contract/compute/allocation.cpp"
    pipeline/checkpoint.cpp
    pipeline/nested.cpp
    pipeline/prepare_memory.cpp
    pipeline/profile.cpp
    pipeline/recurrence.cpp
    pipeline/run.cpp
    suite/batch.cpp
    suite/bulk.cpp
    suite/resident.cpp
    suite/sparse.cpp)
  target_compile_definitions(runD-compute-focus PRIVATE RUND_COMPUTE_FOCUS=1)
  target_include_directories(runD-compute-focus PRIVATE
    "${CMAKE_SOURCE_DIR}/accel/include"
    "${CMAKE_SOURCE_DIR}/node"
    "${CMAKE_SOURCE_DIR}/node/tests/contract/compute")
  target_link_libraries(runD-compute-focus PRIVATE
    node-closure-compute-execution)
  target_compile_features(runD-compute-focus PRIVATE cxx_std_20)

  # Explicit native scale contract. It is deliberately not registered with
  # the default CTest graph: encoding and executing 65,537 Metal ICB commands
  # is a release-evidence boundary, while accel.kernel-core retains the
  # portable allocation-free size-class contract.
  if(APPLE AND RUND_NODE_METAL_FRAMEWORK AND
      RUND_NODE_FOUNDATION_FRAMEWORK)
    enable_language(OBJCXX)
    set(metal_icb_boundary_source pipeline/metal_icb_boundary.mm)
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
      "${CMAKE_SOURCE_DIR}/accel/include")
    target_link_libraries(runD-metal-icb-boundary PRIVATE
      "${RUND_NODE_METAL_FRAMEWORK}"
      "${RUND_NODE_FOUNDATION_FRAMEWORK}")
  endif()
endif()
target_compile_features(runD-compute-measure PRIVATE cxx_std_20)
