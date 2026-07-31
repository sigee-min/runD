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
    pipeline/profile.cpp
    pipeline/recurrence.cpp
    pipeline/run.cpp
    suite/batch.cpp
    suite/bulk.cpp
    suite/resident.cpp
    suite/sparse.cpp)
  target_compile_definitions(runD-compute-focus PRIVATE RUND_COMPUTE_FOCUS=1)
  target_link_libraries(runD-compute-focus PRIVATE
    node-closure-compute-execution)
  target_compile_features(runD-compute-focus PRIVATE cxx_std_20)
endif()
target_compile_features(runD-compute-measure PRIVATE cxx_std_20)
