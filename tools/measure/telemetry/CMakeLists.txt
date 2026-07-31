cmake_minimum_required(VERSION 3.20)
project(runD_telemetry_measure LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../package/installed.cmake")

add_executable(
  runD-telemetry-measure
  main.cpp
  suite/allocation.cpp
  suite/design.cpp
  suite/print.cpp
  suite/run.cpp
  suite/work.cpp)
target_compile_features(runD-telemetry-measure PRIVATE cxx_std_20)
target_link_libraries(runD-telemetry-measure PRIVATE runD::sdk)
