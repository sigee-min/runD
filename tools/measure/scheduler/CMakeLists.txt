cmake_minimum_required(VERSION 3.20)
project(runD_scheduler_measure LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../package/installed.cmake")

add_executable(
  runD-scheduler-measure
  main.cpp
  suite/base.cpp
  suite/latency.cpp
  suite/memory.cpp
  suite/print.cpp
  suite/scale.cpp
  suite/task.cpp)
target_compile_features(runD-scheduler-measure PRIVATE cxx_std_20)
target_link_libraries(runD-scheduler-measure PRIVATE runD::sdk)
