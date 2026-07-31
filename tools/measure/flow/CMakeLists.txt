cmake_minimum_required(VERSION 3.25)
project(runD_flow_measure LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../package/installed.cmake")

add_executable(runD-flow-measure runtime.cpp)
target_compile_features(runD-flow-measure PRIVATE cxx_std_20)
target_link_libraries(runD-flow-measure PRIVATE runD::sdk)
