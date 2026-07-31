cmake_minimum_required(VERSION 3.25)
project(runD_graph_services_measure LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../package/installed.cmake")

add_executable(runD-graph-services-measure main.cpp)
target_compile_features(runD-graph-services-measure PRIVATE cxx_std_20)
target_link_libraries(runD-graph-services-measure PRIVATE runD::sdk)
