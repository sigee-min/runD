include(cmake/cluster/sources.cmake)
option(
  RUND_ENABLE_CLUSTER_CONTRACT_TESTS
  "Register cluster contract CTests and create the test executable"
  OFF)

add_library(cluster-object OBJECT ${CLUSTER_SOURCES})
target_include_directories(cluster-object PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/node/include")
target_compile_features(cluster-object PRIVATE cxx_std_20)

add_library(cluster STATIC)
target_sources(cluster PRIVATE $<TARGET_OBJECTS:cluster-object>)
target_include_directories(cluster
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/node/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cluster PUBLIC cxx_std_20)

# Cluster consumes declaration-only evidence. A link edge here would turn that
# include dependency into an unrelated Node/Math build closure.
foreach(cluster_target IN ITEMS cluster-object cluster)
  get_target_property(cluster_links ${cluster_target} LINK_LIBRARIES)
  if(cluster_links)
    message(FATAL_ERROR
      "${cluster_target} must remain free of implementation link dependencies")
  endif()
endforeach()

if(RUND_ENABLE_CLUSTER_CONTRACT_TESTS)
  if(NOT TARGET rund-test-assertion)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/root/test/assert.cmake")
    rund_test_assertion()
  endif()
  include(cmake/tests/sources.cmake)
  add_executable(test_cluster ${CLUSTER_CONTRACT_TEST_SOURCES})
  target_sources(test_cluster PRIVATE $<TARGET_OBJECTS:cluster-object>)
  set_target_properties(test_cluster PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(test_cluster PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    "${CMAKE_SOURCE_DIR}/node/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tools"
  )
  target_link_libraries(test_cluster PRIVATE rund-test-assertion)
  add_test(NAME cluster.contract COMMAND $<TARGET_FILE:test_cluster>)
  set_tests_properties(cluster.contract PROPERTIES
    LABELS "rund_cluster;rund_contract;rund_owner_local"
    TIMEOUT 900)
  rund_test_route(cluster.contract TARGETS test_cluster)
endif()
