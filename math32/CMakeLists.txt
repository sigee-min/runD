cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rund_math32 LANGUAGES CXX)
  enable_testing()
endif()

option(
  RUND_ENABLE_MATH32_CONTRACT_TESTS
  "Register math32 contract CTests and create the test executable"
  OFF)

add_library(math32 INTERFACE)
target_include_directories(math32 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(math32 INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(math32 INTERFACE -fno-fast-math -ffp-contract=off)
elseif(MSVC)
  target_compile_options(math32 INTERFACE /fp:strict)
endif()

if(RUND_ENABLE_MATH32_CONTRACT_TESTS)
  if(NOT TARGET rund-test-assertion)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/root/test/assert.cmake")
    rund_test_assertion()
  endif()
  include(cmake/tests/sources.cmake)
  add_executable(test_math32 ${RUND_MATH32_CONTRACT_TEST_SOURCES})
  set_target_properties(test_math32 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(test_math32 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tools")
  target_link_libraries(test_math32 PRIVATE math32 rund-test-assertion)
  add_test(NAME math32.contract COMMAND test_math32)
  set_tests_properties(math32.contract PROPERTIES
    LABELS "rund_math32;rund_contract;rund_owner_local"
    TIMEOUT 900)
  if(COMMAND rund_test_route)
    rund_test_route(math32.contract TARGETS test_math32)
  endif()
endif()
