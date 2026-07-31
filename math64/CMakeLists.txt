cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rund_math64 LANGUAGES CXX)
  enable_testing()
endif()

option(
  RUND_ENABLE_MATH64_CONTRACT_TESTS
  "Register math64 contract CTests and create the test executable"
  OFF)

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if !defined(__SIZEOF_INT128__)
#error no deterministic 128-bit integer support
#endif
int main() { __int128_t value = 1; return static_cast<int>(value - 1); }
" RUND_MATH64_HAS_INT128)
if(NOT RUND_MATH64_HAS_INT128)
  message(FATAL_ERROR "math64 requires deterministic __int128_t support")
endif()

add_library(math64 INTERFACE)
target_include_directories(math64 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(math64 INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(math64 INTERFACE -fno-fast-math -ffp-contract=off)
elseif(MSVC)
  target_compile_options(math64 INTERFACE /fp:strict)
endif()

if(RUND_ENABLE_MATH64_CONTRACT_TESTS)
  if(NOT TARGET rund-test-assertion)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/root/test/assert.cmake")
    rund_test_assertion()
  endif()
  include(cmake/tests/sources.cmake)
  add_executable(test_math64 ${RUND_MATH64_CONTRACT_TEST_SOURCES})
  set_target_properties(test_math64 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(test_math64 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tools")
  target_link_libraries(test_math64 PRIVATE math64 rund-test-assertion)
  add_test(NAME math64.contract COMMAND test_math64)
  set_tests_properties(math64.contract PROPERTIES
    LABELS "rund_math64;rund_contract;rund_owner_local"
    TIMEOUT 900)
  if(COMMAND rund_test_route)
    rund_test_route(math64.contract TARGETS test_math64)
  endif()
endif()
