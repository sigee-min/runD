cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are disabled. Use an out-of-tree build directory.")
endif()

project(runD LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(cmake/root/compile.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

option(
  RUND_ENABLE_PACKAGE_RELEASE_TESTS
  "Register package artifact and staged SDK consumer release-gate CTests"
  OFF)
option(
  RUND_TEST_NODE
  "Register node contract CTests"
  OFF)
option(
  RUND_TEST_ACCEL
  "Register Accel CTests"
  OFF)
option(
  RUND_FORCE_UNAVAILABLE_PLATFORM
  "Select the explicit unavailable reactor and native IO/network adapter"
  OFF)
option(
  RUND_ENABLE_VULKAN
  "Discover and enable the Vulkan SDK and external shader tools"
  ON)
include(cmake/root/test/routes.cmake)
include(cmake/root/test/assert.cmake)
rund_test_assertion()
include(cmake/root/subsystems.cmake)
add_subdirectory(tools/measure/compute tools/measure/compute)
include(cmake/root/package.cmake)
include(cmake/root/tests.cmake)
rund_test_routes_write()
