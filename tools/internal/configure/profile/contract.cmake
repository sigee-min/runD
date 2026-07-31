cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROOT OR ROOT STREQUAL "")
  message(FATAL_ERROR "ROOT is required")
endif()
if(NOT DEFINED BUILD OR BUILD STREQUAL "")
  message(FATAL_ERROR "BUILD is required")
endif()
get_filename_component(ROOT "${ROOT}" ABSOLUTE)
get_filename_component(BUILD "${BUILD}" ABSOLUTE)

set(fixture "${BUILD}/configure-contract-profile-fixture")
set(fake_bin "${fixture}/bin")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${fake_bin}")
file(WRITE "${fake_bin}/cmake" [=[#!/bin/sh
set -eu
: "${RUND_CONFIGURE_CAPTURE:?}"
: > "$RUND_CONFIGURE_CAPTURE"
for argument do
  printf '%s\n' "$argument" >> "$RUND_CONFIGURE_CAPTURE"
done
]=])
file(CHMOD "${fake_bin}/cmake"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

function(capture_profile profile build_type focus capture)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PATH=${fake_bin}:$ENV{PATH}"
      "RUND_CONFIGURE_CAPTURE=${capture}"
      sh "${ROOT}/tools/internal/configure/contracts"
      "${ROOT}" "${fixture}/${profile}" "${profile}" "${build_type}"
      "${focus}"
      -DRUND_ENABLE_VULKAN=OFF
      -DRUND_FORCE_UNAVAILABLE_PLATFORM=ON
      -DRUND_ENABLE_PACKAGE_RELEASE_TESTS=INHERITED
      -DRUND_ENABLE_MATH32_CONTRACT_TESTS=INHERITED
      -DRUND_ENABLE_MATH64_CONTRACT_TESTS=INHERITED
      -DRUND_ENABLE_CLUSTER_CONTRACT_TESTS=INHERITED
      -DRUND_TEST_KERNEL=OFF
      -DRUND_TEST_NODE=OFF
      -DRUND_TEST_ACCEL=OFF
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR
      "configure-contracts ${profile} capture failed: ${detail}")
  endif()
endfunction()

function(require_last capture name expected)
  file(STRINGS "${capture}" arguments)
  set(found FALSE)
  set(value "")
  foreach(argument IN LISTS arguments)
    if(argument MATCHES "^-D${name}(:[^=]+)?=(.*)$")
      set(found TRUE)
      set(value "${CMAKE_MATCH_2}")
    endif()
  endforeach()
  if(NOT found)
    message(FATAL_ERROR "configure profile omitted ${name}")
  endif()
  if(NOT value STREQUAL expected)
    message(FATAL_ERROR
      "configure profile left ${name}=${value}; expected ${expected}")
  endif()
endfunction()

function(require_ninja capture)
  file(READ "${capture}" arguments)
  set(pair "-G\nNinja\n")
  string(FIND "${arguments}" "${pair}" first)
  if(first EQUAL -1)
    message(FATAL_ERROR "configure profile did not select Ninja")
  endif()
  string(LENGTH "${pair}" pair_length)
  math(EXPR suffix_start "${first} + ${pair_length}")
  string(SUBSTRING "${arguments}" ${suffix_start} -1 suffix)
  string(FIND "${suffix}" "${pair}" duplicate)
  if(NOT duplicate EQUAL -1)
    message(FATAL_ERROR "configure profile selected Ninja more than once")
  endif()
endfunction()

set(local_capture "${fixture}/local.tsv")
capture_profile(local Debug - "${local_capture}")
require_ninja("${local_capture}")
require_last("${local_capture}" RUND_STRICT_WARNINGS ON)
require_last("${local_capture}" RUND_ENABLE_VULKAN ON)
require_last("${local_capture}" RUND_FORCE_UNAVAILABLE_PLATFORM OFF)
require_last("${local_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS OFF)
require_last("${local_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS ON)
require_last("${local_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS ON)
require_last("${local_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS ON)
require_last("${local_capture}" RUND_TEST_KERNEL ON)
require_last("${local_capture}" RUND_TEST_NODE ON)
require_last("${local_capture}" RUND_TEST_ACCEL ON)
require_last("${local_capture}" CMAKE_MAKE_PROGRAM
  "${ROOT}/tools/internal/state/ninja")

set(release_capture "${fixture}/release.tsv")
capture_profile(release Release - "${release_capture}")
require_ninja("${release_capture}")
require_last("${release_capture}" RUND_STRICT_WARNINGS ON)
require_last("${release_capture}" RUND_ENABLE_VULKAN ON)
require_last("${release_capture}" RUND_FORCE_UNAVAILABLE_PLATFORM OFF)
require_last("${release_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS ON)
require_last("${release_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS ON)
require_last("${release_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS ON)
require_last("${release_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS ON)
require_last("${release_capture}" RUND_TEST_KERNEL ON)
require_last("${release_capture}" RUND_TEST_NODE ON)
require_last("${release_capture}" RUND_TEST_ACCEL ON)
require_last("${release_capture}" CMAKE_MAKE_PROGRAM
  "${ROOT}/tools/internal/state/ninja")

set(sanitize_capture "${fixture}/sanitize.tsv")
capture_profile(sanitize Debug - "${sanitize_capture}")
require_ninja("${sanitize_capture}")
require_last("${sanitize_capture}" RUND_STRICT_WARNINGS ON)
require_last("${sanitize_capture}" RUND_ENABLE_VULKAN ON)
require_last("${sanitize_capture}" RUND_FORCE_UNAVAILABLE_PLATFORM OFF)
require_last("${sanitize_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS OFF)
require_last("${sanitize_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS ON)
require_last("${sanitize_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS ON)
require_last("${sanitize_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS ON)
require_last("${sanitize_capture}" RUND_TEST_KERNEL ON)
require_last("${sanitize_capture}" RUND_TEST_NODE ON)
require_last("${sanitize_capture}" RUND_TEST_ACCEL ON)
require_last("${sanitize_capture}" CMAKE_MAKE_PROGRAM
  "${ROOT}/tools/internal/state/ninja")

set(focused_node_capture "${fixture}/focused-node.tsv")
capture_profile(local Debug runtime.memory "${focused_node_capture}")
require_last("${focused_node_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS OFF)
require_last("${focused_node_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS OFF)
require_last("${focused_node_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS OFF)
require_last("${focused_node_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS OFF)
require_last("${focused_node_capture}" RUND_TEST_KERNEL OFF)
require_last("${focused_node_capture}" RUND_TEST_NODE ON)
require_last("${focused_node_capture}" RUND_TEST_ACCEL ON)

set(focused_accel_capture "${fixture}/focused-accel.tsv")
capture_profile(local Debug accel.surface "${focused_accel_capture}")
require_last("${focused_accel_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS OFF)
require_last("${focused_accel_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS OFF)
require_last("${focused_accel_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS OFF)
require_last("${focused_accel_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS OFF)
require_last("${focused_accel_capture}" RUND_TEST_KERNEL OFF)
require_last("${focused_accel_capture}" RUND_TEST_NODE ON)
require_last("${focused_accel_capture}" RUND_TEST_ACCEL ON)

function(capture_platform vulkan capture)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PATH=${fake_bin}:$ENV{PATH}"
      "RUND_CONFIGURE_CAPTURE=${capture}"
      sh "${ROOT}/tools/internal/configure/contracts"
      "${ROOT}" "${fixture}/platform-${vulkan}" platform Debug
      runtime.platform-adapter "${vulkan}"
      -DRUND_ENABLE_VULKAN=INHERITED
      -DRUND_FORCE_UNAVAILABLE_PLATFORM=OFF
      -DRUND_ENABLE_PACKAGE_RELEASE_TESTS=INHERITED
      -DRUND_ENABLE_MATH32_CONTRACT_TESTS=INHERITED
      -DRUND_ENABLE_MATH64_CONTRACT_TESTS=INHERITED
      -DRUND_ENABLE_CLUSTER_CONTRACT_TESTS=INHERITED
      -DRUND_TEST_KERNEL=ON
      -DRUND_TEST_NODE=OFF
      -DRUND_TEST_ACCEL=ON
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR
      "configure-contracts platform ${vulkan} capture failed: ${detail}")
  endif()
endfunction()

foreach(platform_vulkan IN ITEMS OFF ON)
  set(platform_capture "${fixture}/platform-${platform_vulkan}.tsv")
  capture_platform("${platform_vulkan}" "${platform_capture}")
  require_ninja("${platform_capture}")
  require_last("${platform_capture}" RUND_STRICT_WARNINGS ON)
  require_last("${platform_capture}" RUND_ENABLE_VULKAN
    "${platform_vulkan}")
  require_last("${platform_capture}" RUND_FORCE_UNAVAILABLE_PLATFORM ON)
  require_last("${platform_capture}" RUND_ENABLE_PACKAGE_RELEASE_TESTS OFF)
  require_last("${platform_capture}" RUND_ENABLE_MATH32_CONTRACT_TESTS OFF)
  require_last("${platform_capture}" RUND_ENABLE_MATH64_CONTRACT_TESTS OFF)
  require_last("${platform_capture}" RUND_ENABLE_CLUSTER_CONTRACT_TESTS OFF)
  require_last("${platform_capture}" RUND_TEST_KERNEL OFF)
  require_last("${platform_capture}" RUND_TEST_NODE ON)
  require_last("${platform_capture}" RUND_TEST_ACCEL OFF)
endforeach()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${fake_bin}"
    /bin/sh "${ROOT}/tools/internal/configure/contracts"
    "${ROOT}" "${fixture}/missing-ninja" local Debug -
  RESULT_VARIABLE missing_ninja_result
  OUTPUT_VARIABLE missing_ninja_output
  ERROR_VARIABLE missing_ninja_error)
if(NOT missing_ninja_result EQUAL 1 OR
   NOT missing_ninja_error STREQUAL "contract verification requires Ninja\n")
  message(FATAL_ERROR
    "configure profile did not fail closed without Ninja: "
    "${missing_ninja_output}${missing_ninja_error}")
endif()

set(non_ninja "${fixture}/non-ninja")
file(MAKE_DIRECTORY "${non_ninja}")
file(WRITE "${non_ninja}/CMakeCache.txt"
  "CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${fake_bin}:$ENV{PATH}"
    sh "${ROOT}/tools/internal/configure/contracts"
    "${ROOT}" "${non_ninja}" local Debug -
  RESULT_VARIABLE non_ninja_result
  OUTPUT_VARIABLE non_ninja_output
  ERROR_VARIABLE non_ninja_error)
set(non_ninja_expected
  "contract build tree is not configured with Ninja: ${non_ninja}\n")
if(NOT non_ninja_result EQUAL 1 OR
   NOT non_ninja_error STREQUAL non_ninja_expected)
  message(FATAL_ERROR
    "configure profile admitted a non-Ninja tree: "
    "${non_ninja_output}${non_ninja_error}")
endif()

file(REMOVE_RECURSE "${fixture}")
