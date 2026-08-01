cmake_minimum_required(VERSION 3.20)
project(runD_target_collision_consumer LANGUAGES NONE)

if(NOT DEFINED RUND_EXPECTED_SDK_VERSION OR
   RUND_EXPECTED_SDK_VERSION STREQUAL "")
  message(FATAL_ERROR "collision probe requires the expected SDK version")
endif()

# Model a target already imported by a different package prefix. The installed
# runD package must reject it instead of treating its public name as proof of
# ownership.
add_library(runD::sdk INTERFACE IMPORTED)
set_target_properties(runD::sdk PROPERTIES
  RUND_PACKAGE_PREFIX "/foreign/runD-prefix")

find_package(runD ${RUND_EXPECTED_SDK_VERSION} EXACT CONFIG REQUIRED)
message(FATAL_ERROR "runD unexpectedly accepted a foreign runD::sdk target")
