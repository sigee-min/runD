cmake_minimum_required(VERSION 3.20)
project(runD_rejected_component_consumer LANGUAGES NONE)

if(NOT DEFINED RUND_EXPECTED_PACKAGE_DIR OR
   RUND_EXPECTED_PACKAGE_DIR STREQUAL "")
  message(FATAL_ERROR
    "component probe requires the freshly installed package directory")
endif()

find_package(runD 1.0.1 EXACT CONFIG QUIET COMPONENTS unsupported)
if(runD_FOUND)
  message(FATAL_ERROR "runD unexpectedly accepted an unsupported component")
endif()
if(TARGET runD::sdk)
  message(FATAL_ERROR
    "runD imported runD::sdk before rejecting an unsupported component")
endif()

# The outer package validator requires this exact marker.  Making the isolated
# configure fail prevents a rejected lookup from being mistaken for a positive
# consumer configuration.
message(FATAL_ERROR "runD expected component rejection verified")
