if(NOT DEFINED RUND_EXPECTED_PACKAGE_DIR OR
   RUND_EXPECTED_PACKAGE_DIR STREQUAL "")
  message(FATAL_ERROR
    "installed measurement requires the freshly installed package directory")
endif()
if(NOT DEFINED RUND_EXPECTED_SDK_VERSION OR
   RUND_EXPECTED_SDK_VERSION STREQUAL "")
  message(FATAL_ERROR
    "installed measurement requires the expected SDK version")
endif()

find_package(runD ${RUND_EXPECTED_SDK_VERSION} EXACT CONFIG REQUIRED)

file(REAL_PATH "${RUND_EXPECTED_PACKAGE_DIR}" expected_runD_dir)
file(REAL_PATH "${runD_DIR}" resolved_runD_dir)
if(NOT resolved_runD_dir STREQUAL expected_runD_dir)
  message(FATAL_ERROR
    "measurement resolved runD outside the fresh install prefix\n"
    "expected: ${expected_runD_dir}\n"
    "actual:   ${resolved_runD_dir}")
endif()

if(NOT DEFINED runD_VERSION OR
   NOT runD_VERSION STREQUAL RUND_EXPECTED_SDK_VERSION)
  message(FATAL_ERROR
    "measurement resolved a non-current SDK version: ${runD_VERSION}")
endif()

if(NOT TARGET runD::sdk)
  message(FATAL_ERROR "installed package is missing runD::sdk")
endif()
