# A configured development tree may be reused only after the generator checks
# every CMake and glob dependency.  This target has no command or output; its
# sole purpose is to provide that portable graph-synchronization boundary.
add_custom_target(rund-contract-graph)

# Root-owned CTest registration.
add_test(
  NAME tools.source-manifest
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/source/manifest/contract.cmake")
set_tests_properties(tools.source-manifest PROPERTIES
  LABELS "rund_tools"
  TIMEOUT 30)
rund_test_route(tools.source-manifest NO_BUILD_TARGETS)

add_test(
  NAME tools.public-commands
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/command/public/contract.cmake")
set_tests_properties(tools.public-commands PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.public-commands NO_BUILD_TARGETS)

add_test(
  NAME tools.lock
  COMMAND sh "${CMAKE_SOURCE_DIR}/tools/internal/lock/contract"
          "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/lock-contract")
set_tests_properties(tools.lock PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.lock NO_BUILD_TARGETS)

add_test(
  NAME tools.process
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -D "BUILD=${CMAKE_BINARY_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/process/contract.cmake")
set_tests_properties(tools.process PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.process NO_BUILD_TARGETS)

add_test(
  NAME tools.measure
  COMMAND sh "${CMAKE_SOURCE_DIR}/tools/internal/measure/contract"
          "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/measure-contract")
set_tests_properties(tools.measure PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.measure NO_BUILD_TARGETS)

if(TARGET runD-compute-focus)
  add_test(
    NAME tools.compute-focus
    COMMAND $<TARGET_FILE:runD-compute-focus> --sort cpu)
  set_tests_properties(tools.compute-focus PROPERTIES
    LABELS "rund_tools;rund_harness"
    TIMEOUT 60)
  rund_test_route(tools.compute-focus TARGETS runD-compute-focus)
endif()

find_program(rund_build_measure_python NAMES python3 REQUIRED)
add_test(
  NAME tools.build-measure
  COMMAND "${rund_build_measure_python}"
          "${CMAKE_SOURCE_DIR}/tools/measure/build/contract.py")
set_tests_properties(tools.build-measure PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.build-measure NO_BUILD_TARGETS)

add_test(
  NAME tools.build-ctest-selection
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -D "BUILD=${CMAKE_BINARY_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/ctest/selection/contract.cmake")
set_tests_properties(tools.build-ctest-selection PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.build-ctest-selection NO_BUILD_TARGETS)

add_test(
  NAME tools.case-query
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -D "BUILD=${CMAKE_BINARY_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/case/contract.cmake")
set_tests_properties(tools.case-query PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.case-query NO_BUILD_TARGETS)

add_test(
  NAME tools.case-execution
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -D "BUILD=${CMAKE_BINARY_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/case/execution/contract.cmake")
set_tests_properties(tools.case-execution PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.case-execution NO_BUILD_TARGETS)

add_test(
  NAME tools.configure-contract-profile
  COMMAND "${CMAKE_COMMAND}"
          -D "ROOT=${CMAKE_SOURCE_DIR}"
          -D "BUILD=${CMAKE_BINARY_DIR}"
          -P "${CMAKE_SOURCE_DIR}/tools/internal/configure/profile/contract.cmake")
set_tests_properties(tools.configure-contract-profile PROPERTIES
  LABELS "rund_tools;rund_harness"
  TIMEOUT 30)
rund_test_route(tools.configure-contract-profile NO_BUILD_TARGETS)

add_test(
  NAME tools.evidence-status
  COMMAND ${CMAKE_COMMAND}
    -D "ROOT=${CMAKE_SOURCE_DIR}"
    -D "BUILD=${CMAKE_BINARY_DIR}"
    -P "${CMAKE_SOURCE_DIR}/tools/internal/evidence/status/contract.cmake")
set_tests_properties(tools.evidence-status PROPERTIES
  LABELS "rund_tools;rund_contract"
  TIMEOUT 30)
rund_test_route(tools.evidence-status NO_BUILD_TARGETS)

if(RUND_ENABLE_PACKAGE_RELEASE_TESTS)
  add_test(
    NAME package.release-workflow
    COMMAND "${CMAKE_COMMAND}"
            -D "ROOT=${CMAKE_SOURCE_DIR}"
            -D "BUILD=${CMAKE_BINARY_DIR}"
            -P "${CMAKE_SOURCE_DIR}/package/cmake/release/workflow.cmake")
  set_tests_properties(package.release-workflow PROPERTIES
    LABELS "rund_package;rund_release"
    TIMEOUT 30)
  rund_test_route(package.release-workflow NO_BUILD_TARGETS)

  add_test(
    NAME package.consumer
    COMMAND "${CMAKE_COMMAND}"
            -D "ROOT=${CMAKE_SOURCE_DIR}"
            -D "BUILD_DIR=${CMAKE_BINARY_DIR}"
            -D "PREFIX=${RUND_PACKAGE_INSTALL_PREFIX}"
            -D "EXPECTED_SDK_VERSION=${RUND_PACKAGE_VERSION}"
            -P "${CMAKE_SOURCE_DIR}/package/cmake/consumer.cmake")
  set_tests_properties(package.consumer PROPERTIES
    LABELS "rund_package;rund_release;rund_smoke"
    TIMEOUT 1800)
  rund_test_route(package.consumer TARGETS runD_package_install)
endif()
