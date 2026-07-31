set_property(GLOBAL PROPERTY RUND_TEST_ROUTE_NAMES "")
set_property(GLOBAL PROPERTY RUND_TEST_ROUTE_ROWS "")
set_property(GLOBAL PROPERTY RUND_ACCEL_TEST_NAMES "")

# CTest RESOURCE_LOCK serializes tests only inside one build tree. Real
# adapter users also need one repository-wide lock across dev, focus,
# sanitizer, and release trees. Keep that command shape under one owner and
# wrap only the tests that actually submit accelerator work.
function(rund_accel_test_command out name)
  if(NOT name MATCHES "^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    message(FATAL_ERROR "Invalid accelerator CTest name: ${name}")
  endif()
  if(NOT ARGN)
    message(FATAL_ERROR "Accelerator test command cannot be empty")
  endif()
  get_property(accel_names GLOBAL PROPERTY RUND_ACCEL_TEST_NAMES)
  list(FIND accel_names "${name}" duplicate)
  if(NOT duplicate EQUAL -1)
    message(FATAL_ERROR "Duplicate accelerator test command: ${name}")
  endif()
  set_property(GLOBAL APPEND PROPERTY RUND_ACCEL_TEST_NAMES "${name}")
  set(${out}
    sh
    "${CMAKE_SOURCE_DIR}/tools/internal/lock/accel"
    --wait
    sh
    "${CMAKE_SOURCE_DIR}/tools/internal/process/run"
    900
    "${name}"
    ${ARGN}
    PARENT_SCOPE)
endfunction()

function(rund_test_route name)
  cmake_parse_arguments(RUND_ROUTE "NO_BUILD_TARGETS" "" "TARGETS" ${ARGN})
  if(RUND_ROUTE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown CTest route arguments for ${name}: ${RUND_ROUTE_UNPARSED_ARGUMENTS}")
  endif()
  if(RUND_ROUTE_NO_BUILD_TARGETS AND RUND_ROUTE_TARGETS)
    message(FATAL_ERROR
      "CTest route cannot mix TARGETS and NO_BUILD_TARGETS: ${name}")
  endif()
  if(NOT RUND_ROUTE_NO_BUILD_TARGETS AND NOT RUND_ROUTE_TARGETS)
    message(FATAL_ERROR
      "CTest route must declare TARGETS or NO_BUILD_TARGETS: ${name}")
  endif()
  if(NOT name MATCHES "^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    message(FATAL_ERROR "Malformed CTest route name: ${name}")
  endif()
  if(NOT TEST "${name}")
    message(FATAL_ERROR "CTest route has no registered test: ${name}")
  endif()

  get_property(names GLOBAL PROPERTY RUND_TEST_ROUTE_NAMES)
  list(FIND names "${name}" duplicate)
  if(NOT duplicate EQUAL -1)
    message(FATAL_ERROR "Duplicate CTest route: ${name}")
  endif()

  foreach(target IN LISTS RUND_ROUTE_TARGETS)
    if(NOT target MATCHES "^[A-Za-z0-9_.+-]+$" OR NOT TARGET "${target}")
      message(FATAL_ERROR "Invalid CTest build target ${target}: ${name}")
    endif()
  endforeach()
  get_property(RUND_ROUTE_RESOURCES TEST "${name}" PROPERTY RESOURCE_LOCK)
  foreach(resource IN LISTS RUND_ROUTE_RESOURCES)
    if(NOT resource MATCHES "^[A-Za-z0-9_.-]+$")
      message(FATAL_ERROR "Invalid CTest resource ${resource}: ${name}")
    endif()
  endforeach()
  get_property(timeout TEST "${name}" PROPERTY TIMEOUT)
  if(NOT timeout MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "CTest route has no finite timeout: ${name}")
  endif()

  get_property(accel_names GLOBAL PROPERTY RUND_ACCEL_TEST_NAMES)
  list(FIND accel_names "${name}" accel_command)
  list(FIND RUND_ROUTE_RESOURCES "rund_accel" accel_resource)
  if(NOT accel_resource EQUAL -1)
    if(accel_command EQUAL -1)
      message(FATAL_ERROR
        "Accelerator CTest route has no repository runner: ${name}")
    endif()
    if(NOT timeout STREQUAL "1800")
      message(FATAL_ERROR
        "Accelerator CTest route must reserve 1800 seconds: ${name}")
    endif()
    set(runner accel)
  else()
    if(NOT accel_command EQUAL -1)
      message(FATAL_ERROR
        "Repository accelerator runner has no CTest resource: ${name}")
    endif()
    set(runner direct)
  endif()

  if(RUND_ROUTE_TARGETS)
    list(REMOVE_DUPLICATES RUND_ROUTE_TARGETS)
    list(SORT RUND_ROUTE_TARGETS)
    list(JOIN RUND_ROUTE_TARGETS "," targets)
  else()
    set(targets "-")
  endif()
  if(RUND_ROUTE_RESOURCES)
    list(REMOVE_DUPLICATES RUND_ROUTE_RESOURCES)
    list(SORT RUND_ROUTE_RESOURCES)
    list(JOIN RUND_ROUTE_RESOURCES "," resources)
  else()
    set(resources "-")
  endif()

  set_property(GLOBAL APPEND PROPERTY RUND_TEST_ROUTE_NAMES "${name}")
  set_property(GLOBAL APPEND PROPERTY RUND_TEST_ROUTE_ROWS
    "route\t${name}\t${targets}\t${resources}\t${timeout}\t${runner}")
endfunction()

function(rund_test_routes_write)
  get_property(written GLOBAL PROPERTY RUND_TEST_ROUTES_WRITTEN)
  if(written)
    message(FATAL_ERROR "CTest route index generated more than once")
  endif()
  set_property(GLOBAL PROPERTY RUND_TEST_ROUTES_WRITTEN TRUE)

  get_property(rows GLOBAL PROPERTY RUND_TEST_ROUTE_ROWS)
  if(NOT rows)
    message(FATAL_ERROR "CTest route index has no rows")
  endif()
  get_property(names GLOBAL PROPERTY RUND_TEST_ROUTE_NAMES)
  get_property(accel_names GLOBAL PROPERTY RUND_ACCEL_TEST_NAMES)
  foreach(name IN LISTS accel_names)
    list(FIND names "${name}" route)
    if(route EQUAL -1)
      message(FATAL_ERROR
        "Accelerator test command has no CTest route: ${name}")
    endif()
  endforeach()
  list(SORT rows)
  list(JOIN rows "\n" routes)
  set(body "routes\n${routes}\n")
  string(SHA256 seal "${body}")
  set(content "${body}seal\t${seal}\n")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/rund-test-routes.tsv"
    CONTENT "${content}")
endfunction()
