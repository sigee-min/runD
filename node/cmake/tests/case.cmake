function(rund_node_case name)
  set(options)
  set(one_value_args RESOURCE_LOCK TIMEOUT)
  set(multi_value_args BUILD_TARGETS COMMAND LABELS)
  cmake_parse_arguments(RUND_NODE_CASE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN})

  if(RUND_NODE_CASE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown Node contract arguments for ${name}: ${RUND_NODE_CASE_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT RUND_NODE_CASE_COMMAND)
    message(FATAL_ERROR "rund_node_case requires COMMAND")
  endif()
  if(NOT RUND_NODE_CASE_BUILD_TARGETS)
    message(FATAL_ERROR "rund_node_case requires BUILD_TARGETS")
  endif()
  if(NOT RUND_NODE_CASE_LABELS)
    set(RUND_NODE_CASE_LABELS rund_node)
  endif()
  if(NOT RUND_NODE_CASE_TIMEOUT)
    if(RUND_NODE_CASE_RESOURCE_LOCK STREQUAL "rund_accel")
      set(RUND_NODE_CASE_TIMEOUT 1800)
    else()
      set(RUND_NODE_CASE_TIMEOUT 900)
    endif()
  endif()

  if(RUND_NODE_CASE_RESOURCE_LOCK STREQUAL "rund_accel")
    rund_accel_test_command(RUND_NODE_CASE_COMMAND "${name}"
      ${RUND_NODE_CASE_COMMAND})
  endif()

  add_test(NAME ${name} COMMAND ${RUND_NODE_CASE_COMMAND})
  set_tests_properties(${name}
    PROPERTIES
      LABELS "${RUND_NODE_CASE_LABELS}"
      TIMEOUT "${RUND_NODE_CASE_TIMEOUT}")
  if(RUND_NODE_CASE_RESOURCE_LOCK)
    set_tests_properties(${name}
      PROPERTIES RESOURCE_LOCK "${RUND_NODE_CASE_RESOURCE_LOCK}")
  endif()
  rund_test_route(${name}
    TARGETS ${RUND_NODE_CASE_BUILD_TARGETS})
endfunction()
