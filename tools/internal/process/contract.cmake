cmake_minimum_required(VERSION 3.20)

foreach(required IN ITEMS ROOT BUILD)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "process contract requires ${required}")
  endif()
endforeach()

find_program(shell NAMES sh REQUIRED)
set(runner "${ROOT}/tools/internal/process/run")
set(fixture "${BUILD}/process-contract")
set(empty_path "${fixture}/empty")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${empty_path}")

function(run_probe prefix)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  set(${prefix}_result "${result}" PARENT_SCOPE)
  set(${prefix}_stdout "${stdout}" PARENT_SCOPE)
  set(${prefix}_stderr "${stderr}" PARENT_SCOPE)
endfunction()

run_probe(usage "${shell}" "${runner}")
if(NOT usage_result EQUAL 2 OR NOT usage_stderr STREQUAL
    "usage: tools/internal/process/run <seconds> <name> <command> [args...]\n")
  message(FATAL_ERROR "bounded runner usage contract changed")
endif()

foreach(invalid_timeout IN ITEMS 0 invalid 1.5 -1)
  run_probe(invalid "${shell}" "${runner}" "${invalid_timeout}" invalid
    "${CMAKE_COMMAND}" -E true)
  if(NOT invalid_result EQUAL 2 OR NOT invalid_stderr STREQUAL
      "bounded run requires a positive integer timeout\n")
    message(FATAL_ERROR
      "bounded runner accepted invalid timeout: ${invalid_timeout}")
  endif()
endforeach()

run_probe(missing
  "${CMAKE_COMMAND}" -E env "PATH=${empty_path}"
  "${shell}" "${runner}" 1 missing "${CMAKE_COMMAND}" -E true)
if(NOT missing_result EQUAL 2 OR NOT missing_stderr STREQUAL
    "bounded run requires perl\n")
  message(FATAL_ERROR "bounded runner retained a second missing-Perl path")
endif()

run_probe(success "${shell}" "${runner}" 2 success
  "${CMAKE_COMMAND}" -E true)
if(NOT success_result EQUAL 0 OR NOT success_stdout STREQUAL "" OR
   NOT success_stderr STREQUAL "")
  message(FATAL_ERROR "bounded runner changed successful execution")
endif()

run_probe(failure "${shell}" "${runner}" 2 failure
  "${shell}" -c "exit 37")
if(NOT failure_result EQUAL 37)
  message(FATAL_ERROR
    "bounded runner changed child status: ${failure_result}")
endif()

run_probe(timeout "${shell}" "${runner}" 1 deadline
  "${CMAKE_COMMAND}" -E sleep 3)
if(NOT timeout_result EQUAL 142 OR
   NOT timeout_stderr MATCHES "deadline exceeded 1 seconds")
  message(FATAL_ERROR
    "bounded runner deadline contract changed: ${timeout_result}; ${timeout_stderr}")
endif()

set(leak_runner "${ROOT}/tools/internal/leaks/run")
set(leak_root "${fixture}/root")
set(leak_build "${fixture}/bin")
set(leak_target "${leak_build}/owner")
set(leak_tool "${fixture}/leaks")
set(leak_trace "${fixture}/leaks.trace")
set(leak_process "${leak_root}/tools/internal/process/run")
file(MAKE_DIRECTORY "${leak_build}"
  "${leak_root}/tools/internal/process")
file(WRITE "${leak_process}" [=[#!/bin/sh
shift 2
exec "$@"
]=])
file(WRITE "${leak_target}" [=[#!/bin/sh
printf 'target:%s\n' "$2" >>"$RUND_LEAK_TRACE"
case "$2" in
  fail) exit 37 ;;
  *) exit 0 ;;
esac
]=])
file(WRITE "${leak_tool}" [=[#!/bin/sh
if [ "$1" != "--atExit" ] || [ "$2" != "--" ]; then
  exit 64
fi
shift 2
printf 'leaks\n' >>"$RUND_LEAK_TRACE"
"$@" || child=$?
case "${RUND_LEAK_MODE:-pass}" in
  pass) exit 0 ;;
  leak) exit 23 ;;
  *) exit 64 ;;
esac
]=])
file(CHMOD "${leak_process}" "${leak_target}" "${leak_tool}"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

run_probe(leak_usage "${shell}" "${leak_runner}")
if(NOT leak_usage_result EQUAL 2 OR NOT leak_usage_stderr STREQUAL
    "usage: tools/internal/leaks/run <root> <build> <leaks> <target> <case>\n")
  message(FATAL_ERROR "leak runner usage contract changed")
endif()

file(REMOVE "${leak_trace}")
run_probe(leak_success "${CMAKE_COMMAND}" -E env
  "RUND_LEAK_TRACE=${leak_trace}"
  "${shell}" "${leak_runner}" "${leak_root}" "${leak_build}"
  "${leak_tool}" owner success)
file(READ "${leak_trace}" leak_success_trace)
if(NOT leak_success_result EQUAL 0 OR NOT leak_success_trace STREQUAL
    "target:success\nleaks\ntarget:success\n")
  message(FATAL_ERROR
    "leak runner changed semantic/leak execution order: ${leak_success_result}; ${leak_success_trace}")
endif()

file(REMOVE "${leak_trace}")
run_probe(leak_contract_failure "${CMAKE_COMMAND}" -E env
  "RUND_LEAK_TRACE=${leak_trace}"
  "${shell}" "${leak_runner}" "${leak_root}" "${leak_build}"
  "${leak_tool}" owner fail)
file(READ "${leak_trace}" leak_contract_failure_trace)
if(NOT leak_contract_failure_result EQUAL 37 OR
   NOT leak_contract_failure_trace STREQUAL "target:fail\n")
  message(FATAL_ERROR
    "leak runner accepted a failed semantic owner: ${leak_contract_failure_result}; ${leak_contract_failure_trace}")
endif()

file(REMOVE "${leak_trace}")
run_probe(leak_failure "${CMAKE_COMMAND}" -E env
  "RUND_LEAK_TRACE=${leak_trace}" "RUND_LEAK_MODE=leak"
  "${shell}" "${leak_runner}" "${leak_root}" "${leak_build}"
  "${leak_tool}" owner success)
if(NOT leak_failure_result EQUAL 23)
  message(FATAL_ERROR
    "leak runner changed native leak status: ${leak_failure_result}")
endif()

file(REMOVE_RECURSE "${fixture}")
