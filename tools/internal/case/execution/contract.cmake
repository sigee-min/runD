cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROOT OR ROOT STREQUAL "")
  message(FATAL_ERROR "ROOT is required")
endif()
if(NOT DEFINED BUILD OR BUILD STREQUAL "")
  message(FATAL_ERROR "BUILD is required")
endif()
get_filename_component(ROOT "${ROOT}" ABSOLUTE)
get_filename_component(BUILD "${BUILD}" ABSOLUTE)

set(fixture "${BUILD}/case-build-contract")
set(root "${fixture}/root")
set(build "${root}/.cache/focus/core")
set(bin "${fixture}/bin")
set(index "${build}/node/node-contract/index.tsv")
set(build_capture "${fixture}/build.tsv")
set(run_capture "${fixture}/run.tsv")
set(config_capture "${fixture}/config.tsv")
set(cmake_capture "${fixture}/cmake.called")
set(name "fixture.case")
set(target "node-runtime")
set(profile "core")
set(focus "fixture.anchor")
set(route "${name}\t${target}\t-\t${profile}")
set(case_shell sh)

file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY
  "${bin}"
  "${root}/tools/internal/case"
  "${root}/tools/internal/configure"
  "${root}/tools/internal/process"
  "${root}/tools/internal/state"
  "${build}/node/node-contract")
file(COPY "${ROOT}/tools/internal/case/run"
  DESTINATION "${root}/tools/internal/case")
file(WRITE "${index}" "${route}\n")

file(WRITE "${root}/tools/internal/state/lock" [=[#!/bin/sh
set -eu
[ "${1:-}" = --held ] || exit 2
exit "${RUND_HELD_CODE:-0}"
]=])
file(WRITE "${root}/tools/internal/configure/state" [=[#!/bin/sh
set -eu
[ "$#" -ge 1 ] && [ "$1" = matches ] || exit 2
: "${RUND_CONFIG_CAPTURE:?}"
: >"$RUND_CONFIG_CAPTURE"
for argument do
  printf '%s\n' "$argument" >>"$RUND_CONFIG_CAPTURE"
done
exit 0
]=])
file(WRITE "${root}/tools/internal/configure/contracts" [=[#!/bin/sh
exit 91
]=])
file(WRITE "${root}/tools/internal/state/ninja" [=[#!/bin/sh
set -eu
: "${RUND_BUILD_CAPTURE:?}"
pwd >"$RUND_BUILD_CAPTURE"
for argument do
  printf '%s\n' "$argument" >>"$RUND_BUILD_CAPTURE"
done
if [ "${RUND_REGEN_DRIFT:-0}" = 1 ]; then
  printf 'fixture.changed\tnode-runtime\t-\tcore\n' >"$RUND_ROUTE_INDEX"
fi
exit "${RUND_BUILD_CODE:-0}"
]=])
file(WRITE "${root}/tools/internal/process/run" [=[#!/bin/sh
set -eu
: "${RUND_RUN_CAPTURE:?}"
: >"$RUND_RUN_CAPTURE"
for argument do
  printf '%s\n' "$argument" >>"$RUND_RUN_CAPTURE"
done
if [ -n "${RUND_RUN_COUNT:-}" ]; then
  count=$(cat "$RUND_RUN_COUNT")
  count=$((count + 1))
  printf '%s\n' "$count" >"$RUND_RUN_COUNT"
fi
exit "${RUND_RUN_CODE:-0}"
]=])
file(WRITE "${bin}/cmake"
  "#!/bin/sh\nset -eu\nif [ \"\${1:-}\" = -E ] && [ \"\${2:-}\" = sha256sum ]; then\n  exec \"${CMAKE_COMMAND}\" \"\$@\"\nfi\n: >\"${cmake_capture}\"\nexit 99\n")
file(CHMOD
  "${root}/tools/internal/state/lock"
  "${root}/tools/internal/configure/state"
  "${root}/tools/internal/configure/contracts"
  "${root}/tools/internal/state/ninja"
  "${root}/tools/internal/process/run"
  "${bin}/cmake"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

function(run_case output)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PATH=${bin}:$ENV{PATH}"
      "RUND_BUILD_CAPTURE=${build_capture}"
      "RUND_RUN_CAPTURE=${run_capture}"
      "RUND_CONFIG_CAPTURE=${config_capture}"
      "RUND_ROUTE_INDEX=${index}"
      "RUND_TEST_FRESH=0"
      ${ARGN}
      "${case_shell}" "${root}/tools/internal/case/run"
      "${root}" "${build}" "${name}" -
      "${target}" - "${profile}" "${focus}" "${route}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  set(${output} "${result}" PARENT_SCOPE)
  set(${output}_stdout "${stdout}" PARENT_SCOPE)
  set(${output}_stderr "${stderr}" PARENT_SCOPE)
endfunction()

run_case(success)
if(NOT success EQUAL 0)
  message(FATAL_ERROR
    "Exact case execution failed: ${success_stdout}${success_stderr}")
endif()
file(READ "${build_capture}" build_arguments)
set(expected_build "${build}\n${target}\n")
if(NOT build_arguments STREQUAL expected_build)
  message(FATAL_ERROR
    "Exact case did not invoke the configured Ninja owner directly")
endif()
file(READ "${config_capture}" config_arguments)
set(expected_config
  "matches\n${root}\n${build}\nlocal\nDebug\n${focus}\n${focus}\t${target}\t-\t${profile}\n-DRUND_NODE_TEST_TAG:STRING=\n-DRUND_NODE_FOCUSED_BACKEND:STRING=\n")
if(NOT config_arguments STREQUAL expected_config)
  message(FATAL_ERROR
    "Exact case did not preserve its stable materialization focus")
endif()
set(first_config_arguments "${config_arguments}")
file(READ "${run_capture}" run_arguments)
set(expected_run
  "900\n${name}\n${build}/node/${target}\n--case\n${name}\n")
if(NOT run_arguments STREQUAL expected_run)
  message(FATAL_ERROR "Exact case execution arguments changed")
endif()
if(EXISTS "${cmake_capture}")
  message(FATAL_ERROR "Exact case retained the redundant CMake dispatcher")
endif()

# A sibling requested case sharing the same anchor must preserve the exact
# configuration identity and target while changing only --case execution.
set(name "fixture.second")
set(route "${name}\t${target}\t-\t${profile}")
file(WRITE "${index}" "${route}\n")
run_case(switch_success)
if(NOT switch_success EQUAL 0)
  message(FATAL_ERROR
    "Exact sibling switch failed: ${switch_success_stdout}${switch_success_stderr}")
endif()
file(READ "${config_capture}" switch_config_arguments)
if(NOT switch_config_arguments STREQUAL first_config_arguments)
  message(FATAL_ERROR
    "Exact sibling switch changed its materialization identity")
endif()
file(READ "${run_capture}" switch_run_arguments)
set(expected_switch_run
  "900\n${name}\n${build}/node/${target}\n--case\n${name}\n")
if(NOT switch_run_arguments STREQUAL expected_switch_run)
  message(FATAL_ERROR "Exact sibling switch executed the wrong case")
endif()

set(name "fixture.case")
set(route "${name}\t${target}\t-\t${profile}")
file(WRITE "${index}" "${route}\n")

file(REMOVE "${build_capture}" "${run_capture}")
run_case(build_failure "RUND_BUILD_CODE=42")
if(NOT build_failure EQUAL 42 OR EXISTS "${run_capture}")
  message(FATAL_ERROR
    "Exact case did not preserve the Ninja failure status")
endif()

file(WRITE "${index}" "${route}\n")
file(REMOVE "${build_capture}" "${run_capture}")
run_case(route_drift "RUND_REGEN_DRIFT=1")
if(NOT route_drift EQUAL 2 OR EXISTS "${run_capture}")
  message(FATAL_ERROR
    "Exact case executed after its regenerated route changed")
endif()

file(WRITE "${index}" "${route}\n")
file(REMOVE "${build_capture}" "${run_capture}")
run_case(run_failure "RUND_RUN_CODE=37")
if(NOT run_failure EQUAL 37)
  message(FATAL_ERROR
    "Exact case did not preserve the contract executable status")
endif()

file(WRITE "${index}" "${route}\n")
file(REMOVE "${build_capture}" "${run_capture}")
run_case(lock_failure "RUND_HELD_CODE=1")
if(NOT lock_failure EQUAL 2 OR EXISTS "${build_capture}" OR
   EXISTS "${run_capture}")
  message(FATAL_ERROR
    "Exact case executed without its validated profile lock")
endif()

find_program(dash NAMES dash)
if(dash)
  file(WRITE "${index}" "${route}\n")
  file(REMOVE "${build_capture}" "${run_capture}")
  set(case_shell "${dash}")
  run_case(dash_success)
  if(NOT dash_success EQUAL 0)
    message(FATAL_ERROR
      "Exact case is not portable to dash: "
      "${dash_success_stdout}${dash_success_stderr}")
  endif()
endif()

# Direct exact cases reuse only a byte-identical executable/index/runner pass.
# A changed executable invalidates the pass, and failures are never stored.
set(case_shell sh)
set(pass_count "${fixture}/pass-count")
set(case_artifact "${build}/node/${target}")
file(WRITE "${pass_count}" "0\n")
file(WRITE "${case_artifact}" "#!/bin/sh\nexit 0\n")
file(CHMOD "${case_artifact}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
file(WRITE "${index}" "${route}\n")

run_case(cache_first "RUND_RUN_COUNT=${pass_count}")
run_case(cache_hit "RUND_RUN_COUNT=${pass_count}")
file(READ "${pass_count}" cache_hit_count)
string(STRIP "${cache_hit_count}" cache_hit_count)
if(NOT cache_first EQUAL 0 OR NOT cache_hit EQUAL 0 OR
   NOT cache_hit_count STREQUAL "1")
  message(FATAL_ERROR
    "Exact case pass cache did not reuse one successful execution")
endif()

file(APPEND "${case_artifact}" "# changed bytes\n")
run_case(cache_invalidated "RUND_RUN_COUNT=${pass_count}")
file(READ "${pass_count}" cache_invalidated_count)
string(STRIP "${cache_invalidated_count}" cache_invalidated_count)
if(NOT cache_invalidated EQUAL 0 OR
   NOT cache_invalidated_count STREQUAL "2")
  message(FATAL_ERROR
    "Exact case pass cache ignored changed executable bytes")
endif()

file(APPEND "${case_artifact}" "# failing identity\n")
run_case(cache_failure_one
  "RUND_RUN_COUNT=${pass_count}" "RUND_RUN_CODE=37")
run_case(cache_failure_two
  "RUND_RUN_COUNT=${pass_count}" "RUND_RUN_CODE=37")
run_case(cache_recovery "RUND_RUN_COUNT=${pass_count}")
run_case(cache_recovery_hit "RUND_RUN_COUNT=${pass_count}")
file(READ "${pass_count}" cache_failure_count)
string(STRIP "${cache_failure_count}" cache_failure_count)
if(NOT cache_failure_one EQUAL 37 OR NOT cache_failure_two EQUAL 37 OR
   NOT cache_recovery EQUAL 0 OR NOT cache_recovery_hit EQUAL 0 OR
   NOT cache_failure_count STREQUAL "5")
  message(FATAL_ERROR
    "Exact case pass cache stored a failure or lost its status")
endif()

file(REMOVE_RECURSE "${fixture}")
