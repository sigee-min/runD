cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROOT OR ROOT STREQUAL "")
  message(FATAL_ERROR "ROOT is required")
endif()
if(NOT DEFINED BUILD OR BUILD STREQUAL "")
  message(FATAL_ERROR "BUILD is required")
endif()
get_filename_component(ROOT "${ROOT}" ABSOLUTE)
get_filename_component(BUILD "${BUILD}" ABSOLUTE)

set(fixture "${BUILD}/build-ctest-selection-fixture")
set(fake_bin "${fixture}/bin")
set(fake_root "${fixture}/root")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY
  "${fake_bin}"
  "${fake_root}/tools/internal/state")
file(WRITE "${fake_bin}/cmake" [=[#!/bin/sh
set -eu
: "${RUND_SELECTION_METADATA_COUNT:?}"
: "${RUND_SELECTION_MODE:?}"
case "$1" in
  -P)
    count=$(cat "$RUND_SELECTION_METADATA_COUNT")
    count=$((count + 1))
    printf '%s\n' "$count" > "$RUND_SELECTION_METADATA_COUNT"
    if [ "$RUND_SELECTION_MODE" = shared ]; then
      printf 'target\tnode-runtime\ntest\tnode.runtime\ntest\tnode.runtime.task\ntest\tnode.runtime.task.host\ntest\tnode.runtime.task.net\ntest\tnode.runtime.task.reactor\ntest\tnode.runtime.task.replay\n'
    elif [ "$RUND_SELECTION_MODE" = target ] ||
         [ "$RUND_SELECTION_MODE" = parallel ]; then
      printf 'target\tfixture-target\ntest\tfixture-target-test\n'
    elif [ "$RUND_SELECTION_MODE" = empty ]; then
      :
    else
      printf 'test\tfixture-script\n'
    fi
    ;;
  *)
    echo "unexpected cmake invocation: $*" >&2
    exit 2
    ;;
esac
]=])
file(CHMOD "${fake_bin}/cmake"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
file(WRITE "${fake_root}/tools/internal/state/ninja" [=[#!/bin/sh
set -eu
: "${RUND_SELECTION_BUILD_COUNT:?}"
: "${RUND_SELECTION_BUILD_ARGS:?}"
count=$(cat "$RUND_SELECTION_BUILD_COUNT")
count=$((count + 1))
printf '%s\n' "$count" > "$RUND_SELECTION_BUILD_COUNT"
printf '%s\n' "$@" > "$RUND_SELECTION_BUILD_ARGS"
]=])
file(CHMOD "${fake_root}/tools/internal/state/ninja"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

function(run_selection_contract mode expected_metadata expected_build)
  set(case_dir "${fixture}/${mode}")
  set(metadata_count "${case_dir}/metadata-count")
  set(build_count "${case_dir}/build-count")
  set(build_args "${case_dir}/build-args")
  set(output "${case_dir}/selection.tsv")
  file(MAKE_DIRECTORY "${case_dir}" "${case_dir}/build")
  file(WRITE "${metadata_count}" "0\n")
  file(WRITE "${build_count}" "0\n")
  file(WRITE "${build_args}" "")
  file(WRITE "${case_dir}/build/rund-test-routes.tsv" "fixture\n")

  set(parallel_env "--unset=CMAKE_BUILD_PARALLEL_LEVEL")
  if(ARGC GREATER 3)
    list(APPEND parallel_env "CMAKE_BUILD_PARALLEL_LEVEL=${ARGV3}")
  endif()
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      ${parallel_env}
      "PATH=${fake_bin}:$ENV{PATH}"
      "RUND_SELECTION_METADATA_COUNT=${metadata_count}"
      "RUND_SELECTION_BUILD_COUNT=${build_count}"
      "RUND_SELECTION_BUILD_ARGS=${build_args}"
      "RUND_SELECTION_MODE=${mode}"
      sh "${ROOT}/tools/internal/ctest/selection/build"
      "${fake_root}" "${case_dir}/build" fixture "${output}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    string(STRIP "${stdout}\n${stderr}" detail)
    message(FATAL_ERROR
      "build-ctest-selection ${mode} contract failed: ${detail}")
  endif()

  file(READ "${metadata_count}" actual_metadata)
  file(READ "${build_count}" actual_build)
  string(STRIP "${actual_metadata}" actual_metadata)
  string(STRIP "${actual_build}" actual_build)
  if(NOT actual_metadata STREQUAL expected_metadata OR
     NOT actual_build STREQUAL expected_build)
    message(FATAL_ERROR
      "build-ctest-selection ${mode} calls metadata=${actual_metadata}, build=${actual_build}; expected ${expected_metadata}, ${expected_build}")
  endif()

  file(READ "${output}" selection)
  if(mode STREQUAL "empty")
    if(NOT selection STREQUAL "")
      message(FATAL_ERROR "empty selection did not remain empty")
    endif()
  elseif(mode STREQUAL "script")
    if(NOT selection STREQUAL "test\tfixture-script\n")
      message(FATAL_ERROR
        "script-only selection did not preserve its test metadata")
    endif()
  elseif(mode STREQUAL "shared")
    string(CONCAT expected_selection
      "target\tnode-runtime\n"
      "test\tnode.runtime\n"
      "test\tnode.runtime.task\n"
      "test\tnode.runtime.task.host\n"
      "test\tnode.runtime.task.net\n"
      "test\tnode.runtime.task.reactor\n"
      "test\tnode.runtime.task.replay\n")
    if(NOT selection STREQUAL expected_selection)
      message(FATAL_ERROR
        "shared target selection did not retain six process rows")
    endif()
    file(READ "${build_args}" actual_build_args)
    string(CONCAT expected_build_args
      "-C\n${case_dir}/build\nnode-runtime\n")
    if(NOT actual_build_args STREQUAL expected_build_args)
      message(FATAL_ERROR
        "shared target selection built a noncanonical target:\n${actual_build_args}")
    endif()
  else()
    string(CONCAT expected_selection
      "target\tfixture-target\n"
      "test\tfixture-target-test\n")
    if(NOT selection STREQUAL expected_selection)
      message(FATAL_ERROR
        "target selection did not retain its cache identity")
    endif()
    if(mode STREQUAL "parallel")
      file(READ "${build_args}" actual_build_args)
      string(CONCAT expected_build_args
        "-C\n${case_dir}/build\n-j\n3\nfixture-target\n")
      if(NOT actual_build_args STREQUAL expected_build_args)
        message(FATAL_ERROR
          "parallel selection did not forward its build bound:\n${actual_build_args}")
      endif()
    endif()
  endif()
endfunction()

run_selection_contract(script 1 0)
run_selection_contract(target 1 1)
run_selection_contract(parallel 1 1 3)
run_selection_contract(shared 1 1)
run_selection_contract(empty 1 0)

# Six Runtime CTest process owners share one executable.  The metadata owner
# must preserve all six rows while reducing their build-target union to one.
set(resource_build "${fixture}/resource-build")
file(MAKE_DIRECTORY "${resource_build}")
string(CONCAT resource_routes
  "routes\n"
  "route\tnode.runtime\tnode-runtime\t-\t900\tdirect\n"
  "route\tnode.runtime.task\tnode-runtime\t-\t900\tdirect\n"
  "route\tnode.runtime.task.host\tnode-runtime\t-\t900\tdirect\n"
  "route\tnode.runtime.task.net\tnode-runtime\t-\t900\tdirect\n"
  "route\tnode.runtime.task.reactor\tnode-runtime\t-\t900\tdirect\n"
  "route\tnode.runtime.task.replay\tnode-runtime\t-\t900\tdirect\n")
string(SHA256 resource_seal "${resource_routes}")
file(WRITE "${resource_build}/rund-test-routes.tsv"
  "${resource_routes}seal\t${resource_seal}\n")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "RUND_CTEST_ROUTES=${resource_build}/rund-test-routes.tsv"
    "RUND_CTEST_PATTERN=^node[.]runtime"
    "${CMAKE_COMMAND}" -P
    "${ROOT}/tools/internal/ctest/selection/query.cmake"
  RESULT_VARIABLE resource_result
  OUTPUT_VARIABLE resource_output
  ERROR_VARIABLE resource_error)
if(NOT resource_result EQUAL 0)
  string(STRIP "${resource_output}\n${resource_error}" detail)
  message(FATAL_ERROR "shared Runtime CTest metadata failed: ${detail}")
endif()
string(STRIP "${resource_output}" resource_output)
string(CONCAT expected_resource_output
  "target\tnode-runtime\n"
  "test\tnode.runtime\n"
  "test\tnode.runtime.task\n"
  "test\tnode.runtime.task.host\n"
  "test\tnode.runtime.task.net\n"
  "test\tnode.runtime.task.reactor\n"
  "test\tnode.runtime.task.replay")
if(NOT resource_output STREQUAL expected_resource_output)
  message(FATAL_ERROR
    "shared Runtime CTest metadata did not deduplicate its target:\n${resource_output}")
endif()

file(WRITE "${fake_bin}/ctest" [=[#!/bin/sh
set -eu
: "${RUND_SELECTION_CTEST_ARGS:?}"
printf '%s\n' "$@" > "$RUND_SELECTION_CTEST_ARGS"
]=])
file(CHMOD "${fake_bin}/ctest"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

set(run_selection "${fixture}/run-selection.tsv")
set(run_args "${fixture}/run-args.tsv")
file(WRITE "${run_selection}"
  "target\tfixture-target\n"
  "test\talpha.one\n"
  "test\tbeta-two\n")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${fake_bin}:$ENV{PATH}"
    "RUND_SELECTION_CTEST_ARGS=${run_args}"
    sh "${ROOT}/tools/internal/ctest/selection/run"
    "${fixture}/run-build" "${run_selection}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error)
if(NOT run_result EQUAL 0)
  string(STRIP "${run_output}\n${run_error}" detail)
  message(FATAL_ERROR "frozen CTest runner failed: ${detail}")
endif()
file(READ "${run_args}" actual_run_args)
string(CONCAT expected_run_args
  "--test-dir\n${fixture}/run-build\n-R\n^(alpha[.]one|beta-two)$\n"
  "--output-on-failure\n--no-tests=error\n--parallel\n")
if(NOT actual_run_args STREQUAL expected_run_args)
  message(FATAL_ERROR
    "frozen CTest runner arguments disagree:\n${actual_run_args}")
endif()

file(WRITE "${run_selection}" "target\tfixture-target\n")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${fake_bin}:$ENV{PATH}"
    "RUND_SELECTION_CTEST_ARGS=${run_args}"
    sh "${ROOT}/tools/internal/ctest/selection/run"
    "${fixture}/run-build" "${run_selection}"
  RESULT_VARIABLE malformed_result
  OUTPUT_QUIET ERROR_QUIET)
if(NOT malformed_result EQUAL 2)
  message(FATAL_ERROR
    "frozen CTest runner accepted malformed metadata: ${malformed_result}")
endif()

# A local pass is reusable only while the frozen selection, route index,
# generated CTest metadata, host/tool identity, and built executable bytes are
# unchanged.  Failures and resource-runner selections never populate it.
set(cache_fixture "${fixture}/cache")
set(cache_root "${cache_fixture}/root")
set(cache_build "${cache_fixture}/build")
set(cache_bin "${cache_fixture}/bin")
set(cache_count "${cache_fixture}/ctest-count")
set(cache_selection "${cache_build}/selection.tsv")
set(cache_routes "${cache_build}/rund-test-routes.tsv")
set(cache_artifact "${cache_build}/bin/fixture-target")
file(MAKE_DIRECTORY
  "${cache_root}/tools/internal/ctest/selection"
  "${cache_root}/tools/internal/state"
  "${cache_build}/bin"
  "${cache_bin}")
file(COPY
  "${ROOT}/tools/internal/ctest/selection/run"
  "${ROOT}/tools/internal/ctest/selection/cache"
  DESTINATION "${cache_root}/tools/internal/ctest/selection")
file(WRITE "${cache_root}/tools/internal/state/ninja" [=[#!/bin/sh
set -eu
[ "$#" -eq 5 ] && [ "$1" = -C ] && [ "$3" = -t ] &&
  [ "$4" = query ] && [ "$5" = fixture-target ] || exit 2
printf 'fixture-target:\n  input: phony\n    bin/fixture-target\n  outputs:\n'
]=])
file(WRITE "${cache_bin}/ctest" [=[#!/bin/sh
set -eu
if [ "${1:-}" = --version ]; then
  printf 'ctest version fixture\n'
  exit 0
fi
: "${RUND_CACHE_CTEST_COUNT:?}"
count=$(cat "$RUND_CACHE_CTEST_COUNT")
count=$((count + 1))
printf '%s\n' "$count" >"$RUND_CACHE_CTEST_COUNT"
exit "${RUND_CACHE_CTEST_CODE:-0}"
]=])
file(WRITE "${cache_count}" "0\n")
file(WRITE "${cache_selection}"
  "target\tfixture-target\n"
  "test\tfixture-test\n")
file(WRITE "${cache_routes}"
  "route\tfixture-test\tfixture-target\t-\t30\tdirect\n")
file(WRITE "${cache_build}/CTestTestfile.cmake" "# fixture\n")
file(WRITE "${cache_artifact}" "#!/bin/sh\nexit 0\n")
file(CHMOD
  "${cache_root}/tools/internal/state/ninja"
  "${cache_bin}/ctest"
  "${cache_artifact}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

function(run_cache_contract expected_count expected_result)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PATH=${cache_bin}:$ENV{PATH}"
      "RUND_CACHE_CTEST_COUNT=${cache_count}"
      ${ARGN}
      sh "${cache_root}/tools/internal/ctest/selection/cache"
      "${cache_root}" "${cache_build}" "${cache_selection}"
    RESULT_VARIABLE cache_result
    OUTPUT_VARIABLE cache_output
    ERROR_VARIABLE cache_error)
  if(NOT cache_result EQUAL expected_result)
    string(STRIP "${cache_output}\n${cache_error}" detail)
    message(FATAL_ERROR
      "CTest pass cache returned ${cache_result}, expected ${expected_result}: ${detail}")
  endif()
  file(READ "${cache_count}" actual_count)
  string(STRIP "${actual_count}" actual_count)
  if(NOT actual_count STREQUAL expected_count)
    message(FATAL_ERROR
      "CTest pass cache executed ${actual_count} time(s), expected ${expected_count}")
  endif()
endfunction()

run_cache_contract(1 0)
run_cache_contract(1 0)
file(APPEND "${cache_artifact}" "# changed bytes\n")
run_cache_contract(2 0)
file(WRITE "${cache_routes}"
  "route\tfixture-test\tfixture-target\trund_accel\t1800\taccel\n")
run_cache_contract(3 0)
file(WRITE "${cache_routes}"
  "route\tfixture-test\tfixture-target\t-\t30\tdirect\n")
file(APPEND "${cache_artifact}" "# failure identity\n")
run_cache_contract(4 41 "RUND_CACHE_CTEST_CODE=41")
run_cache_contract(5 41 "RUND_CACHE_CTEST_CODE=41")
run_cache_contract(6 0)
run_cache_contract(6 0)

file(WRITE "${run_selection}" "")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${fake_bin}:$ENV{PATH}"
    "RUND_SELECTION_CTEST_ARGS=${run_args}"
    sh "${ROOT}/tools/internal/ctest/selection/run"
    "${fixture}/run-build" "${run_selection}"
  RESULT_VARIABLE empty_result
  OUTPUT_QUIET ERROR_VARIABLE empty_error)
if(NOT empty_result EQUAL 2 OR
   NOT empty_error MATCHES "CTest selection matched no tests")
  message(FATAL_ERROR
    "frozen CTest runner did not report an empty selection exactly: ${empty_result}; ${empty_error}")
endif()
file(REMOVE_RECURSE "${fixture}")
