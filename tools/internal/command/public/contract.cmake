if(NOT DEFINED ROOT OR ROOT STREQUAL "")
  message(FATAL_ERROR "public-command contract requires ROOT")
endif()

find_program(public_command_shell NAMES sh REQUIRED)

set(commands
  test/run
  check/run
  release/run
  release/darwin
  sanitize/run
  check/leaks
  check/platform/unavailable
  measure/scheduler/run
  measure/compute/run
  measure/flow/run
  measure/graph/services/run
  measure/telemetry/run
  measure/build/run
  evidence/status)

set(original_path "$ENV{PATH}")
set(ENV{PATH} "")

foreach(command IN LISTS commands)
  set(script "${ROOT}/tools/${command}")
  if(NOT EXISTS "${script}")
    message(FATAL_ERROR "public command is missing: ${script}")
  endif()

  set(short_output "")
  foreach(help_arg IN ITEMS -h --help)
    execute_process(
      COMMAND "${public_command_shell}" "${script}" "${help_arg}"
      WORKING_DIRECTORY "${ROOT}"
      RESULT_VARIABLE help_result
      OUTPUT_VARIABLE help_output
      ERROR_VARIABLE help_error)
    if(NOT help_result EQUAL 0 OR NOT help_error STREQUAL "")
      message(FATAL_ERROR
        "${command} ${help_arg} failed: exit=${help_result}\n${help_error}")
    endif()
    if(help_output STREQUAL "" OR
       NOT help_output MATCHES "^[^\n]+\nusage: tools/${command}")
      message(FATAL_ERROR
        "${command} ${help_arg} lacks purpose and canonical usage:\n${help_output}")
    endif()
    if(help_arg STREQUAL "-h")
      set(short_output "${help_output}")
    elseif(NOT help_output STREQUAL "${short_output}")
      message(FATAL_ERROR "${command} help spellings disagree")
    endif()
  endforeach()

  string(FIND "${short_output}" "-h, --help" help_mode_index)
  if(help_mode_index EQUAL -1)
    message(FATAL_ERROR "${command} help lacks the shared help modes")
  endif()

  if(command STREQUAL "test/run")
    foreach(token IN ITEMS
        "usage: tools/test/run [--fresh] [case]"
        "tools/test/run [--fresh] telemetry:detail"
        "tools/test/run [--fresh] <case> --backend cpu|metal|vulkan"
        "tools/test/run [--fresh] --match REGEX"
        "tools/test/run [--fresh] --list"
        "--fresh    Bypass the local pass cache.")
      string(FIND "${short_output}" "${token}" token_index)
      if(token_index EQUAL -1)
        message(FATAL_ERROR "tools/test/run help lacks mode: ${token}")
      endif()
    endforeach()
  elseif(command STREQUAL "sanitize/run")
    string(FIND "${short_output}" "<address|thread>" token_index)
    if(token_index EQUAL -1)
      message(FATAL_ERROR "tools/sanitize/run help lacks current modes")
    endif()
  elseif(command STREQUAL "evidence/status")
    string(FIND "${short_output}" "[route ...]" token_index)
    if(token_index EQUAL -1)
      message(FATAL_ERROR "tools/evidence/status help lacks route selection")
    endif()
  elseif(command STREQUAL "measure/compute/run")
    string(FIND "${short_output}"
      "--collective <cpu|metal|vulkan>" token_index)
    if(token_index EQUAL -1)
      message(FATAL_ERROR
        "tools/measure/compute/run help lacks focused collective mode")
    endif()
    string(FIND "${short_output}"
      "--pipeline <metal|vulkan>" token_index)
    if(token_index EQUAL -1)
      message(FATAL_ERROR
        "tools/measure/compute/run help lacks focused Pipeline mode")
    endif()
  endif()

  if(command STREQUAL "test/run")
    set(invalid_arguments compute.flow --backend invalid)
  else()
    set(invalid_arguments --invalid)
  endif()
  execute_process(
    COMMAND "${public_command_shell}" "${script}" ${invalid_arguments}
    WORKING_DIRECTORY "${ROOT}"
    RESULT_VARIABLE invalid_result
    OUTPUT_VARIABLE invalid_output
    ERROR_VARIABLE invalid_error)
  if(NOT invalid_result EQUAL 2 OR NOT invalid_output STREQUAL "" OR
     NOT invalid_error STREQUAL "${short_output}")
    message(FATAL_ERROR
      "${command} invalid-argument contract failed: exit=${invalid_result}\n"
      "stdout=${invalid_output}\nstderr=${invalid_error}")
  endif()
endforeach()

set(ENV{PATH} "${original_path}")
list(LENGTH commands command_count)
message(STATUS "public-command help contract: ${command_count} commands")
