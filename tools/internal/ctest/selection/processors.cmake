# Read actual generated registration rather than copying the probe's default.
# CTest JSON is a test oracle here, not selection/discovery in the public runner.
find_program(processor_ctest NAMES ctest REQUIRED)
execute_process(
  COMMAND "${processor_ctest}" --test-dir "${BUILD}" --show-only=json-v1
    -R "^tools[.]native-headers$"
  RESULT_VARIABLE processor_discovered
  OUTPUT_VARIABLE processor_metadata ERROR_VARIABLE processor_error)
if(NOT processor_discovered EQUAL 0)
  message(FATAL_ERROR "Cannot read native-header registration: ${processor_error}")
endif()
string(JSON processor_tests LENGTH "${processor_metadata}" tests)
if(processor_tests EQUAL 0)
  # Focused graphs without both native object contexts do not own this test.
  return()
endif()
if(NOT processor_tests EQUAL 1)
  message(FATAL_ERROR "Native-header registration is not unique")
endif()
string(JSON processor_argc LENGTH "${processor_metadata}" tests 0 command)
math(EXPR processor_last_arg "${processor_argc} - 1")
set(processor_jobs "")
foreach(index RANGE 0 ${processor_last_arg})
  string(JSON argument GET "${processor_metadata}" tests 0 command ${index})
  if(argument STREQUAL "--jobs")
    math(EXPR value_index "${index} + 1")
    if(NOT processor_jobs STREQUAL "" OR value_index GREATER processor_last_arg)
      message(FATAL_ERROR "Invalid native-header worker argument")
    endif()
    string(JSON processor_jobs GET "${processor_metadata}" tests 0 command ${value_index})
  endif()
endforeach()
string(JSON processor_properties LENGTH "${processor_metadata}" tests 0 properties)
math(EXPR processor_last_property "${processor_properties} - 1")
set(processor_slots "")
foreach(index RANGE 0 ${processor_last_property})
  string(JSON name GET "${processor_metadata}" tests 0 properties ${index} name)
  if(name STREQUAL "PROCESSORS")
    string(JSON processor_slots GET "${processor_metadata}" tests 0 properties ${index} value)
  endif()
endforeach()
if(NOT processor_jobs MATCHES "^[1-4]$" OR
   NOT processor_jobs STREQUAL processor_slots)
  message(FATAL_ERROR
    "Native-header workers are not reserved: jobs=${processor_jobs}, slots=${processor_slots}")
endif()

# Observe concurrent allocated CPU slots with the real scheduler. The wide
# worker accounts for its nested width; ordinary workers consume one slot.
set(processor_build "${fixture}/processor-build")
file(MAKE_DIRECTORY "${processor_build}")
set(processor_state "${processor_build}/state")
set(processor_selection "${processor_build}/selection.tsv")
file(WRITE "${processor_selection}" "target\tfixture-target\ntest\twide\n")
file(WRITE "${processor_build}/CTestTestfile.cmake"
  "add_test(wide \"${CMAKE_COMMAND}\" \"-DSTATE=${processor_state}\" \"-DWEIGHT=${processor_jobs}\" -P \"${concurrency_worker}\")\n"
  "set_tests_properties(wide PROPERTIES PROCESSORS ${processor_slots})\n")
foreach(index RANGE 1 4)
  file(APPEND "${processor_selection}" "test\tordinary-${index}\n")
  file(APPEND "${processor_build}/CTestTestfile.cmake"
    "add_test(ordinary-${index} \"${CMAKE_COMMAND}\" \"-DSTATE=${processor_state}\" -P \"${concurrency_worker}\")\n")
endforeach()
math(EXPR processor_mixed_limit "${processor_slots} + 1")
foreach(limit IN ITEMS "${processor_slots}" "${processor_mixed_limit}")
  file(WRITE "${processor_state}" "0;0")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "CTEST_PARALLEL_LEVEL=${limit}"
      sh "${ROOT}/tools/internal/ctest/selection/run"
      "${processor_build}" "${processor_selection}"
    RESULT_VARIABLE processor_result
    OUTPUT_VARIABLE processor_output ERROR_VARIABLE processor_run_error)
  file(READ "${processor_state}" processor_counts)
  list(GET processor_counts 0 processor_active)
  list(GET processor_counts 1 processor_peak)
  if(NOT processor_result EQUAL 0 OR NOT processor_active EQUAL 0 OR
     processor_peak LESS processor_jobs OR processor_peak GREATER limit)
    message(FATAL_ERROR
      "CTest nested-worker budget failed: active=${processor_active}, peak=${processor_peak}, limit=${limit}\n${processor_output}\n${processor_run_error}")
  endif()
  message(STATUS "CTest nested-worker slots: peak=${processor_peak} bound=${limit}")
endforeach()
