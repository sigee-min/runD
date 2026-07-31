if(NOT DEFINED CONSUMER_BINARY_DIR)
  message(FATAL_ERROR "consumer_run missing CONSUMER_BINARY_DIR")
endif()
if(NOT DEFINED ROOT)
  message(FATAL_ERROR "consumer_run missing ROOT")
endif()

set(rund_package_run_manifest "${CONSUMER_BINARY_DIR}/runs.tsv")
if(NOT EXISTS "${rund_package_run_manifest}")
  message(FATAL_ERROR "package consumer run manifest is missing")
endif()
file(STRINGS "${rund_package_run_manifest}" rund_package_runs)
if("${rund_package_runs}" STREQUAL "")
  message(FATAL_ERROR "package consumer run manifest is empty")
endif()

foreach(rund_package_run IN LISTS rund_package_runs)
  string(REPLACE "\t" ";" rund_package_fields "${rund_package_run}")
  list(LENGTH rund_package_fields rund_package_field_count)
  if(NOT rund_package_field_count EQUAL 3)
    message(FATAL_ERROR
      "package consumer run row is invalid: ${rund_package_run}")
  endif()
  list(GET rund_package_fields 0 rund_package_consumer)
  list(GET rund_package_fields 1 rund_package_expected)
  list(GET rund_package_fields 2 rund_package_resource)
  if(NOT rund_package_consumer MATCHES "^rund[-_][A-Za-z0-9_-]+$" OR
     NOT rund_package_expected MATCHES "^[0-9]+$" OR
     NOT rund_package_resource MATCHES "^(general|accel)$")
    message(FATAL_ERROR
      "package consumer run fields are invalid: ${rund_package_run}")
  endif()
  set(rund_package_consumer_path
      "${CONSUMER_BINARY_DIR}/${rund_package_consumer}")
  if(WIN32)
    set(rund_package_consumer_path
        "${CONSUMER_BINARY_DIR}/${rund_package_consumer}.exe")
  endif()
  set(rund_package_consumer_command "${rund_package_consumer_path}")
  if(rund_package_resource STREQUAL "accel")
    set(rund_package_consumer_command
        sh "${ROOT}/tools/internal/lock/accel"
        --wait
        sh "${ROOT}/tools/internal/process/run" 120
        "installed Compute consumer"
        "${rund_package_consumer_path}")
    set(rund_package_consumer_timeout 1200)
  else()
    set(rund_package_consumer_timeout 120)
  endif()
  execute_process(
    COMMAND ${rund_package_consumer_command}
    RESULT_VARIABLE rund_package_consumer_result
    OUTPUT_VARIABLE rund_package_consumer_stdout
    ERROR_VARIABLE rund_package_consumer_stderr
    TIMEOUT ${rund_package_consumer_timeout})
  if(NOT "${rund_package_consumer_result}" STREQUAL
         "${rund_package_expected}")
    message(FATAL_ERROR
            "package consumer failed: ${rund_package_consumer}\n"
            "expected: ${rund_package_expected}\n"
            "result: ${rund_package_consumer_result}\n"
            "stdout: ${rund_package_consumer_stdout}\n"
            "stderr: ${rund_package_consumer_stderr}")
  endif()
endforeach()
