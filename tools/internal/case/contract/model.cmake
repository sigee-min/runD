function(rund_case_contract_invoke out_result out_output out_error root)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -D "ROOT=${root}"
      ${ARGN}
      -P "${rund_case_contract_query}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  set(${out_result} "${result}" PARENT_SCOPE)
  set(${out_output} "${output}" PARENT_SCOPE)
  set(${out_error} "${error}" PARENT_SCOPE)
endfunction()

function(rund_case_contract_case root name backend expected)
  set(arguments -D "NAME=${name}")
  if(NOT backend STREQUAL "")
    list(APPEND arguments -D "BACKEND=${backend}")
  endif()
  rund_case_contract_invoke(
    result output error "${root}" ${arguments})
  if(NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    if(backend STREQUAL "")
      message(FATAL_ERROR "Exact-case query failed for ${name}: ${detail}")
    else()
      message(FATAL_ERROR
        "Exact-case backend query failed for ${name}: ${detail}")
    endif()
  endif()
  string(STRIP "${output}" output)
  if(NOT output STREQUAL expected)
    if(backend STREQUAL "")
      message(FATAL_ERROR
        "Exact-case query for ${name} returned '${output}', expected '${expected}'")
    else()
      message(FATAL_ERROR
        "Exact-case backend query for ${name} returned '${output}', expected '${expected}'")
    endif()
  endif()
endfunction()

function(rund_case_contract_rows root mode value expected_row expected_count)
  if(mode STREQUAL "list")
    set(arguments -D LIST_CASES=ON)
    set(label "Case list")
  elseif(mode STREQUAL "tag")
    set(arguments -D "TAG=${value}")
    set(label "Tag query for ${value}")
  else()
    message(FATAL_ERROR "Unknown case row contract mode: ${mode}")
  endif()

  rund_case_contract_invoke(
    result output error "${root}" ${arguments})
  if(NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR "${label} failed: ${detail}")
  endif()
  string(STRIP "${output}" output)
  if(output STREQUAL "")
    set(rows)
  else()
    string(REPLACE "\n" ";" rows "${output}")
  endif()
  list(LENGTH rows row_count)
  if(NOT expected_count STREQUAL "-" AND
     NOT row_count EQUAL expected_count)
    message(FATAL_ERROR
      "${label} returned ${row_count} rows, expected ${expected_count}")
  endif()
  if(NOT expected_row STREQUAL "-")
    list(FIND rows "${expected_row}" expected_index)
    if(expected_index EQUAL -1)
      message(FATAL_ERROR "${label} is missing '${expected_row}'")
    endif()
  endif()
  foreach(row IN LISTS rows)
    string(REPLACE "\t" ";" fields "${row}")
    list(LENGTH fields field_count)
    if(NOT field_count EQUAL 4)
      message(FATAL_ERROR "Malformed ${mode} query row: ${row}")
    endif()
  endforeach()
endfunction()

function(rund_case_contract_catalog out root mode value backend)
  execute_process(
    COMMAND sh "${root}/tools/internal/case/catalog"
      "${root}" "${mode}" "${value}" "${backend}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR
      "Case catalog ${mode} query failed for ${value}: ${detail}")
  endif()
  string(STRIP "${output}" output)
  set(${out} "${output}" PARENT_SCOPE)
endfunction()

function(rund_case_contract_catalog_failure
    root mode value backend expected)
  execute_process(
    COMMAND sh "${root}/tools/internal/case/catalog"
      "${root}" "${mode}" "${value}" "${backend}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(result EQUAL 0 OR NOT error MATCHES "${expected}")
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR
      "Case catalog accepted an invalid owner state: ${detail}")
  endif()
endfunction()

function(rund_case_contract_query_failure
    root name backend expected description)
  set(arguments -D "NAME=${name}")
  if(NOT backend STREQUAL "")
    list(APPEND arguments -D "BACKEND=${backend}")
  endif()
  rund_case_contract_invoke(
    result output error "${root}" ${arguments})
  if(result EQUAL 0 OR NOT error MATCHES "${expected}")
    string(STRIP "${output}\n${error}" detail)
    if(backend STREQUAL "")
      message(FATAL_ERROR
        "Exact-case query accepted ${description}: ${detail}")
    else()
      message(FATAL_ERROR
        "Exact-case backend query accepted ${description}: ${detail}")
    endif()
  endif()
endfunction()

function(rund_case_contract_prepare fixture root)
  file(REMOVE_RECURSE "${fixture}")
  file(MAKE_DIRECTORY
    "${fixture}/node/cmake/node/contract"
    "${fixture}/node/cmake/tests"
    "${fixture}/node/src"
    "${fixture}/node/tests/contract/cases"
    "${fixture}/tools/internal/case"
    "${fixture}/tools/internal/lock")
  file(COPY "${root}/node/cmake/node/profiles.cmake"
    DESTINATION "${fixture}/node/cmake/node")
  file(COPY "${root}/node/cmake/node/contract/routes.cmake"
    DESTINATION "${fixture}/node/cmake/node/contract")
  file(COPY "${root}/node/cmake/tests/registry.cmake"
    DESTINATION "${fixture}/node/cmake/tests")
  file(COPY
    "${root}/tools/internal/case/catalog"
    "${root}/tools/internal/case/query.cmake"
    DESTINATION "${fixture}/tools/internal/case")
  file(COPY "${root}/tools/internal/lock/run"
    DESTINATION "${fixture}/tools/internal/lock")
  file(WRITE "${fixture}/node/tests/contract/cases.def"
    "RUND_NODE_TEST_SUITE(\"cases/new.def\", compute)\n")
  file(WRITE "${fixture}/node/tests/contract/new.cpp" "")
  file(WRITE "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.new\", RunFixtureNew, \"tests/contract/new.cpp\", compute, backend+telemetry:detail)\n")
endfunction()
