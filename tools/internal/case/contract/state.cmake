function(rund_case_contract_state_call
    out fixture build operation focus intent)
  set(selected_backend "")
  if(ARGN)
    list(LENGTH ARGN backend_count)
    if(NOT backend_count EQUAL 1)
      message(FATAL_ERROR
        "Configuration state accepts at most one backend")
    endif()
    list(GET ARGN 0 selected_backend)
  endif()
  execute_process(
    COMMAND sh "${fixture}/tools/internal/configure/state"
      "${operation}" "${fixture}" "${build}"
      local Debug "${focus}" "${intent}"
      -DRUND_NODE_TEST_TAG:STRING=
      "-DRUND_NODE_FOCUSED_BACKEND:STRING=${selected_backend}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(operation STREQUAL "write" AND NOT result EQUAL 0)
    string(STRIP "${output}\n${error}" detail)
    message(FATAL_ERROR
      "Contract configuration write failed: ${detail}")
  endif()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()

function(rund_case_contract_state fixture root)
  set(build "${fixture}/focus")
  set(test_runner "${fixture}/tools/test/run")
  get_filename_component(test_owner "${test_runner}" DIRECTORY)
  file(MAKE_DIRECTORY
    "${test_owner}"
    "${fixture}/tools/internal/state"
    "${fixture}/tools/internal/case"
    "${fixture}/tools/internal/configure"
    "${fixture}/tools/internal/lock"
    "${fixture}/tools/internal/source"
    "${build}")
  file(COPY "${root}/tools/test/run"
    DESTINATION "${test_owner}")
  file(COPY
    "${root}/tools/internal/state/lock"
    "${root}/tools/internal/state/ninja"
    "${root}/tools/internal/state/root"
    "${root}/tools/internal/state/roots.tsv"
    DESTINATION "${fixture}/tools/internal/state")
  file(COPY
    "${root}/tools/internal/configure/contracts"
    "${root}/tools/internal/configure/state"
    DESTINATION "${fixture}/tools/internal/configure")
  file(COPY "${root}/tools/internal/lock/run"
    DESTINATION "${fixture}/tools/internal/lock")
  file(COPY "${root}/tools/internal/case/run"
    DESTINATION "${fixture}/tools/internal/case")
  file(COPY "${root}/tools/internal/source/state"
    DESTINATION "${fixture}/tools/internal/source")
  file(WRITE "${build}/CMakeCache.txt"
    "RUND_NODE_FOCUSED_CASE:STRING=fixture.new\n")
  set(route "fixture.new\tnode-compute\t-\tcompute")

  rund_case_contract_state_call(
    cold "${fixture}" "${build}" matches fixture.new "${route}")
  if(cold EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration accepted a missing stamp")
  endif()
  rund_case_contract_state_call(
    write "${fixture}" "${build}" write fixture.new "${route}")
  rund_case_contract_state_call(
    warm "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT warm EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration rejected its exact stamp")
  endif()

  file(APPEND
    "${fixture}/node/tests/contract/new.cpp" "// content edit\n")
  rund_case_contract_state_call(
    content "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT content EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration treated source contents as topology")
  endif()
  file(REMOVE "${fixture}/node/tests/contract/new.cpp")
  rund_case_contract_state_call(
    deleted "${fixture}" "${build}" matches fixture.new "${route}")
  if(deleted EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a tracked source deletion")
  endif()
  file(WRITE
    "${fixture}/node/tests/contract/new.cpp" "// content edit\n")
  rund_case_contract_state_call(
    restored "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT restored EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration did not recover a restored source path")
  endif()

  file(RENAME
    "${fixture}/node/tests/contract/new.cpp"
    "${fixture}/node/tests/contract/moved.cpp")
  rund_case_contract_state_call(
    renamed "${fixture}" "${build}" matches fixture.new "${route}")
  if(renamed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a tracked source rename")
  endif()
  file(RENAME
    "${fixture}/node/tests/contract/moved.cpp"
    "${fixture}/node/tests/contract/new.cpp")
  rund_case_contract_state_call(
    rename_restored "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT rename_restored EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration did not recover a restored source rename")
  endif()

  file(REMOVE "${fixture}/node/tests/contract/new.cpp")
  file(CREATE_LINK
    "${fixture}/node/tests/contract/cases.def"
    "${fixture}/node/tests/contract/new.cpp"
    SYMBOLIC)
  rund_case_contract_state_call(
    type "${fixture}" "${build}" matches fixture.new "${route}")
  if(type EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a source type change")
  endif()
  file(REMOVE "${fixture}/node/tests/contract/new.cpp")
  file(WRITE
    "${fixture}/node/tests/contract/new.cpp" "// content edit\n")
  rund_case_contract_state_call(
    type_restored "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT type_restored EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration did not recover a restored source type")
  endif()

  file(WRITE "${fixture}/node/tests/contract/unowned.cpp" "")
  rund_case_contract_state_call(
    topology "${fixture}" "${build}" matches fixture.new "${route}")
  if(topology EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored an unconfigured source addition")
  endif()
  file(REMOVE "${fixture}/node/tests/contract/unowned.cpp")
  rund_case_contract_state_call(
    topology_restored
    "${fixture}" "${build}" matches fixture.new "${route}")
  if(NOT topology_restored EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration did not recover its source topology")
  endif()
  file(WRITE
    "${fixture}/node/tests/contract/.gitignore" "ignored.cpp\n")
  file(WRITE "${fixture}/node/tests/contract/ignored.cpp" "")
  rund_case_contract_state_call(
    ignored "${fixture}" "${build}" matches fixture.new "${route}")
  if(ignored EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored an excluded source addition")
  endif()
  file(REMOVE
    "${fixture}/node/tests/contract/ignored.cpp"
    "${fixture}/node/tests/contract/.gitignore")

  rund_case_contract_state_call(
    route_changed
    "${fixture}" "${build}" matches fixture.changed "${route}")
  if(route_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration accepted a different focus")
  endif()
  rund_case_contract_state_call(
    intent_changed
    "${fixture}" "${build}" matches fixture.new "tag=thread")
  if(intent_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration accepted a different intent")
  endif()
  rund_case_contract_state_call(
    cpu_changed
    "${fixture}" "${build}" matches fixture.new "${route}" cpu)
  if(cpu_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a focused backend")
  endif()
  rund_case_contract_state_call(
    native_changed
    "${fixture}" "${build}" matches fixture.new "${route}" metal)
  if(native_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a native focus")
  endif()

  file(APPEND
    "${fixture}/tools/internal/configure/contracts" "# changed\n")
  rund_case_contract_state_call(
    helper_changed
    "${fixture}" "${build}" matches fixture.new "${route}")
  if(helper_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a helper change")
  endif()
  rund_case_contract_state_call(
    rewrite "${fixture}" "${build}" write fixture.new "${route}")
  file(APPEND "${build}/CMakeCache.txt" "RUND_TEST_NODE:BOOL=OFF\n")
  rund_case_contract_state_call(
    cache_changed
    "${fixture}" "${build}" matches fixture.new "${route}")
  if(cache_changed EQUAL 0)
    message(FATAL_ERROR
      "Contract configuration ignored a cache change")
  endif()
endfunction()
