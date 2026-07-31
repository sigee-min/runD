function(rund_case_contract_focus build root)
  set(fixture "${build}/case-focus-fixture")
  file(REMOVE_RECURSE "${fixture}")
  file(MAKE_DIRECTORY
    "${fixture}/node/cmake/node/contract"
    "${fixture}/node/cmake/tests"
    "${fixture}/node/tests/contract/cases"
    "${fixture}/node/tests/contract"
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
    "RUND_NODE_TEST_SUITE(\"cases/focus.def\", compute)\n")
  file(WRITE "${fixture}/node/tests/contract/first.cpp" "")
  file(WRITE "${fixture}/node/tests/contract/second.cpp" "")
  set(registry
    "RUND_NODE_TEST_CASE(\"fixture.focus-first\", RunFixtureFirst, \"tests/contract/first.cpp\", product)\nRUND_NODE_TEST_CASE(\"fixture.focus-second\", RunFixtureSecond, \"tests/contract/second.cpp\", product)\n")
  file(WRITE "${fixture}/node/tests/contract/cases/focus.def"
    "${registry}")
  rund_case_contract_catalog(
    first "${fixture}" case fixture.focus-first -)
  rund_case_contract_catalog(
    second "${fixture}" case fixture.focus-second -)
  if(NOT first STREQUAL
      "fixture.focus-first\tnode-compute\t-\tproduct\tfixture.focus-first" OR
     NOT second STREQUAL
      "fixture.focus-second\tnode-compute\t-\tproduct\tfixture.focus-first")
    message(FATAL_ERROR
      "Profile-scoped cases do not share their canonical materialization anchor")
  endif()

  file(WRITE "${fixture}/node/tests/contract/third.cpp" "")
  file(APPEND "${fixture}/node/tests/contract/cases/focus.def"
    "RUND_NODE_TEST_CASE(\"fixture.focus-third\", RunFixtureThird, \"tests/contract/third.cpp\", product)\n")
  rund_case_contract_catalog_failure(
    "${fixture}" case fixture.focus-first - "bound is 2")
  file(REMOVE "${fixture}/node/tests/contract/third.cpp")
  file(WRITE "${fixture}/node/tests/contract/cases/focus.def"
    "RUND_NODE_TEST_CASE(\"fixture.focus-first\", RunFixtureFirst, \"tests/contract/first.cpp\", product)\nRUND_NODE_TEST_CASE(\"fixture.focus-second\", RunFixtureSecond, \"tests/contract/second.cpp\", product, backend)\n")
  rund_case_contract_catalog_failure(
    "${fixture}" case fixture.focus-first -
    "does not admit backend-selectable cases")
  file(WRITE "${fixture}/node/tests/contract/cases.def"
    "RUND_NODE_TEST_SUITE(\"cases/focus.def\", compute)\nRUND_NODE_TEST_SUITE(\"cases/other.def\", compute)\n")
  file(WRITE "${fixture}/node/tests/contract/cases/focus.def"
    "RUND_NODE_TEST_CASE(\"fixture.focus-first\", RunFixtureFirst, \"tests/contract/first.cpp\", product)\n")
  file(WRITE "${fixture}/node/tests/contract/cases/other.def"
    "RUND_NODE_TEST_CASE(\"fixture.focus-second\", RunFixtureSecond, \"tests/contract/second.cpp\", product)\n")
  rund_case_contract_catalog_failure(
    "${fixture}" case fixture.focus-first -
    "crosses source-suite boundaries")
  file(REMOVE "${fixture}/node/tests/contract/cases/other.def")
  file(WRITE "${fixture}/node/tests/contract/cases.def"
    "RUND_NODE_TEST_SUITE(\"cases/focus.def\", compute)\n")
  file(WRITE "${fixture}/node/tests/contract/cases/focus.def"
    "${registry}")
  rund_case_contract_catalog(
    recovered "${fixture}" case fixture.focus-second -)
  if(NOT recovered STREQUAL second)
    message(FATAL_ERROR
      "Profile-scoped catalog did not recover after rejection")
  endif()
  file(REMOVE_RECURSE "${fixture}")
endfunction()
