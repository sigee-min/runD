function(rund_case_contract_catalog_surface fixture)
  rund_case_contract_case(
    "${fixture}" fixture.new ""
    "fixture.new\tnode-compute\t-\tcompute")
  rund_case_contract_case(
    "${fixture}" fixture.new cpu
    "fixture.new\tnode-compute\t-\tcpu-compute")
  rund_case_contract_case(
    "${fixture}" fixture.new metal
    "fixture.new\tnode-compute\t-\tcompute")
  rund_case_contract_rows(
    "${fixture}" list - "fixture.new\tnode-compute\t-\tcompute" 1)
  rund_case_contract_rows("${fixture}" tag backend - 1)
  rund_case_contract_rows("${fixture}" tag telemetry:detail - 1)
  rund_case_contract_rows("${fixture}" tag absent - 0)

  # The canonical parser publishes one sealed derivative catalog. Exact rows
  # additionally carry the parser-derived materialization anchor.
  rund_case_contract_catalog(
    catalog_exact "${fixture}" case fixture.new -)
  if(NOT catalog_exact STREQUAL
      "fixture.new\tnode-compute\t-\tcompute\tfixture.new")
    message(FATAL_ERROR "Case catalog exact projection disagrees")
  endif()
  rund_case_contract_catalog(
    catalog_backend "${fixture}" case fixture.new cpu)
  if(NOT catalog_backend STREQUAL
      "fixture.new\tnode-compute\t-\tcpu-compute\tfixture.new")
    message(FATAL_ERROR "Case catalog backend projection disagrees")
  endif()
  rund_case_contract_catalog(
    catalog_native "${fixture}" case fixture.new metal)
  if(NOT catalog_native STREQUAL catalog_exact)
    message(FATAL_ERROR
      "Case catalog native selection changed the base route")
  endif()
  rund_case_contract_catalog(
    catalog_tag "${fixture}" tag backend -)
  if(NOT catalog_tag STREQUAL
      "fixture.new\tnode-compute\t-\tcompute")
    message(FATAL_ERROR "Case catalog tag projection disagrees")
  endif()
  rund_case_contract_catalog(
    catalog_detail "${fixture}" tag telemetry:detail -)
  if(NOT catalog_detail STREQUAL catalog_tag)
    message(FATAL_ERROR "Namespaced case tag projection disagrees")
  endif()
  rund_case_contract_catalog(
    catalog_list "${fixture}" list - -)
  if(NOT catalog_list STREQUAL catalog_tag)
    message(FATAL_ERROR "Case catalog list projection disagrees")
  endif()

  set(RUND_CASE_CONTRACT_CATALOG_EXACT "${catalog_exact}" PARENT_SCOPE)
endfunction()

function(rund_case_contract_catalog_cache fixture catalog_exact)
  set(catalog_cache "${fixture}/.cache/case/catalog.tsv")
  file(SHA256 "${catalog_cache}" catalog_before)
  file(APPEND "${fixture}/tools/internal/case/query.cmake"
    "\n# parser identity fixture\n")
  rund_case_contract_catalog(
    parser_refresh "${fixture}" case fixture.new -)
  file(SHA256 "${catalog_cache}" catalog_after)
  if(NOT parser_refresh STREQUAL catalog_exact OR
     catalog_before STREQUAL catalog_after)
    message(FATAL_ERROR
      "Case catalog did not refresh its parser contract identity")
  endif()

  file(WRITE "${catalog_cache}" "catalog\n")
  rund_case_contract_catalog(
    partial_refresh "${fixture}" case fixture.new -)
  if(NOT partial_refresh STREQUAL catalog_exact)
    message(FATAL_ERROR "Case catalog accepted a partial cache")
  endif()
  file(READ "${catalog_cache}" corrupt_catalog)
  string(REPLACE "fixture.new" "fixture.bad" corrupt_catalog
    "${corrupt_catalog}")
  file(WRITE "${catalog_cache}" "${corrupt_catalog}")
  rund_case_contract_catalog(
    corrupt_refresh "${fixture}" case fixture.new -)
  if(NOT corrupt_refresh STREQUAL catalog_exact)
    message(FATAL_ERROR "Case catalog accepted a corrupt sealed cache")
  endif()

  file(RENAME
    "${fixture}/node/tests/contract/new.cpp"
    "${fixture}/node/tests/contract/missing.cpp")
  rund_case_contract_catalog_failure(
    "${fixture}" case fixture.new - "Missing node test case owner")
  file(RENAME
    "${fixture}/node/tests/contract/missing.cpp"
    "${fixture}/node/tests/contract/new.cpp")
  rund_case_contract_catalog(
    owner_refresh "${fixture}" case fixture.new -)
  if(NOT owner_refresh STREQUAL catalog_exact)
    message(FATAL_ERROR "Case catalog did not recover its owner topology")
  endif()

  file(WRITE "${fixture}/node/tests/contract/second.cpp" "")
  file(APPEND "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.second\", RunFixtureSecond, \"tests/contract/second.cpp\", compute)\n")
  rund_case_contract_catalog(updated_list "${fixture}" list - -)
  string(REPLACE "\n" ";" updated_rows "${updated_list}")
  list(LENGTH updated_rows updated_count)
  if(NOT updated_count EQUAL 2 OR
     NOT updated_list MATCHES "fixture[.]second")
    message(FATAL_ERROR
      "Case catalog did not invalidate after registry membership changed")
  endif()
  file(REMOVE "${fixture}/node/tests/contract/second.cpp")
  file(WRITE "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.new\", RunFixtureNew, \"tests/contract/new.cpp\", compute, backend+telemetry:detail)\n")

  # The query shares the production parser, including its rejection surface.
  file(APPEND "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.new\", RunFixtureDuplicate, \"tests/contract/new.cpp\", compute)\n")
  rund_case_contract_query_failure(
    "${fixture}" fixture.new "" "Duplicate node test case name"
    "a duplicate registry identity")

  file(WRITE "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.new\", RunFixtureNew, \"tests/contract/missing.cpp\", compute)\n")
  rund_case_contract_query_failure(
    "${fixture}" fixture.new "" "Missing node test case owner"
    "a missing owner source")

  file(WRITE "${fixture}/node/tests/contract/cases/new.def"
    "RUND_NODE_TEST_CASE(\"fixture.new\", RunFixtureNew, \"tests/contract/./new.cpp\", compute)\n")
  rund_case_contract_query_failure(
    "${fixture}" fixture.new "" "must use a canonical path"
    "a non-canonical owner path")
endfunction()
