cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROOT OR ROOT STREQUAL "")
  message(FATAL_ERROR "ROOT is required")
endif()
if(NOT DEFINED BUILD OR BUILD STREQUAL "")
  message(FATAL_ERROR "BUILD is required")
endif()
get_filename_component(ROOT "${ROOT}" ABSOLUTE)
get_filename_component(BUILD "${BUILD}" ABSOLUTE)

set(rund_case_contract_query
  "${ROOT}/tools/internal/case/query.cmake")
set(rund_case_contract_owner
  "${ROOT}/tools/internal/case/contract")
include("${rund_case_contract_owner}/model.cmake")
include("${rund_case_contract_owner}/surface.cmake")
include("${rund_case_contract_owner}/catalog.cmake")
include("${rund_case_contract_owner}/focus.cmake")
include("${rund_case_contract_owner}/state.cmake")

# This runner owns contract order. Leaves expose one semantic operation each
# and do not execute during inclusion.
rund_case_contract_surface("${ROOT}")

set(fixture "${BUILD}/case-query-fixture")
rund_case_contract_prepare("${fixture}" "${ROOT}")
rund_case_contract_catalog_surface("${fixture}")
rund_case_contract_focus("${BUILD}" "${ROOT}")
rund_case_contract_catalog_cache(
  "${fixture}" "${RUND_CASE_CONTRACT_CATALOG_EXACT}")
rund_case_contract_state("${fixture}" "${ROOT}")
file(REMOVE_RECURSE "${fixture}")
