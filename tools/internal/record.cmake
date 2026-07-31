foreach(required IN ITEMS ROOT BUILD ROUTE STATUS SOURCE_MANIFEST)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "verification record requires ${required}")
  endif()
endforeach()

string(TIMESTAMP run_id "%Y%m%dT%H%M%SZ" UTC)
set(route_output "${ROOT}/.cache/evidence/${ROUTE}")
set(output "${route_output}/${run_id}")
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef record_nonce)
set(EVIDENCE_STAGING
    "${route_output}/.record-${run_id}-${record_nonce}")
file(MAKE_DIRECTORY "${route_output}")
file(MAKE_DIRECTORY "${EVIDENCE_STAGING}")

execute_process(
  COMMAND uname -a
  OUTPUT_VARIABLE host
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(cache "${BUILD}/CMakeCache.txt")
set(generator unknown)
set(compiler unknown)
if(EXISTS "${cache}")
  file(STRINGS "${cache}" generator_line
    REGEX "^CMAKE_GENERATOR:INTERNAL=")
  if(generator_line)
    string(REGEX REPLACE "^[^=]*=" "" generator "${generator_line}")
  endif()
endif()

include("${ROOT}/tools/internal/source/manifest/adopt.cmake")
set(revision "${source_revision}")
set(dirty "${source_dirty}")

set(ctest_log "${BUILD}/Testing/Temporary/LastTest.log")
if(EXISTS "${ctest_log}")
  configure_file("${ctest_log}" "${EVIDENCE_STAGING}/ctest.log" COPYONLY)
endif()
if(DEFINED LOG AND EXISTS "${LOG}")
  if(NOT DEFINED LOG_NAME OR "${LOG_NAME}" STREQUAL "")
    set(LOG_NAME run.log)
  endif()
  configure_file("${LOG}" "${EVIDENCE_STAGING}/${LOG_NAME}" COPYONLY)
endif()

set(toolchain_fields "")
set(artifact_fields "")
set(proof_fields "")
if(DEFINED PROOF_KIND)
  execute_process(
    COMMAND perl "${ROOT}/tools/internal/measure/compiler" "${BUILD}"
    RESULT_VARIABLE compiler_status
    OUTPUT_VARIABLE compiler
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT compiler_status EQUAL 0)
    set(compiler unknown)
  endif()
  foreach(required IN ITEMS PROOF PROOF_NAME PROOF_ROUTE PROOF_STATUS
                            PROOF_PROFILE PROOF_METRICS LOG LOG_NAME
                            WORKLOAD_STATUS WORKLOAD_EXIT PROFILE ARTIFACT)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
      file(REMOVE_RECURSE "${EVIDENCE_STAGING}")
      message(FATAL_ERROR "verification proof requires ${required}")
    endif()
  endforeach()
  if(NOT PROFILE MATCHES "^[a-z0-9]+$" OR
     NOT EXISTS "${ARTIFACT}" OR IS_SYMLINK "${ARTIFACT}" OR
     generator STREQUAL "unknown" OR compiler STREQUAL "unknown" OR
     NOT EXISTS "${compiler}")
    file(REMOVE_RECURSE "${EVIDENCE_STAGING}")
    message(FATAL_ERROR "invalid measurement execution context")
  endif()
  get_filename_component(artifact_name "${ARTIFACT}" NAME)
  file(SHA256 "${ARTIFACT}" artifact_sha256)
  file(SHA256 "${compiler}" compiler_sha256)
  string(APPEND toolchain_fields
    "compiler_sha256\t${compiler_sha256}\n")
  string(APPEND artifact_fields
    "profile\t${PROFILE}\n"
    "artifact\t${artifact_name}\n"
    "artifact_sha256\t${artifact_sha256}\n")
  if(NOT PROOF_KIND STREQUAL "performance" OR
     NOT PROOF_ROUTE STREQUAL ROUTE OR
     NOT STATUS MATCHES "^(passed|failed)$" OR
     NOT PROOF_STATUS MATCHES "^(passed|failed)$" OR
     NOT WORKLOAD_STATUS MATCHES "^(passed|failed)$" OR
     NOT WORKLOAD_EXIT MATCHES "^(0|[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])$" OR
     (WORKLOAD_STATUS STREQUAL "passed" AND NOT WORKLOAD_EXIT STREQUAL "0") OR
     (WORKLOAD_STATUS STREQUAL "failed" AND WORKLOAD_EXIT STREQUAL "0") OR
     (WORKLOAD_STATUS STREQUAL "failed" AND NOT PROOF_STATUS STREQUAL "failed") OR
     NOT PROOF_PROFILE MATCHES "^(-|[a-z0-9]+)$" OR
     NOT PROOF_METRICS MATCHES "^(0|[1-9][0-9]*)$" OR
     (STATUS STREQUAL "passed" AND NOT PROOF_STATUS STREQUAL "passed") OR
     (STATUS STREQUAL "passed" AND NOT WORKLOAD_STATUS STREQUAL "passed") OR
     (PROOF_STATUS STREQUAL "passed" AND
      (PROOF_PROFILE STREQUAL "-" OR PROOF_METRICS STREQUAL "0")) OR
     (PROOF_STATUS STREQUAL "failed" AND
      (NOT PROOF_PROFILE STREQUAL "-" OR NOT PROOF_METRICS STREQUAL "0")) OR
     NOT LOG_NAME MATCHES "^[A-Za-z0-9][A-Za-z0-9.-]*$" OR
     NOT PROOF_NAME MATCHES "^[A-Za-z0-9][A-Za-z0-9.-]*$" OR
     PROOF_NAME STREQUAL LOG_NAME OR
     NOT EXISTS "${LOG}" OR NOT EXISTS "${PROOF}")
    file(REMOVE_RECURSE "${EVIDENCE_STAGING}")
    message(FATAL_ERROR "invalid verification proof")
  endif()
  configure_file("${PROOF}" "${EVIDENCE_STAGING}/${PROOF_NAME}" COPYONLY)
  file(SHA256 "${EVIDENCE_STAGING}/${LOG_NAME}" proof_log_sha256)
  file(SHA256 "${EVIDENCE_STAGING}/${PROOF_NAME}" proof_result_sha256)
  string(APPEND proof_fields
    "workload:status\t${WORKLOAD_STATUS}\n"
    "workload:exit\t${WORKLOAD_EXIT}\n"
    "proof:kind\t${PROOF_KIND}\n"
    "proof:route\t${PROOF_ROUTE}\n"
    "proof:status\t${PROOF_STATUS}\n"
    "proof:profile\t${PROOF_PROFILE}\n"
    "proof:metrics\t${PROOF_METRICS}\n"
    "proof:log\t${LOG_NAME}\n"
    "proof:log:sha256\t${proof_log_sha256}\n"
    "proof:result\t${PROOF_NAME}\n"
    "proof:result:sha256\t${proof_result_sha256}\n")
endif()

file(WRITE "${EVIDENCE_STAGING}/run.tsv"
  "route\t${ROUTE}\n"
  "status\t${STATUS}\n"
  "revision\t${revision}\n"
  "dirty\t${dirty}\n"
  "generator\t${generator}\n"
  "compiler\t${compiler}\n"
  "${toolchain_fields}"
  "host\t${host}\n"
  "${artifact_fields}"
  "source_manifest_kind\tverification_after\n"
  "source_manifest_sha256\t${source_manifest_sha256}\n"
  "source_identity_sha256\t${source_identity_sha256}\n"
  "${proof_fields}")
if(EXISTS "${output}")
  file(REMOVE_RECURSE "${EVIDENCE_STAGING}")
  message(FATAL_ERROR "verification evidence already exists: ${output}")
endif()
file(RENAME "${EVIDENCE_STAGING}" "${output}")
message(STATUS "verification evidence: ${output}")
