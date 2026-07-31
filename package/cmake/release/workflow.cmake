if(NOT DEFINED ROOT)
  message(FATAL_ERROR "release workflow contract requires ROOT")
endif()
if(NOT DEFINED BUILD OR "${BUILD}" STREQUAL "")
  set(BUILD "${ROOT}/.cache/package-lifecycle-hardcut")
endif()

set(verifier_path "${ROOT}/tools/internal/package/verify")
set(identity_contract_path "${ROOT}/package/cmake/identity.cmake")
set(version_path "${ROOT}/tools/internal/package/version")
set(platform_status_path "${ROOT}/tools/internal/platform/status")
set(platform_registry_path "${ROOT}/package/docs/platform/tuples.tsv")
set(snippet_generator_path "${ROOT}/package/cmake/snippets/generate")
set(linux_workflow_path "${ROOT}/.github/workflows/release.yml")
set(ctest_selection_query_path
    "${ROOT}/tools/internal/ctest/selection/query.cmake")

file(READ "${linux_workflow_path}" linux_workflow)
if(NOT linux_workflow MATCHES "BUILD_DIR: \\.cache/release" OR
   linux_workflow MATCHES "\\.cache/ci/linux-sdk-build")
  message(FATAL_ERROR
    "Linux SDK workflow must use the registered Release build root")
endif()
string(FIND "${linux_workflow}" "cat \"$checksum_path\""
       linux_local_checksum_read)
string(FIND "${linux_workflow}" "cat \"$CHECKSUM_PATH\""
       linux_exported_checksum_read)
if(linux_local_checksum_read EQUAL -1 OR
   NOT linux_exported_checksum_read EQUAL -1)
  message(FATAL_ERROR
    "Linux SDK candidate step must read its checksum through the local path")
endif()

set(snippet_fixture_root "${BUILD}/docs-snippet-fixture")
file(REMOVE_RECURSE "${snippet_fixture_root}")
file(MAKE_DIRECTORY
  "${snippet_fixture_root}/valid/package/tests/consumer/example"
  "${snippet_fixture_root}/valid/docs"
  "${snippet_fixture_root}/bare"
  "${snippet_fixture_root}/links/docs"
  "${snippet_fixture_root}/metadata"
  "${snippet_fixture_root}/mismatch/package/tests/consumer/example")
set(valid_snippet "int main() { return 0; }\n")
file(WRITE
  "${snippet_fixture_root}/valid/package/tests/consumer/example/quick.cpp"
  "${valid_snippet}")
file(WRITE "${snippet_fixture_root}/valid/README.md"
  "[First step](docs/Guide.md#first-step)\n\n"
  "`[Inline example](docs/Missing.md)`\n\n"
  "```text\n[Fenced example](docs/Missing.md)\n```\n\n"
  "```cpp compile run source=package/tests/consumer/example/quick.cpp\n"
  "${valid_snippet}```\n\n"
  "```cpp fragment\nauto value = context.value();\n```\n\n"
  "```cpp compile\nstatic_assert(sizeof(int) >= 2);\n```\n")
file(WRITE "${snippet_fixture_root}/valid/docs/Guide.md"
  "# Guide\n\n## First Step\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/valid"
          "${snippet_fixture_root}/valid-output"
  RESULT_VARIABLE valid_snippet_result
  OUTPUT_VARIABLE valid_snippet_stdout
  ERROR_VARIABLE valid_snippet_stderr)
if(NOT valid_snippet_result STREQUAL "0" OR
   NOT valid_snippet_stdout MATCHES
       "docs snippets: 2 compile, 1 run, 1 fragment" OR
   NOT valid_snippet_stdout MATCHES "docs links: 1 local, 1 anchored")
  message(FATAL_ERROR
    "valid docs snippet fixture failed\n"
    "${valid_snippet_stdout}${valid_snippet_stderr}")
endif()

file(WRITE "${snippet_fixture_root}/links/docs/Guide.md"
  "# Existing\n")
file(WRITE "${snippet_fixture_root}/links/README.md"
  "[Missing file](docs/Missing.md)\n"
  "[Missing anchor](docs/Guide.md#missing)\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/links"
          "${snippet_fixture_root}/links-output"
  RESULT_VARIABLE links_result
  OUTPUT_VARIABLE links_stdout
  ERROR_VARIABLE links_stderr)
string(CONCAT links_output "${links_stdout}" "${links_stderr}")
if(links_result STREQUAL "0" OR
   NOT links_output MATCHES "local link target is unavailable" OR
   NOT links_output MATCHES "local link anchor is unavailable")
  message(FATAL_ERROR
    "broken documentation links did not fail closed\n${links_output}")
endif()
set(valid_snippet_output "${snippet_fixture_root}/valid-output")
set(valid_generated_snippet "${valid_snippet_output}/snippet-0002.cpp")
file(TIMESTAMP "${valid_generated_snippet}" valid_snippet_timestamp_before
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
file(WRITE "${valid_snippet_output}/stale.cpp" "stale\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/valid"
          "${valid_snippet_output}"
  RESULT_VARIABLE stable_snippet_result
  OUTPUT_VARIABLE stable_snippet_stdout
  ERROR_VARIABLE stable_snippet_stderr)
file(TIMESTAMP "${valid_generated_snippet}" valid_snippet_timestamp_after
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
if(NOT stable_snippet_result STREQUAL "0" OR
   NOT valid_snippet_timestamp_before STREQUAL valid_snippet_timestamp_after OR
   EXISTS "${valid_snippet_output}/stale.cpp" OR
   EXISTS "${valid_snippet_output}.next")
  message(FATAL_ERROR
    "stable docs snippet promotion failed\n"
    "before: ${valid_snippet_timestamp_before}\n"
    "after:  ${valid_snippet_timestamp_after}\n"
    "${stable_snippet_stdout}${stable_snippet_stderr}")
endif()
file(READ "${valid_snippet_output}/snippets.tsv" valid_snippet_manifest)
string(FIND "${valid_snippet_manifest}" "${valid_snippet_output}/snippet-0002.cpp"
       stable_snippet_path)
string(FIND "${valid_snippet_manifest}" "${valid_snippet_output}.next"
       candidate_snippet_path)
if(stable_snippet_path EQUAL -1 OR NOT candidate_snippet_path EQUAL -1)
  message(FATAL_ERROR
    "docs snippet manifest does not retain the stable output path")
endif()
set(valid_canonical_snippet "${valid_snippet_output}/snippet-0001.cpp")
file(TIMESTAMP "${valid_canonical_snippet}" canonical_timestamp_before
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
file(TIMESTAMP "${valid_generated_snippet}" changed_timestamp_before
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
file(READ "${snippet_fixture_root}/valid/README.md" changed_snippet_document)
string(REPLACE "sizeof(int) >= 2" "sizeof(int) >= 4"
       changed_snippet_document "${changed_snippet_document}")
file(WRITE "${snippet_fixture_root}/valid/README.md"
     "${changed_snippet_document}")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/valid"
          "${valid_snippet_output}"
  RESULT_VARIABLE changed_snippet_result
  OUTPUT_VARIABLE changed_snippet_stdout
  ERROR_VARIABLE changed_snippet_stderr)
file(TIMESTAMP "${valid_canonical_snippet}" canonical_timestamp_after
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
file(TIMESTAMP "${valid_generated_snippet}" changed_timestamp_after
     "%Y-%m-%dT%H:%M:%S.%fZ" UTC)
if(NOT changed_snippet_result STREQUAL "0" OR
   NOT canonical_timestamp_before STREQUAL canonical_timestamp_after OR
   changed_timestamp_before STREQUAL changed_timestamp_after OR
   EXISTS "${valid_snippet_output}.next")
  message(FATAL_ERROR
    "changed docs snippet promotion did not isolate its timestamp\n"
    "canonical before: ${canonical_timestamp_before}\n"
    "canonical after:  ${canonical_timestamp_after}\n"
    "changed before:   ${changed_timestamp_before}\n"
    "changed after:    ${changed_timestamp_after}\n"
    "${changed_snippet_stdout}${changed_snippet_stderr}")
endif()

file(WRITE "${snippet_fixture_root}/bare/README.md"
  "```cpp\nint main() { return 0; }\n```\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/bare"
          "${snippet_fixture_root}/bare-output"
  RESULT_VARIABLE bare_snippet_result
  OUTPUT_VARIABLE bare_snippet_stdout
  ERROR_VARIABLE bare_snippet_stderr)
string(CONCAT bare_snippet_output
  "${bare_snippet_stdout}" "${bare_snippet_stderr}")
if(bare_snippet_result STREQUAL "0" OR
   NOT bare_snippet_output MATCHES
       "C\\+\\+ fence must be classified as compile or fragment")
  message(FATAL_ERROR
    "bare docs snippet fixture did not fail closed\n${bare_snippet_output}")
endif()

file(WRITE "${snippet_fixture_root}/metadata/README.md"
  "```cpp compile maybe\nint main() { return 0; }\n```\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/metadata"
          "${snippet_fixture_root}/metadata-output"
  RESULT_VARIABLE metadata_snippet_result
  OUTPUT_VARIABLE metadata_snippet_stdout
  ERROR_VARIABLE metadata_snippet_stderr)
string(CONCAT metadata_snippet_output
  "${metadata_snippet_stdout}" "${metadata_snippet_stderr}")
if(metadata_snippet_result STREQUAL "0" OR
   NOT metadata_snippet_output MATCHES
       "unsupported or duplicate metadata")
  message(FATAL_ERROR
    "unknown docs snippet metadata did not fail closed\n"
    "${metadata_snippet_output}")
endif()

file(WRITE
  "${snippet_fixture_root}/mismatch/package/tests/consumer/example/quick.cpp"
  "${valid_snippet}")
file(WRITE "${snippet_fixture_root}/mismatch/README.md"
  "```cpp compile source=package/tests/consumer/example/quick.cpp\n"
  "int main() { return 1; }\n```\n")
execute_process(
  COMMAND sh "${snippet_generator_path}"
          "${snippet_fixture_root}/mismatch"
          "${snippet_fixture_root}/mismatch-output"
  RESULT_VARIABLE mismatch_snippet_result
  OUTPUT_VARIABLE mismatch_snippet_stdout
  ERROR_VARIABLE mismatch_snippet_stderr)
string(CONCAT mismatch_snippet_output
  "${mismatch_snippet_stdout}" "${mismatch_snippet_stderr}")
if(mismatch_snippet_result STREQUAL "0" OR
   NOT mismatch_snippet_output MATCHES
       "C\\+\\+ fence differs from canonical source")
  message(FATAL_ERROR
    "divergent canonical docs snippet did not fail closed\n"
    "${mismatch_snippet_output}")
endif()

execute_process(
  COMMAND sh "${version_path}" "${ROOT}"
  RESULT_VARIABLE version_result
  OUTPUT_VARIABLE package_version
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT version_result STREQUAL "0" OR
   NOT package_version STREQUAL "1.0.1")
  message(FATAL_ERROR "package version owner did not resolve SDK 1.0.1")
endif()

file(STRINGS "${platform_registry_path}" platform_rows)
list(LENGTH platform_rows platform_row_count)
if(NOT platform_row_count EQUAL 4)
  message(FATAL_ERROR "platform registry must contain one header and three rows")
endif()
list(GET platform_rows 0 platform_header)
if(NOT platform_header STREQUAL
   "triplet\tstatus\tproducer\trunner\tcompiler\tartifact")
  message(FATAL_ERROR "platform registry header is not canonical")
endif()
foreach(triplet IN ITEMS darwin-arm64 linux-x64 windows-x64)
  execute_process(
    COMMAND sh "${platform_status_path}" "${ROOT}" "${triplet}"
    RESULT_VARIABLE status_result
    OUTPUT_VARIABLE platform_status
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT status_result STREQUAL "0" OR platform_status STREQUAL "")
    message(FATAL_ERROR "${triplet} platform authority is invalid")
  endif()
endforeach()

set(platform_fixture "${BUILD}/platform-registry-fixture")
file(REMOVE_RECURSE "${platform_fixture}")
file(MAKE_DIRECTORY "${platform_fixture}/package/docs/platform")
set(platform_fixture_registry
    "${platform_fixture}/package/docs/platform/tuples.tsv")
set(platform_fixture_header
    "triplet\tstatus\tproducer\trunner\tcompiler\tartifact\n")
file(WRITE "${platform_fixture_registry}"
  "${platform_fixture_header}"
  "darwin-arm64\tsupported\ttools/release/darwin\tnative-darwin-arm64\tAppleClang\trelease\n"
  "linux-x64\tvalidated\t.github/workflows/release.yml\tubuntu-24.04\tgcc\tcandidate\n"
  "windows-x64\tnot-supported\tnone\tnone\tnone\tnone\n")
execute_process(
  COMMAND sh "${platform_status_path}" "${platform_fixture}" "darwin-arm64"
  RESULT_VARIABLE platform_fixture_result
  OUTPUT_VARIABLE platform_fixture_status
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT platform_fixture_result STREQUAL "0" OR
   NOT platform_fixture_status STREQUAL "supported")
  message(FATAL_ERROR
    "coherent supported platform fixture was rejected")
endif()

foreach(incoherent_row IN ITEMS
    "darwin-arm64\tsupported\ttools/release/darwin\tnative-darwin-arm64\tAppleClang\tcandidate"
    "darwin-arm64\tvalidated\ttools/release/darwin\tnative-darwin-arm64\tAppleClang\trelease"
    "darwin-arm64\tnot-supported\ttools/release/darwin\tnative-darwin-arm64\tAppleClang\tnone")
  file(WRITE "${platform_fixture_registry}"
    "${platform_fixture_header}"
    "${incoherent_row}\n"
    "linux-x64\tvalidated\t.github/workflows/release.yml\tubuntu-24.04\tgcc\tcandidate\n"
    "windows-x64\tnot-supported\tnone\tnone\tnone\tnone\n")
  execute_process(
    COMMAND sh "${platform_status_path}" "${platform_fixture}" "darwin-arm64"
    RESULT_VARIABLE incoherent_platform_result
    OUTPUT_QUIET ERROR_QUIET)
  if(incoherent_platform_result STREQUAL "0")
    message(FATAL_ERROR
      "platform registry accepted an incoherent status/artifact row")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "RUND_CTEST_ROUTES=${BUILD}/rund-test-routes.tsv"
    "RUND_CTEST_PATTERN=^package[.]consumer$"
    "${CMAKE_COMMAND}" -P "${ctest_selection_query_path}"
  RESULT_VARIABLE package_route_result
  OUTPUT_VARIABLE package_route
  ERROR_VARIABLE package_route_error)
if(NOT package_route_result EQUAL 0)
  string(STRIP "${package_route_error}" package_route_error)
  message(FATAL_ERROR
    "package consumer generated route is invalid: ${package_route_error}")
endif()
string(STRIP "${package_route}" package_route)
if(NOT package_route STREQUAL
    "target\trunD_package_install\ntest\tpackage.consumer")
  message(FATAL_ERROR
    "package consumer generated route disagrees:\n${package_route}")
endif()

set(verifier_fixture "${BUILD}/package-verifier-contract")
set(verifier_name "rund-sdk-1.0.1-darwin-arm64")
set(verifier_payload "${verifier_fixture}/payload")
set(verifier_archive "${verifier_fixture}/${verifier_name}.tar.gz")
set(verifier_checksum "${verifier_fixture}/${verifier_name}.sha256")
set(verifier_destination "${verifier_fixture}/install")
file(REMOVE_RECURSE "${verifier_fixture}")
file(MAKE_DIRECTORY
  "${verifier_payload}/${verifier_name}" "${verifier_destination}")
file(WRITE "${verifier_payload}/${verifier_name}/file" "fixture\n")
file(CREATE_LINK "../outside"
  "${verifier_payload}/${verifier_name}/link" SYMBOLIC)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${verifier_archive}"
          "${verifier_name}"
  WORKING_DIRECTORY "${verifier_payload}"
  RESULT_VARIABLE verifier_archive_result)
if(NOT verifier_archive_result STREQUAL "0")
  message(FATAL_ERROR "SDK verifier fixture archive creation failed")
endif()
file(WRITE "${verifier_checksum}"
  "0000000000000000000000000000000000000000000000000000000000000000  "
  "${verifier_name}.tar.gz\n")
execute_process(
  COMMAND sh "${verifier_path}" "${verifier_archive}"
          "${verifier_checksum}" "${verifier_destination}"
  RESULT_VARIABLE verifier_checksum_result
  OUTPUT_VARIABLE verifier_checksum_stdout
  ERROR_VARIABLE verifier_checksum_stderr)
if(verifier_checksum_result STREQUAL "0" OR
   NOT "${verifier_checksum_stdout}${verifier_checksum_stderr}" MATCHES
       "archive SHA-256 mismatch")
  message(FATAL_ERROR "SDK verifier accepted a mismatched archive checksum")
endif()
file(SHA256 "${verifier_archive}" verifier_archive_sha256)
file(WRITE "${verifier_checksum}"
  "${verifier_archive_sha256}  ${verifier_name}.tar.gz\n")
execute_process(
  COMMAND sh "${verifier_path}" "${verifier_archive}"
          "${verifier_checksum}" "${verifier_destination}"
  RESULT_VARIABLE verifier_link_result
  OUTPUT_VARIABLE verifier_link_stdout
  ERROR_VARIABLE verifier_link_stderr)
file(REMOVE_RECURSE "${verifier_fixture}")
if(verifier_link_result STREQUAL "0" OR
   NOT "${verifier_link_stdout}${verifier_link_stderr}" MATCHES
       "link or unsupported entry type")
  message(FATAL_ERROR "SDK verifier accepted an archive link")
endif()

set(fixture "${BUILD}/package-lifecycle-contract")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${fixture}")
include("${identity_contract_path}")
set(public_target "${fixture}/runDTargets.cmake")
file(WRITE "${public_target}" "fixture-target\n")

set(source_manifest "${fixture}/source-manifest.tsv")
set(source_identity "${fixture}/source-identity.tsv")
file(WRITE "${source_manifest}" "fixture-source\n")
rund_write_sealed_file("${source_manifest}")
rund_write_source_identity(
  "${source_manifest}" "${source_identity}"
  "0000000000000000000000000000000000000000" true fixture)

set(darwin_identity "${fixture}/darwin.tsv")
rund_write_artifact_identity_fixture(
  "${darwin_identity}" darwin-arm64 "${public_target}" 1.0.1)

set(linux_identity "${fixture}/linux.tsv")
rund_write_artifact_identity_fixture(
  "${linux_identity}" linux-x64 "${public_target}" 1.0.1)

function(run_identity identity triplet result output)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -D "IDENTITY=${identity}"
      -D "EXPECTED_SDK_VERSION=1.0.1"
      -D "EXPECTED_TRIPLET=${triplet}"
      -D "PUBLIC_TARGET=${public_target}"
      -D "IDENTITY_OWNER=fixture"
      -D "SOURCE_MANIFEST=${source_manifest}"
      -D "SOURCE_IDENTITY=${source_identity}"
      -D "SOURCE_OWNER=fixture"
      -P "${identity_contract_path}"
    RESULT_VARIABLE command_result
    OUTPUT_VARIABLE command_stdout
    ERROR_VARIABLE command_stderr)
  set(${result} "${command_result}" PARENT_SCOPE)
  set(${output} "${command_stdout}${command_stderr}" PARENT_SCOPE)
endfunction()

foreach(identity_triplet IN ITEMS darwin-arm64 linux-x64)
  if(identity_triplet STREQUAL "darwin-arm64")
    set(identity "${darwin_identity}")
  else()
    set(identity "${linux_identity}")
  endif()
  run_identity("${identity}" "${identity_triplet}" result output)
  if(NOT result STREQUAL "0")
    file(REMOVE_RECURSE "${fixture}")
    message(FATAL_ERROR
      "valid ${identity_triplet} identity was rejected: ${output}")
  endif()
endforeach()

file(APPEND "${linux_identity}.sha256" " ")
run_identity("${linux_identity}" "linux-x64" result output)
if(result STREQUAL "0" OR
   NOT output MATCHES "artifact identity provenance seal is not canonical")
  file(REMOVE_RECURSE "${fixture}")
  message(FATAL_ERROR "noncanonical identity seal was not rejected")
endif()
rund_write_sealed_file("${linux_identity}")

file(STRINGS "${linux_identity}" malformed_linux_rows)
list(REMOVE_AT malformed_linux_rows 0)
list(JOIN malformed_linux_rows "\n" malformed_linux)
file(WRITE "${linux_identity}" "${malformed_linux}\n")
rund_write_sealed_file("${linux_identity}")
run_identity("${linux_identity}" "linux-x64" result output)
if(result STREQUAL "0" OR
   NOT output MATCHES "keys are not the exact canonical schema")
  file(REMOVE_RECURSE "${fixture}")
  message(FATAL_ERROR "malformed Linux identity schema was not rejected")
endif()

file(APPEND "${public_target}" "changed\n")
run_identity("${darwin_identity}" "darwin-arm64" result output)
if(result STREQUAL "0" OR
   NOT output MATCHES "public target SHA-256 differs from artifact identity")
  file(REMOVE_RECURSE "${fixture}")
  message(FATAL_ERROR "changed public target was not rejected")
endif()

set(installed_manifest "${fixture}/installed-source-manifest.tsv")
configure_file("${source_manifest}" "${installed_manifest}" COPYONLY)
configure_file("${source_manifest}.sha256" "${installed_manifest}.sha256" COPYONLY)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -D "INSTALLED_PROVENANCE=${installed_manifest}"
    -D "EXPECTED_PROVENANCE=${source_manifest}"
    -D "PROVENANCE_LABEL=source manifest"
    -P "${identity_contract_path}"
  RESULT_VARIABLE exact_result
  OUTPUT_VARIABLE exact_stdout
  ERROR_VARIABLE exact_stderr)
if(NOT exact_result STREQUAL "0")
  file(REMOVE_RECURSE "${fixture}")
  message(FATAL_ERROR
    "exact source provenance was rejected: ${exact_stdout}${exact_stderr}")
endif()
file(APPEND "${installed_manifest}" "changed\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -D "INSTALLED_PROVENANCE=${installed_manifest}"
    -D "EXPECTED_PROVENANCE=${source_manifest}"
    -D "PROVENANCE_LABEL=source manifest"
    -P "${identity_contract_path}"
  RESULT_VARIABLE tampered_result
  OUTPUT_VARIABLE tampered_stdout
  ERROR_VARIABLE tampered_stderr)
file(REMOVE_RECURSE "${fixture}")
if(tampered_result STREQUAL "0" OR
   NOT "${tampered_stdout}${tampered_stderr}" MATCHES
       "seal does not match its payload")
  message(FATAL_ERROR "tampered source provenance was not rejected")
endif()
