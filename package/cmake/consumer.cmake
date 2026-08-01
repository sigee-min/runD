foreach(required_var IN ITEMS ROOT BUILD_DIR PREFIX EXPECTED_SDK_VERSION)
  if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "consumer validation requires ${required_var}")
  endif()
endforeach()

if(NOT IS_DIRECTORY "${PREFIX}" OR
   NOT EXISTS "${PREFIX}/include" OR
   NOT EXISTS "${PREFIX}/lib/cmake/runD/runDConfig.cmake")
  message(FATAL_ERROR
    "package consumer requires the staged runD_package_install prefix: ${PREFIX}")
endif()

file(REAL_PATH "${PREFIX}" staged_prefix)
if(NOT IS_DIRECTORY "${BUILD_DIR}")
  message(FATAL_ERROR
    "package consumer requires an existing external build root: ${BUILD_DIR}")
endif()
file(REAL_PATH "${BUILD_DIR}" external_build_root)
set(external_consumer_dir "${external_build_root}/package-consumer")
set(component_consumer_dir
    "${external_build_root}/package-component-consumer")
set(collision_consumer_dir
    "${external_build_root}/package-collision-consumer")
set(staged_prefix_boundary "${staged_prefix}/")
set(external_consumer_boundary "${external_consumer_dir}/")
string(FIND "${external_consumer_boundary}" "${staged_prefix_boundary}"
       consumer_inside_prefix)
string(FIND "${staged_prefix_boundary}" "${external_consumer_boundary}"
       prefix_inside_consumer)
if(consumer_inside_prefix EQUAL 0 OR prefix_inside_consumer EQUAL 0)
  message(FATAL_ERROR
    "external consumer build and staged package prefix must not overlap")
endif()
set(consumer_dir "${external_consumer_dir}")

include("${ROOT}/package/cmake/identity.cmake")
if(DEFINED EXPECTED_RELEASE_MANIFEST)
  if("${EXPECTED_RELEASE_MANIFEST}" STREQUAL "" OR
     NOT EXISTS "${EXPECTED_RELEASE_MANIFEST}")
    message(FATAL_ERROR
      "expected Release source manifest is unavailable: ${EXPECTED_RELEASE_MANIFEST}")
  endif()

  set(expected_release_identity
      "${EXPECTED_RELEASE_MANIFEST}.identity.tsv")
  set(installed_release_dir "${PREFIX}/share/runD/release")
  set(installed_release_manifest
      "${installed_release_dir}/source-manifest.tsv")
  set(installed_release_identity
      "${installed_release_dir}/source-identity.tsv")
  rund_require_exact_provenance(
    "${installed_release_manifest}" "${EXPECTED_RELEASE_MANIFEST}"
    "source manifest")
  rund_require_exact_provenance(
    "${installed_release_identity}" "${expected_release_identity}"
    "source identity")
  rund_validate_source_identity(
    "${installed_release_manifest}" "${installed_release_identity}"
    "installed")

  if(DEFINED EXPECTED_ARTIFACT_IDENTITY)
    if("${EXPECTED_ARTIFACT_IDENTITY}" STREQUAL "" OR
       NOT EXISTS "${EXPECTED_ARTIFACT_IDENTITY}")
      message(FATAL_ERROR
        "expected artifact identity is unavailable: ${EXPECTED_ARTIFACT_IDENTITY}")
    endif()
    set(installed_artifact_identity
        "${installed_release_dir}/artifact-identity.tsv")
    rund_require_exact_provenance(
      "${installed_artifact_identity}" "${EXPECTED_ARTIFACT_IDENTITY}"
      "artifact identity")
    rund_validate_artifact_identity(
      "${installed_artifact_identity}" "${EXPECTED_SDK_VERSION}"
      "${PREFIX}/lib/cmake/runD/runDTargets.cmake" "" "installed")
  endif()
endif()

set(installed_license "${PREFIX}/share/runD/LICENSE")
if(NOT EXISTS "${installed_license}")
  message(FATAL_ERROR "installed package is missing the runD license")
endif()
file(SHA256 "${ROOT}/LICENSE" source_license_sha256)
file(SHA256 "${installed_license}" installed_license_sha256)
if(NOT source_license_sha256 STREQUAL installed_license_sha256)
  message(FATAL_ERROR "installed package license differs from source authority")
endif()

set(xxhash_source_license
    "${ROOT}/node/src/host/vendor/xxhash/LICENSE")
set(xxhash_installed_license
    "${PREFIX}/share/runD/licenses/xxhash/LICENSE")
if(NOT EXISTS "${xxhash_installed_license}")
  message(FATAL_ERROR "installed package is missing the xxHash license")
endif()
file(SHA256 "${xxhash_source_license}" xxhash_source_license_sha256)
file(SHA256 "${xxhash_installed_license}" xxhash_installed_license_sha256)
if(NOT xxhash_source_license_sha256 STREQUAL xxhash_installed_license_sha256)
  message(FATAL_ERROR
    "installed xxHash license differs from source authority")
endif()

file(REMOVE_RECURSE
  "${component_consumer_dir}"
  "${collision_consumer_dir}")

include("${ROOT}/package/cmake/sdk/surface.cmake")
rund_read_sdk_surface(
  "${ROOT}/package/docs/surface/headers.tsv"
  direct_headers private_header_files private_header_trees)
set(installed_header_manifest "${PREFIX}/share/runD/sdk-headers.tsv")
rund_read_sdk_inventory("${installed_header_manifest}"
                        installed_header_inventory)
rund_check_sdk_closure(
  installed_header_inventory direct_headers private_header_files
  private_header_trees)

file(GLOB_RECURSE installed_header_paths
  LIST_DIRECTORIES false
  RELATIVE "${PREFIX}/include"
  "${PREFIX}/include/*")
list(SORT installed_header_paths)
foreach(installed_header IN LISTS installed_header_paths)
  if(IS_SYMLINK "${PREFIX}/include/${installed_header}")
    message(FATAL_ERROR
      "installed SDK header is a symlink: ${installed_header}")
  endif()
endforeach()
if(NOT "${installed_header_paths}" STREQUAL "${installed_header_inventory}")
  message(FATAL_ERROR
    "installed SDK header tree differs from its exact closure manifest\n"
    "manifest: ${installed_header_inventory}\n"
    "tree:     ${installed_header_paths}")
endif()

foreach(direct_header IN LISTS direct_headers)
  if(NOT EXISTS "${PREFIX}/include/${direct_header}")
    message(FATAL_ERROR
      "installed SDK is missing direct header: ${direct_header}")
  endif()
endforeach()

set(doc_snippet_dir "${external_build_root}/package-doc-snippets")
execute_process(
  COMMAND "${ROOT}/package/cmake/snippets/generate"
          "${ROOT}" "${doc_snippet_dir}"
  RESULT_VARIABLE doc_snippet_result
  OUTPUT_VARIABLE doc_snippet_stdout
  ERROR_VARIABLE doc_snippet_stderr
  TIMEOUT 120)
if(NOT doc_snippet_result STREQUAL "0")
  message(FATAL_ERROR
    "official C++ documentation classification failed\n"
    "${doc_snippet_stdout}${doc_snippet_stderr}")
endif()
set(doc_snippet_targets "${doc_snippet_dir}/targets.cmake")
if(NOT EXISTS "${doc_snippet_targets}")
  message(FATAL_ERROR "documentation compiler target list is missing")
endif()

set(consumer_generator_arguments
  -G Ninja
  "-DCMAKE_MAKE_PROGRAM:FILEPATH=${ROOT}/tools/internal/state/ninja")

execute_process(
  COMMAND sh "${ROOT}/tools/internal/state/lock" --state
    "${ROOT}" "${component_consumer_dir}" "${CMAKE_COMMAND}"
    -S "${ROOT}/package/tests/consumer/component"
    -B "${component_consumer_dir}"
    ${consumer_generator_arguments}
    "-DCMAKE_PREFIX_PATH=${PREFIX}"
    "-DrunD_DIR:PATH=${PREFIX}/lib/cmake/runD"
    "-DRUND_EXPECTED_PACKAGE_DIR:PATH=${PREFIX}/lib/cmake/runD"
    "-DRUND_EXPECTED_SDK_VERSION:STRING=${EXPECTED_SDK_VERSION}"
  RESULT_VARIABLE component_configure_result
  OUTPUT_VARIABLE component_configure_stdout
  ERROR_VARIABLE component_configure_stderr
  TIMEOUT 300)
if(component_configure_result STREQUAL "0")
  message(FATAL_ERROR
    "unsupported component probe unexpectedly configured successfully")
endif()
string(CONCAT component_configure_output
  "${component_configure_stdout}" "${component_configure_stderr}")
string(FIND "${component_configure_output}"
  "runD expected component rejection verified"
  component_rejection_marker)
if(component_rejection_marker EQUAL -1)
  message(FATAL_ERROR
    "unsupported component probe failed before proving target isolation\n"
    "${component_configure_output}")
endif()

execute_process(
  COMMAND sh "${ROOT}/tools/internal/state/lock" --state
    "${ROOT}" "${collision_consumer_dir}" "${CMAKE_COMMAND}"
    -S "${ROOT}/package/tests/consumer/collision"
    -B "${collision_consumer_dir}"
    ${consumer_generator_arguments}
    "-DCMAKE_PREFIX_PATH=${PREFIX}"
    "-DrunD_DIR:PATH=${PREFIX}/lib/cmake/runD"
    "-DRUND_EXPECTED_SDK_VERSION:STRING=${EXPECTED_SDK_VERSION}"
  RESULT_VARIABLE collision_configure_result
  OUTPUT_VARIABLE collision_configure_stdout
  ERROR_VARIABLE collision_configure_stderr
  TIMEOUT 300)
if(collision_configure_result STREQUAL "0")
  message(FATAL_ERROR
    "foreign runD::sdk collision probe unexpectedly configured successfully")
endif()
string(CONCAT collision_configure_output
  "${collision_configure_stdout}" "${collision_configure_stderr}")
string(FIND "${collision_configure_output}"
  "runD::sdk already exists and is not owned by"
  collision_rejection_marker)
if(collision_rejection_marker EQUAL -1)
  message(FATAL_ERROR
    "foreign runD::sdk probe failed before proving prefix ownership\n"
    "${collision_configure_output}")
endif()

execute_process(
  COMMAND sh "${ROOT}/tools/internal/state/lock" --state
    "${ROOT}" "${consumer_dir}" "${CMAKE_COMMAND}"
    -S "${ROOT}/package/tests/consumer"
    -B "${consumer_dir}"
    "-DCMAKE_BUILD_TYPE=Release"
    ${consumer_generator_arguments}
    "-DCMAKE_PREFIX_PATH=${PREFIX}"
    "-DrunD_DIR:PATH=${PREFIX}/lib/cmake/runD"
    "-DRUND_EXPECTED_PACKAGE_DIR:PATH=${PREFIX}/lib/cmake/runD"
    "-DRUND_EXPECTED_SDK_VERSION:STRING=${EXPECTED_SDK_VERSION}"
    "-DRUND_DOC_SNIPPET_TARGETS:FILEPATH=${doc_snippet_targets}"
  RESULT_VARIABLE configure_result
  TIMEOUT 300)
if(NOT configure_result STREQUAL "0")
  message(FATAL_ERROR "package consumer configure failed: ${configure_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    --build "${consumer_dir}"
    --target rund_package_consumers
  RESULT_VARIABLE build_result
  TIMEOUT 900)
if(NOT build_result STREQUAL "0")
  message(FATAL_ERROR "package consumer build failed: ${build_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -D "ROOT=${ROOT}"
    -D "CONSUMER_BINARY_DIR=${consumer_dir}"
    -P "${ROOT}/package/cmake/consumer/run.cmake"
  RESULT_VARIABLE run_result
  TIMEOUT 300)
if(NOT run_result STREQUAL "0")
  message(FATAL_ERROR "package consumer run failed: ${run_result}")
endif()
