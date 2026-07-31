include_guard(GLOBAL)

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

function(rund_require_sealed_file payload label owner)
  if(NOT EXISTS "${payload}" OR NOT EXISTS "${payload}.sha256")
    message(FATAL_ERROR "${label} provenance file is missing: ${payload}")
  endif()

  file(READ "${payload}.sha256" seal_text)
  string(LENGTH "${seal_text}" seal_length)
  if(NOT seal_length EQUAL 65)
    message(FATAL_ERROR
      "${owner} ${label} provenance seal is not canonical")
  endif()
  string(SUBSTRING "${seal_text}" 0 64 sealed_sha256)
  string(SUBSTRING "${seal_text}" 64 1 seal_terminal)
  if(NOT sealed_sha256 MATCHES "^[0-9a-f]+$" OR
     NOT seal_terminal STREQUAL "\n")
    message(FATAL_ERROR
      "${owner} ${label} provenance seal is not canonical")
  endif()

  file(SHA256 "${payload}" observed_sha256)
  if(NOT observed_sha256 STREQUAL sealed_sha256)
    message(FATAL_ERROR
      "${owner} ${label} provenance seal does not match its payload")
  endif()
endfunction()

function(rund_write_sealed_file payload)
  if(NOT EXISTS "${payload}" OR IS_DIRECTORY "${payload}")
    message(FATAL_ERROR "sealed payload is unavailable: ${payload}")
  endif()
  file(SHA256 "${payload}" payload_sha256)
  string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef seal_nonce)
  set(seal "${payload}.sha256")
  set(temporary "${seal}.${seal_nonce}.tmp")
  file(WRITE "${temporary}" "${payload_sha256}\n")
  file(RENAME "${temporary}" "${seal}")
endfunction()

function(rund_require_exact_provenance installed expected label)
  rund_require_sealed_file("${installed}" "${label}" "installed")
  rund_require_sealed_file("${expected}" "${label}" "expected")

  file(READ "${installed}" installed_payload HEX)
  file(READ "${expected}" expected_payload HEX)
  file(READ "${installed}.sha256" installed_seal)
  file(READ "${expected}.sha256" expected_seal)
  if(NOT installed_payload STREQUAL expected_payload OR
     NOT installed_seal STREQUAL expected_seal)
    message(FATAL_ERROR
      "installed ${label} provenance differs from the producing Release route")
  endif()
endfunction()

function(rund_read_source_identity
         manifest identity owner revision_output dirty_output)
  rund_require_sealed_file("${manifest}" "source manifest" "${owner}")
  rund_require_sealed_file("${identity}" "source identity" "${owner}")
  file(SHA256 "${manifest}" manifest_sha256)
  file(READ "${identity}" identity_text)
  string(LENGTH "${identity_text}" identity_length)
  if(identity_length EQUAL 0 OR identity_text MATCHES "\r" OR
     identity_text MATCHES "\n\n")
    message(FATAL_ERROR "${owner} source identity is not canonical TSV")
  endif()
  math(EXPR identity_last "${identity_length} - 1")
  string(SUBSTRING "${identity_text}" ${identity_last} 1 identity_terminal)
  if(NOT identity_terminal STREQUAL "\n")
    message(FATAL_ERROR
      "${owner} source identity lacks a canonical terminal newline")
  endif()
  file(STRINGS "${identity}" identity_rows)
  list(LENGTH identity_rows identity_row_count)
  if(NOT identity_row_count EQUAL 3)
    message(FATAL_ERROR "${owner} source identity schema is not exact")
  endif()
  list(GET identity_rows 0 manifest_row)
  list(GET identity_rows 1 revision_row)
  list(GET identity_rows 2 dirty_row)
  if(NOT manifest_row STREQUAL
     "source_manifest_sha256\t${manifest_sha256}")
    message(FATAL_ERROR
      "${owner} source identity does not bind the source manifest")
  endif()
  if(NOT revision_row MATCHES "^revision\t([0-9a-f]+)$")
    message(FATAL_ERROR "${owner} source identity revision is invalid")
  endif()
  set(revision "${CMAKE_MATCH_1}")
  if(NOT dirty_row MATCHES "^dirty\t(true|false)$")
    message(FATAL_ERROR "${owner} source identity dirty state is invalid")
  endif()
  set(dirty "${CMAKE_MATCH_1}")
  string(LENGTH "${revision}" revision_length)
  if(NOT revision_length EQUAL 40 AND NOT revision_length EQUAL 64)
    message(FATAL_ERROR "${owner} source identity revision is invalid")
  endif()
  set(${revision_output} "${revision}" PARENT_SCOPE)
  set(${dirty_output} "${dirty}" PARENT_SCOPE)
endfunction()

function(rund_validate_source_identity manifest identity owner)
  rund_read_source_identity(
    "${manifest}" "${identity}" "${owner}"
    ignored_revision ignored_dirty)
endfunction()

function(rund_write_source_identity manifest identity revision dirty owner)
  rund_require_sealed_file("${manifest}" "source manifest" "${owner}")
  string(LENGTH "${revision}" revision_length)
  if((NOT revision_length EQUAL 40 AND NOT revision_length EQUAL 64) OR
     NOT revision MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${owner} source identity revision is invalid")
  endif()
  if(NOT dirty MATCHES "^(true|false)$")
    message(FATAL_ERROR "${owner} source identity dirty state is invalid")
  endif()
  file(SHA256 "${manifest}" manifest_sha256)
  file(WRITE "${identity}"
    "source_manifest_sha256\t${manifest_sha256}\n"
    "revision\t${revision}\n"
    "dirty\t${dirty}\n")
  rund_write_sealed_file("${identity}")
endfunction()

function(rund_artifact_identity_keys triplet output)
  set(common_keys
    build_type
    compiler_id
    compiler_target
    compiler_version
    cxx_standard
    cxx_standard_library
    cxx_standard_library_version
    platform_triplet
    public_compile_definitions
    public_target_sha256
    sdk_version
    target_architecture)
  if(triplet STREQUAL "darwin-arm64")
    set(platform_keys
      apple_sdk_build
      apple_sdk_name
      apple_sdk_version
      foundation_sdk_version
      macos_deployment_target_macro
      metal_sdk_version
      molten_vk_formula
      molten_vk_version
      vulkan_headers_formula
      vulkan_headers_version
      vulkan_loader_formula
      vulkan_loader_version)
  elseif(triplet STREQUAL "linux-x64")
    set(platform_keys glibc_version llvm_version)
  else()
    message(FATAL_ERROR
      "artifact identity schema has unsupported platform triplet: ${triplet}")
  endif()
  set(keys ${common_keys} ${platform_keys})
  list(SORT keys)
  set(${output} "${keys}" PARENT_SCOPE)
endfunction()

function(rund_read_artifact_identity_rows identity owner keys_output)
  file(READ "${identity}" identity_text)
  string(LENGTH "${identity_text}" identity_length)
  if(identity_length EQUAL 0 OR identity_text MATCHES "\r" OR
     identity_text MATCHES "\n\n")
    message(FATAL_ERROR "${owner} artifact identity is not canonical TSV")
  endif()
  math(EXPR identity_last "${identity_length} - 1")
  string(SUBSTRING "${identity_text}" ${identity_last} 1 identity_terminal)
  if(NOT identity_terminal STREQUAL "\n")
    message(FATAL_ERROR
      "${owner} artifact identity lacks a canonical terminal newline")
  endif()

  file(STRINGS "${identity}" identity_rows)
  set(identity_keys)
  foreach(row IN LISTS identity_rows)
    if(NOT row MATCHES
       "^([a-z][a-z0-9_]*)\t([A-Za-z0-9._+:/=-]+)$")
      message(FATAL_ERROR
        "${owner} artifact identity has a malformed row: ${row}")
    endif()
    set(key "${CMAKE_MATCH_1}")
    set(value "${CMAKE_MATCH_2}")
    if(key IN_LIST identity_keys)
      message(FATAL_ERROR
        "${owner} artifact identity repeats key: ${key}")
    endif()
    list(APPEND identity_keys "${key}")
    set("identity_${key}" "${value}" PARENT_SCOPE)
  endforeach()
  set(${keys_output} "${identity_keys}" PARENT_SCOPE)
endfunction()

function(rund_write_artifact_identity
         observation identity expected_version public_target
         expected_triplet owner)
  if(NOT EXISTS "${observation}" OR IS_DIRECTORY "${observation}")
    message(FATAL_ERROR
      "${owner} artifact identity observation is unavailable")
  endif()
  if(NOT EXISTS "${public_target}" OR IS_DIRECTORY "${public_target}")
    message(FATAL_ERROR
      "${owner} artifact identity public target is missing")
  endif()
  rund_read_artifact_identity_rows(
    "${observation}" "${owner} observation" observation_keys)
  rund_artifact_identity_keys("${expected_triplet}" expected_keys)
  set(sorted_observation_keys ${observation_keys})
  list(SORT sorted_observation_keys)
  if(NOT "${sorted_observation_keys}" STREQUAL "${expected_keys}")
    message(FATAL_ERROR
      "${owner} artifact identity observation does not match the schema")
  endif()

  set(identity_text "")
  foreach(key IN LISTS expected_keys)
    string(APPEND identity_text "${key}\t${identity_${key}}\n")
  endforeach()
  file(WRITE "${identity}" "${identity_text}")
  rund_write_sealed_file("${identity}")
  rund_validate_artifact_identity(
    "${identity}" "${expected_version}" "${public_target}"
    "${expected_triplet}" "${owner}")
endfunction()

function(rund_write_artifact_identity_fixture
         identity triplet public_target sdk_version)
  file(SHA256 "${public_target}" public_target_sha256)
  set(fixture_build_type Release)
  set(fixture_cxx_standard 20)
  set(fixture_platform_triplet "${triplet}")
  set(fixture_public_compile_definitions none)
  set(fixture_public_target_sha256 "${public_target_sha256}")
  set(fixture_sdk_version "${sdk_version}")
  if(triplet STREQUAL "darwin-arm64")
    set(fixture_apple_sdk_name macosx)
    set(fixture_apple_sdk_version 1.0)
    set(fixture_compiler_id AppleClang)
    set(fixture_compiler_target arm64-apple-fixture)
    set(fixture_cxx_standard_library libc++)
    set(fixture_foundation_sdk_version 1.0)
    set(fixture_macos_deployment_target_macro 1)
    set(fixture_metal_sdk_version 1.0)
    set(fixture_molten_vk_formula molten-vk)
    set(fixture_target_architecture arm64)
    set(fixture_vulkan_headers_formula vulkan-headers)
    set(fixture_vulkan_loader_formula vulkan-loader)
  elseif(triplet STREQUAL "linux-x64")
    set(fixture_compiler_id GNU)
    set(fixture_compiler_target x86_64-linux-gnu)
    set(fixture_compiler_version 13.1)
    set(fixture_cxx_standard_library libstdc++)
    set(fixture_glibc_version 1.0)
    set(fixture_llvm_version 18.1)
    set(fixture_target_architecture x86_64)
  else()
    message(FATAL_ERROR
      "artifact identity fixture has unsupported platform triplet: ${triplet}")
  endif()

  rund_artifact_identity_keys("${triplet}" fixture_keys)
  set(observation "${identity}.observation")
  set(observation_text "")
  foreach(key IN LISTS fixture_keys)
    if(DEFINED fixture_${key})
      set(value "${fixture_${key}}")
    else()
      set(value 1)
    endif()
    string(APPEND observation_text "${key}\t${value}\n")
  endforeach()
  file(WRITE "${observation}" "${observation_text}")
  rund_write_artifact_identity(
    "${observation}" "${identity}" "${sdk_version}" "${public_target}"
    "${triplet}" fixture)
  file(REMOVE "${observation}")
endfunction()

function(rund_validate_artifact_identity
         identity expected_version public_target expected_triplet owner)
  rund_require_sealed_file("${identity}" "artifact identity" "${owner}")
  if(NOT EXISTS "${public_target}")
    message(FATAL_ERROR "${owner} artifact identity public target is missing")
  endif()

  rund_read_artifact_identity_rows("${identity}" "${owner}" identity_keys)

  if(NOT identity_platform_triplet STREQUAL "darwin-arm64" AND
     NOT identity_platform_triplet STREQUAL "linux-x64")
    message(FATAL_ERROR
      "${owner} artifact identity has unsupported platform triplet: "
      "${identity_platform_triplet}")
  endif()
  rund_artifact_identity_keys(
    "${identity_platform_triplet}" expected_keys)
  set(sorted_keys ${identity_keys})
  list(SORT sorted_keys)
  if(NOT "${identity_keys}" STREQUAL "${sorted_keys}" OR
     NOT "${identity_keys}" STREQUAL "${expected_keys}")
    message(FATAL_ERROR
      "${owner} artifact identity keys are not the exact canonical schema")
  endif()

  if(NOT "${expected_triplet}" STREQUAL "" AND
     NOT identity_platform_triplet STREQUAL expected_triplet)
    message(FATAL_ERROR
      "${owner} artifact identity uses the wrong platform triplet")
  endif()
  if(NOT identity_sdk_version STREQUAL expected_version OR
     NOT identity_build_type STREQUAL "Release" OR
     NOT identity_cxx_standard STREQUAL "20" OR
     NOT identity_public_compile_definitions STREQUAL "none")
    message(FATAL_ERROR
      "${owner} artifact identity contradicts the current SDK tuple")
  endif()

  if(identity_platform_triplet STREQUAL "darwin-arm64")
    if(NOT identity_compiler_id STREQUAL "AppleClang" OR
       NOT identity_cxx_standard_library STREQUAL "libc++" OR
       NOT identity_target_architecture STREQUAL "arm64" OR
       NOT identity_compiler_target MATCHES "^(arm64|aarch64)-apple-" OR
       NOT identity_macos_deployment_target_macro MATCHES "^[0-9]+$" OR
       NOT identity_apple_sdk_name STREQUAL "macosx" OR
       NOT identity_foundation_sdk_version STREQUAL
           identity_apple_sdk_version OR
       NOT identity_metal_sdk_version STREQUAL identity_apple_sdk_version OR
       NOT identity_molten_vk_formula STREQUAL "molten-vk" OR
       NOT identity_vulkan_headers_formula STREQUAL "vulkan-headers" OR
       NOT identity_vulkan_loader_formula STREQUAL "vulkan-loader")
      message(FATAL_ERROR
        "${owner} artifact identity contradicts the Darwin SDK tuple")
    endif()
  else()
    if(NOT identity_compiler_id STREQUAL "GNU" OR
       NOT identity_compiler_version MATCHES "^13[.]" OR
       NOT identity_cxx_standard_library STREQUAL "libstdc++" OR
       NOT identity_target_architecture STREQUAL "x86_64" OR
       NOT identity_compiler_target MATCHES "^x86_64.*-linux-gnu" OR
       NOT identity_llvm_version MATCHES "^18[.]" OR
       NOT identity_glibc_version MATCHES "^[0-9]+[.][0-9]+$")
      message(FATAL_ERROR
        "${owner} artifact identity contradicts the Linux SDK tuple")
    endif()
  endif()

  string(LENGTH "${identity_public_target_sha256}" target_sha256_length)
  if(NOT target_sha256_length EQUAL 64 OR
     NOT identity_public_target_sha256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "${owner} artifact identity has an invalid public target SHA-256")
  endif()
  file(SHA256 "${public_target}" observed_target_sha256)
  if(NOT observed_target_sha256 STREQUAL identity_public_target_sha256)
    message(FATAL_ERROR
      "${owner} public target SHA-256 differs from artifact identity")
  endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  if(DEFINED ARTIFACT_IDENTITY_OUTPUT)
    foreach(required IN ITEMS
        ARTIFACT_OBSERVATION EXPECTED_SDK_VERSION PUBLIC_TARGET
        EXPECTED_TRIPLET)
      if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR
          "artifact identity generation requires ${required}")
      endif()
    endforeach()
    if(NOT DEFINED IDENTITY_OWNER)
      set(IDENTITY_OWNER "producer")
    endif()
    rund_write_artifact_identity(
      "${ARTIFACT_OBSERVATION}" "${ARTIFACT_IDENTITY_OUTPUT}"
      "${EXPECTED_SDK_VERSION}" "${PUBLIC_TARGET}"
      "${EXPECTED_TRIPLET}" "${IDENTITY_OWNER}")
  endif()

  if(DEFINED IDENTITY)
    foreach(required IN ITEMS EXPECTED_SDK_VERSION PUBLIC_TARGET)
      if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "artifact identity validation requires ${required}")
      endif()
    endforeach()
    if(NOT DEFINED EXPECTED_TRIPLET)
      set(EXPECTED_TRIPLET "")
    endif()
    if(NOT DEFINED IDENTITY_OWNER)
      set(IDENTITY_OWNER "producer")
    endif()
    rund_validate_artifact_identity(
      "${IDENTITY}" "${EXPECTED_SDK_VERSION}" "${PUBLIC_TARGET}"
      "${EXPECTED_TRIPLET}" "${IDENTITY_OWNER}")
  endif()

  if(DEFINED SOURCE_IDENTITY_OUTPUT)
    foreach(required IN ITEMS SOURCE_MANIFEST SOURCE_REVISION SOURCE_DIRTY)
      if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR
          "source identity generation requires ${required}")
      endif()
    endforeach()
    if(NOT DEFINED SOURCE_OWNER)
      set(SOURCE_OWNER "producer")
    endif()
    rund_write_sealed_file("${SOURCE_MANIFEST}")
    rund_write_source_identity(
      "${SOURCE_MANIFEST}" "${SOURCE_IDENTITY_OUTPUT}"
      "${SOURCE_REVISION}" "${SOURCE_DIRTY}" "${SOURCE_OWNER}")
  elseif(DEFINED SOURCE_MANIFEST)
    if(NOT DEFINED SOURCE_IDENTITY OR "${SOURCE_IDENTITY}" STREQUAL "")
      message(FATAL_ERROR "source identity validation requires SOURCE_IDENTITY")
    endif()
    if(NOT DEFINED SOURCE_OWNER)
      set(SOURCE_OWNER "producer")
    endif()
    rund_validate_source_identity(
      "${SOURCE_MANIFEST}" "${SOURCE_IDENTITY}" "${SOURCE_OWNER}")
  endif()

  if(DEFINED INSTALLED_PROVENANCE)
    foreach(required IN ITEMS EXPECTED_PROVENANCE PROVENANCE_LABEL)
      if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "exact provenance validation requires ${required}")
      endif()
    endforeach()
    rund_require_exact_provenance(
      "${INSTALLED_PROVENANCE}" "${EXPECTED_PROVENANCE}"
      "${PROVENANCE_LABEL}")
  endif()
endif()

cmake_policy(POP)
