foreach(required IN ITEMS ROOT SOURCE_MANIFEST EVIDENCE_STAGING)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "source-manifest adoption requires ${required}")
  endif()
endforeach()

set(_rund_identity "${SOURCE_MANIFEST}.identity.tsv")
include("${ROOT}/package/cmake/identity.cmake")
rund_read_source_identity(
  "${SOURCE_MANIFEST}" "${_rund_identity}" "verified"
  source_revision source_dirty)
file(SHA256 "${SOURCE_MANIFEST}" _rund_manifest_expected)
file(SHA256 "${_rund_identity}" _rund_identity_expected)

file(MAKE_DIRECTORY "${EVIDENCE_STAGING}")
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _rund_manifest_nonce)
set(_rund_manifest_recorded
    "${EVIDENCE_STAGING}/source-manifest.tsv")
set(_rund_manifest_temp
    "${EVIDENCE_STAGING}/.source-manifest-${_rund_manifest_nonce}.tmp")
set(_rund_identity_recorded "${EVIDENCE_STAGING}/source-identity.tsv")
set(_rund_identity_temp
    "${EVIDENCE_STAGING}/.source-identity-${_rund_manifest_nonce}.tmp")
configure_file("${SOURCE_MANIFEST}" "${_rund_manifest_temp}" COPYONLY)
configure_file("${_rund_identity}" "${_rund_identity_temp}" COPYONLY)
file(SHA256 "${_rund_manifest_temp}" _rund_manifest_copied)
file(SHA256 "${_rund_identity_temp}" _rund_identity_copied)
if(NOT _rund_manifest_copied STREQUAL _rund_manifest_expected)
  file(REMOVE "${_rund_manifest_temp}" "${_rund_identity_temp}")
  message(FATAL_ERROR
          "verified source manifest changed while evidence was recorded")
endif()
if(NOT _rund_identity_copied STREQUAL _rund_identity_expected)
  file(REMOVE "${_rund_manifest_temp}" "${_rund_identity_temp}")
  message(FATAL_ERROR
          "verified source identity changed while evidence was recorded")
endif()
file(RENAME "${_rund_manifest_temp}" "${_rund_manifest_recorded}")
file(RENAME "${_rund_identity_temp}" "${_rund_identity_recorded}")

set(recorded_source_manifest "${_rund_manifest_recorded}")
set(source_manifest_sha256 "${_rund_manifest_copied}")
set(recorded_source_identity "${_rund_identity_recorded}")
set(source_identity_sha256 "${_rund_identity_copied}")
