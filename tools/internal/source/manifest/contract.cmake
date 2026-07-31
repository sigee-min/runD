if(NOT DEFINED ROOT)
  message(FATAL_ERROR "source-manifest contract requires ROOT")
endif()

execute_process(
  COMMAND sh "${ROOT}/tools/source/manifest" first-output second-output
  WORKING_DIRECTORY "${ROOT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)

if(NOT result EQUAL 64)
  message(FATAL_ERROR
          "source-manifest accepted two output paths: exit=${result}")
endif()
if(NOT output STREQUAL "")
  message(FATAL_ERROR "source-manifest usage failure wrote stdout")
endif()
if(NOT error STREQUAL "usage: tools/source/manifest [output-path]\n")
  message(FATAL_ERROR
          "source-manifest usage failure changed diagnostic: ${error}")
endif()

string(SHA256 fixture_id "${CMAKE_BINARY_DIR}")
set(fixture "${ROOT}/.cache/source-manifest-contract/${fixture_id}")
set(verified "${fixture}/verified-after.tsv")
set(EVIDENCE_STAGING "${fixture}/packet")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${fixture}")

set(live_manifest "$ENV{RUND_VERIFIED_SOURCE_MANIFEST}")
if(live_manifest STREQUAL "")
  # Direct CTest use has no verification boundary to borrow. Keep that
  # standalone contract useful by taking its own live-root snapshot.
  set(live_manifest "${fixture}/live-root.tsv")
  execute_process(
    COMMAND "${ROOT}/tools/source/manifest" "${live_manifest}"
    WORKING_DIRECTORY "${ROOT}"
    RESULT_VARIABLE live_manifest_result
    ERROR_VARIABLE live_manifest_error)
  if(NOT live_manifest_result EQUAL 0)
    message(FATAL_ERROR
      "source-manifest rejected the registered live root: ${live_manifest_error}")
  endif()
elseif(NOT IS_ABSOLUTE "${live_manifest}" OR NOT EXISTS "${live_manifest}" OR
       IS_DIRECTORY "${live_manifest}")
  message(FATAL_ERROR
    "verified source-manifest input is unavailable: ${live_manifest}")
endif()
file(READ "${live_manifest}" live_manifest_text)
string(FIND "${live_manifest_text}"
  "docs/architecture/root/layout.tsv" layout_manifest_index)
if(layout_manifest_index EQUAL -1)
  message(FATAL_ERROR "source-manifest omitted its root-layout authority")
endif()
# Root membership is consumed from one registry. An unregistered physical root
# entry and a missing admitted owner must both fail closed.
set(manifest_root "${fixture}/manifest-root")
file(MAKE_DIRECTORY
  "${manifest_root}/tools/source"
  "${manifest_root}/docs/architecture/root"
  "${manifest_root}/.cache")
file(COPY "${ROOT}/tools/source/manifest"
  DESTINATION "${manifest_root}/tools/source")
file(CHMOD "${manifest_root}/tools/source/manifest"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)
file(WRITE "${manifest_root}/keep.txt" "registered\n")
file(CHMOD "${manifest_root}/keep.txt"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
string(CONCAT manifest_layout
  "admitted-file\tkeep.txt\n"
  "admitted-dir\tdocs\n"
  "admitted-dir\ttools\n"
  "local-dir\t.cache\n"
  "debris\t.DS_Store\n"
  "debris\tThumbs.db\n")
file(WRITE "${manifest_root}/docs/architecture/root/layout.tsv"
  "${manifest_layout}")
file(WRITE "${manifest_root}/docs/.DS_Store" "platform debris\n")
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/registered.tsv"
  RESULT_VARIABLE registered_result
  ERROR_VARIABLE registered_error)
if(NOT registered_result EQUAL 0)
  message(FATAL_ERROR
    "source-manifest registry fixture failed: ${registered_error}")
endif()
file(READ "${manifest_root}/.cache/registered.tsv" registered_manifest)
string(FIND "${registered_manifest}" "\t-\tkeep.txt" nonexec_mode_index)
if(nonexec_mode_index EQUAL -1)
  message(FATAL_ERROR "source-manifest omitted a non-executable mode identity")
endif()
string(FIND "${registered_manifest}" "docs/.DS_Store" debris_index)
if(NOT debris_index EQUAL -1)
  message(FATAL_ERROR "source-manifest included registered platform debris")
endif()
file(CHMOD "${manifest_root}/keep.txt"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ WORLD_READ)
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/executable.tsv"
  RESULT_VARIABLE executable_result
  ERROR_VARIABLE executable_error)
if(NOT executable_result EQUAL 0)
  message(FATAL_ERROR
    "source-manifest executable-mode fixture failed: ${executable_error}")
endif()
file(READ "${manifest_root}/.cache/executable.tsv" executable_manifest)
string(FIND "${executable_manifest}" "\tx\tkeep.txt" exec_mode_index)
if(exec_mode_index EQUAL -1 OR executable_manifest STREQUAL registered_manifest)
  message(FATAL_ERROR "source-manifest ignored an executable-bit mutation")
endif()
file(CHMOD "${manifest_root}/keep.txt"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E create_symlink
          "${manifest_root}/keep.txt" "${manifest_root}/docs/source-link"
  RESULT_VARIABLE symlink_create_result)
if(NOT symlink_create_result EQUAL 0)
  message(FATAL_ERROR "source-manifest symlink fixture creation failed")
endif()
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/symlink.tsv"
  RESULT_VARIABLE symlink_result
  ERROR_VARIABLE symlink_error)
if(symlink_result EQUAL 0 OR NOT symlink_error MATCHES
   "unsupported entries under admitted repository roots")
  message(FATAL_ERROR "source-manifest silently omitted an admitted symlink")
endif()
file(REMOVE "${manifest_root}/docs/source-link")

file(MAKE_DIRECTORY "${manifest_root}/docs/__pycache__")
file(WRITE "${manifest_root}/docs/__pycache__/probe.pyc" "generated cache\n")
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/generated.tsv"
  RESULT_VARIABLE generated_result
  ERROR_VARIABLE generated_error)
file(REMOVE_RECURSE "${manifest_root}/docs/__pycache__")
if(NOT generated_result EQUAL 0)
  message(FATAL_ERROR
    "source-manifest rejected an excluded Python cache: ${generated_error}")
endif()
file(READ "${manifest_root}/.cache/generated.tsv" generated_manifest)
string(FIND "${generated_manifest}" "__pycache__" generated_cache_index)
string(FIND "${generated_manifest}" "probe.pyc" generated_file_index)
if(NOT generated_cache_index EQUAL -1 OR NOT generated_file_index EQUAL -1)
  message(FATAL_ERROR "source-manifest included an excluded Python cache")
endif()

function(expect_rejected_manifest_spelling spelling label)
  set(path "${manifest_root}/docs/${spelling}")
  file(WRITE "${path}" "unsupported path spelling\n")
  execute_process(
    COMMAND "${manifest_root}/tools/source/manifest"
            "${manifest_root}/.cache/${label}.tsv"
    RESULT_VARIABLE spelling_result
    ERROR_VARIABLE spelling_error)
  file(REMOVE "${path}")
  if(spelling_result EQUAL 0 OR NOT spelling_error MATCHES
     "unsupported repository path spelling")
    message(FATAL_ERROR
      "source-manifest accepted a ${label} path spelling: ${spelling_error}")
  endif()
endfunction()
expect_rejected_manifest_spelling("tab\tname" "tab")
expect_rejected_manifest_spelling("line\nname" "newline")
expect_rejected_manifest_spelling([=[back\slash]=] "backslash")

find_program(source_manifest_real_find NAMES find REQUIRED)
set(find_bin "${manifest_root}/.cache/find-bin")
file(MAKE_DIRECTORY "${find_bin}")
file(WRITE "${find_bin}/find"
  "#!/bin/sh\n"
  "if [ \"\$1\" = docs ]; then\n"
  "  echo simulated-find-traversal-failure >&2\n"
  "  exit 7\n"
  "fi\n"
  "exec \"${source_manifest_real_find}\" \"\$@\"\n")
file(CHMOD "${find_bin}/find"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
          "PATH=${find_bin}:$ENV{PATH}"
          "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/traversal.tsv"
  RESULT_VARIABLE traversal_result
  ERROR_VARIABLE traversal_error)
if(traversal_result EQUAL 0 OR NOT traversal_error MATCHES
   "simulated-find-traversal-failure")
  message(FATAL_ERROR
    "source-manifest masked a traversal failure: ${traversal_error}")
endif()

file(MAKE_DIRECTORY "${manifest_root}/rogue")
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/rogue.tsv"
  RESULT_VARIABLE rogue_result
  ERROR_VARIABLE rogue_error)
if(rogue_result EQUAL 0 OR
   NOT rogue_error MATCHES "unadmitted repository root entries")
  message(FATAL_ERROR "source-manifest accepted an unregistered root entry")
endif()
file(REMOVE_RECURSE "${manifest_root}/rogue")
file(REMOVE "${manifest_root}/keep.txt")
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest"
          "${manifest_root}/.cache/missing.tsv"
  RESULT_VARIABLE missing_result
  ERROR_VARIABLE missing_error)
if(missing_result EQUAL 0 OR
   NOT missing_error MATCHES "admitted repository root file is missing")
  message(FATAL_ERROR "source-manifest accepted a missing admitted owner")
endif()
file(WRITE "${manifest_root}/keep.txt" "registered\n")

string(CONCAT fixture_manifest
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\tdocs/a.md\n"
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\ttools/b\n")
file(WRITE "${verified}" "${fixture_manifest}")
include("${ROOT}/package/cmake/identity.cmake")
rund_write_sealed_file("${verified}")
file(SHA256 "${verified}" fixture_sha256)
set(fixture_revision "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
rund_write_source_identity(
  "${verified}" "${verified}.identity.tsv"
  "${fixture_revision}" true fixture)
file(READ "${verified}.identity.tsv" fixture_identity)
file(SHA256 "${verified}.identity.tsv" fixture_identity_sha256)

set(SOURCE_MANIFEST "${verified}")
include("${ROOT}/tools/internal/source/manifest/adopt.cmake")
if(NOT recorded_source_manifest STREQUAL
       "${EVIDENCE_STAGING}/source-manifest.tsv")
  message(FATAL_ERROR "recorded manifest path does not name packet artifact")
endif()
file(READ "${recorded_source_manifest}" recorded_manifest)
if(NOT recorded_manifest STREQUAL fixture_manifest)
  message(FATAL_ERROR "recorded manifest is not the verified after manifest")
endif()
if(NOT source_manifest_sha256 STREQUAL fixture_sha256)
  message(FATAL_ERROR "recorded manifest SHA does not name packet artifact")
endif()
if(NOT source_revision STREQUAL fixture_revision OR
   NOT source_dirty STREQUAL "true" OR
   NOT source_identity_sha256 STREQUAL fixture_identity_sha256)
  message(FATAL_ERROR "recorded source identity differs from verification")
endif()
file(READ "${recorded_source_identity}" recorded_identity)
if(NOT recorded_identity STREQUAL fixture_identity)
  message(FATAL_ERROR "recorded source identity is not the verified identity")
endif()

# The manifest and Git identity must describe one bracketed boundary. Model a
# revision mutation and a porcelain mutation independently with a deterministic
# fake Git executable.
set(fake_bin "${manifest_root}/.cache/bin")
file(MAKE_DIRECTORY "${fake_bin}")
file(WRITE "${fake_bin}/git" [=[#!/bin/sh
set -eu
self=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mode=$(cat "$self/mode")
case "$*" in
  *"rev-parse HEAD"*)
    if [ "$mode" = revision ]; then
      count=0
      if [ -f "$self/revision.count" ]; then count=$(cat "$self/revision.count"); fi
      count=$((count + 1))
      printf '%s\n' "$count" >"$self/revision.count"
      if [ "$count" -eq 1 ]; then
        printf '%040d\n' 1
      else
        printf '%040d\n' 2
      fi
    else
      printf '%040d\n' 1
    fi
    ;;
  *"status --porcelain=v1"*)
    if [ "$mode" = status ]; then
      count=0
      if [ -f "$self/status.count" ]; then count=$(cat "$self/status.count"); fi
      count=$((count + 1))
      printf '%s\n' "$count" >"$self/status.count"
      if [ "$count" -gt 1 ]; then printf ' M keep.txt\0'; fi
    fi
    ;;
  *) exit 2 ;;
esac
]=])
file(CHMOD "${fake_bin}/git"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)
set(bracket_before "${manifest_root}/.cache/bracket-before.tsv")
execute_process(
  COMMAND "${manifest_root}/tools/source/manifest" "${bracket_before}"
  RESULT_VARIABLE bracket_before_result)
if(NOT bracket_before_result EQUAL 0)
  message(FATAL_ERROR "source identity bracket fixture manifest failed")
endif()
file(WRITE "${fake_bin}/mode" "stable\n")
set(bracket_stable "${manifest_root}/.cache/bracket-stable.tsv")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
          "PATH=${fake_bin}:$ENV{PATH}"
          sh "${ROOT}/tools/internal/source/verify"
          "${manifest_root}" "${bracket_before}" "${bracket_stable}"
  RESULT_VARIABLE stable_result
  OUTPUT_VARIABLE stable_output
  ERROR_VARIABLE stable_error)
if(NOT stable_result EQUAL 0)
  message(FATAL_ERROR
    "stable source identity fixture failed: ${stable_output}${stable_error}")
endif()
rund_read_source_identity(
  "${bracket_stable}" "${bracket_stable}.identity.tsv" fixture
  bracket_revision bracket_dirty)
if(NOT bracket_revision STREQUAL
   "0000000000000000000000000000000000000001" OR
   NOT bracket_dirty STREQUAL "false")
  message(FATAL_ERROR "stable source identity fixture changed its facts")
endif()
foreach(mutation IN ITEMS revision status)
  file(WRITE "${fake_bin}/mode" "${mutation}\n")
  file(REMOVE "${fake_bin}/revision.count" "${fake_bin}/status.count")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${fake_bin}:$ENV{PATH}"
            sh "${ROOT}/tools/internal/source/verify"
            "${manifest_root}" "${bracket_before}"
            "${manifest_root}/.cache/bracket-${mutation}.tsv"
    RESULT_VARIABLE bracket_result
    ERROR_VARIABLE bracket_error)
  if(bracket_result EQUAL 0 OR NOT bracket_error MATCHES
     "source identity changed during source stability check")
    message(FATAL_ERROR
      "source stability accepted a ${mutation} mutation: ${bracket_error}")
  endif()
endforeach()

# The verification-state owner must consume Ninja dependency output once
# without losing either producer failure or zero-dependency diagnostics.
set(state_build "${fixture}/state")
file(MAKE_DIRECTORY "${state_build}")
file(WRITE "${state_build}/ninja" [=[#!/bin/sh
set -eu
self=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -ne 4 ] || [ "$1" != -C ] || [ "$2" != "$self" ] ||
   [ "$3" != -t ] || [ "$4" != deps ]; then
  exit 2
fi
case "$(cat "$self/mode")" in
  healthy)
    printf '%s\n' \
      'alpha.o: #deps 2, deps mtime 1 (VALID)' \
      '    alpha.cpp' \
      '    alpha.hpp' \
      '' \
      'beta.obj: #deps 1, deps mtime 1 (VALID)' \
      '    beta.cpp'
    ;;
  empty)
    printf '%s\n' 'non-object dependency noise'
    ;;
  missing)
    printf '%s\n' \
      'healthy.o: #deps 1, deps mtime 1 (VALID)' \
      '    healthy.cpp' \
      '' \
      'missing.obj: #deps 0, deps mtime 0 (STALE)' \
      '' \
      'also.o: #deps 0, deps mtime 0 (STALE)'
    ;;
  failure)
    printf '%s\n' 'partial.o: #deps 1, deps mtime 1 (VALID)'
    exit 7
    ;;
  *) exit 2 ;;
esac
]=])
file(CHMOD "${state_build}/ninja"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)

function(expect_state_verify mode expected_error)
  file(WRITE "${state_build}/mode" "${mode}\n")
  file(REMOVE "${state_build}/.rund-ninja-deps.txt")
  execute_process(
    COMMAND sh "${ROOT}/tools/internal/state/deps"
            "${state_build}" "${state_build}/ninja"
    RESULT_VARIABLE state_result
    OUTPUT_VARIABLE state_output
    ERROR_VARIABLE state_error)
  if(expected_error STREQUAL "")
    if(NOT state_result EQUAL 0 OR NOT state_output STREQUAL "" OR
       NOT state_error STREQUAL "")
      message(FATAL_ERROR
        "healthy Ninja dependency stream failed: "
        "${state_output}${state_error}")
    endif()
    file(READ "${state_build}/.rund-ninja-deps.txt" state_deps)
    string(CONCAT expected_deps
      "alpha.o: #deps 2, deps mtime 1 (VALID)\n"
      "beta.obj: #deps 1, deps mtime 1 (VALID)\n")
    if(NOT state_deps STREQUAL expected_deps)
      message(FATAL_ERROR
        "Ninja dependency stream retained non-object rows: ${state_deps}")
    endif()
  elseif(state_result EQUAL 0 OR
         NOT state_output STREQUAL "" OR
         NOT state_error STREQUAL "${expected_error}")
    message(FATAL_ERROR
      "Ninja dependency ${mode} failure was masked or misreported: "
      "${state_output}${state_error}")
  endif()
  file(GLOB state_residue
    "${state_build}/.rund-ninja-deps.txt.pending.*"
    "${state_build}/.rund-ninja-deps.txt.missing.*"
    "${state_build}/.rund-ninja-deps.txt.status.*")
  if(state_residue)
    message(FATAL_ERROR
      "Ninja dependency ${mode} left temporary state: ${state_residue}")
  endif()
endfunction()

expect_state_verify(healthy "")
expect_state_verify(empty
  "Ninja dependency audit found no compiled objects\n")
expect_state_verify(missing
  "Ninja dependency audit found objects without dependencies:\nmissing.obj: #deps 0, deps mtime 0 (STALE)\nalso.o: #deps 0, deps mtime 0 (STALE)\n")
expect_state_verify(failure "Ninja dependency audit failed\n")

# A post-verification mutation must fail closed instead of being copied into a
# packet under the preceding seal.
file(APPEND "${verified}" "changed\tproduct-source\n")
set(rejected "${fixture}/rejected")
set(driver "${fixture}/reject-mutated.cmake")
file(WRITE "${driver}"
  "set(ROOT [==[${ROOT}]==])\n"
  "set(SOURCE_MANIFEST [==[${verified}]==])\n"
  "set(EVIDENCE_STAGING [==[${rejected}]==])\n"
  "include([==[${ROOT}/tools/internal/source/manifest/adopt.cmake]==])\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${driver}"
  RESULT_VARIABLE rejected_result
  ERROR_VARIABLE rejected_error)
if(rejected_result EQUAL 0)
  message(FATAL_ERROR "record accepted a manifest changed after verification")
endif()
if(NOT rejected_error MATCHES
   "verified source manifest provenance seal does not match its payload")
  message(FATAL_ERROR
          "record changed post-verification rejection: ${rejected_error}")
endif()
file(READ "${recorded_source_manifest}" recorded_after_mutation)
if(NOT recorded_after_mutation STREQUAL fixture_manifest)
  message(FATAL_ERROR "recorded manifest changed with its verified source")
endif()

file(WRITE "${verified}" "${fixture_manifest}")
file(APPEND "${verified}.identity.tsv" "changed\tidentity\n")
set(identity_rejected "${fixture}/identity-rejected")
set(identity_driver "${fixture}/reject-mutated-identity.cmake")
file(WRITE "${identity_driver}"
  "set(ROOT [==[${ROOT}]==])\n"
  "set(SOURCE_MANIFEST [==[${verified}]==])\n"
  "set(EVIDENCE_STAGING [==[${identity_rejected}]==])\n"
  "include([==[${ROOT}/tools/internal/source/manifest/adopt.cmake]==])\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${identity_driver}"
  RESULT_VARIABLE identity_rejected_result
  ERROR_VARIABLE identity_rejected_error)
if(identity_rejected_result EQUAL 0 OR NOT identity_rejected_error MATCHES
   "verified source identity provenance seal does not match its payload")
  message(FATAL_ERROR
          "record accepted or misreported a mutated source identity")
endif()
file(READ "${recorded_source_identity}" recorded_identity_after_mutation)
if(NOT recorded_identity_after_mutation STREQUAL fixture_identity)
  message(FATAL_ERROR "recorded identity changed with its verified source")
endif()

set(measure_fixture "${fixture}/measure")
set(measure_fixture_root "${measure_fixture}/root")
set(measure_fixture_build "${measure_fixture}/build")
set(measure_fixture_release "${measure_fixture}/release")
file(MAKE_DIRECTORY
  "${measure_fixture_root}/tools/internal/state"
  "${measure_fixture_release}/runD-install/lib/cmake/runD"
  "${measure_fixture_build}")
file(WRITE "${measure_fixture_root}/tools/internal/state/root"
  "#!/bin/sh\n"
  "[ \"$2\" = release ] || exit 2\n"
  "printf '%s\\n' \"${measure_fixture_release}\"\n")
file(WRITE "${measure_fixture_root}/tools/internal/state/ninja"
  "#!/bin/sh\nexit 19\n")
file(CHMOD
  "${measure_fixture_root}/tools/internal/state/root"
  "${measure_fixture_root}/tools/internal/state/ninja"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
file(WRITE
  "${measure_fixture_release}/runD-install/lib/cmake/runD/runDConfig.cmake"
  "# measurement fixture\n")
set(measure_sentinel "${measure_fixture_build}/retained")
file(WRITE "${measure_sentinel}" "retained\n")
execute_process(
  COMMAND sh "${ROOT}/tools/internal/measure/prepare"
          "${measure_fixture_root}" "${measure_fixture_build}"
          "${measure_fixture}/source"
  RESULT_VARIABLE measure_missing_driver_result
  OUTPUT_VARIABLE measure_missing_driver_output
  ERROR_VARIABLE measure_missing_driver_error)
if(measure_missing_driver_result EQUAL 0 OR
   NOT measure_missing_driver_output STREQUAL "" OR
   NOT measure_missing_driver_error STREQUAL
       "installed-package measurement requires Ninja\n" OR
   NOT EXISTS "${measure_sentinel}")
  message(FATAL_ERROR
    "measurement missing-driver admission changed or deleted its prior tree: "
    "exit=${measure_missing_driver_result}\n"
    "stdout=${measure_missing_driver_output}\n"
    "stderr=${measure_missing_driver_error}")
endif()

file(REMOVE_RECURSE "${fixture}")
