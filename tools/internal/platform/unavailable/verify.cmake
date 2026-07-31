foreach(required IN ITEMS ROOT BUILD PHASE VULKAN_ENABLED OBJECTS_REQUIRED REPORT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR
      "platform-unavailable verification requires ${required}")
  endif()
endforeach()

if(NOT PHASE MATCHES "^[a-z][a-z0-9-]*$" OR
   NOT VULKAN_ENABLED MATCHES "^(ON|OFF)$" OR
   NOT OBJECTS_REQUIRED MATCHES "^(ON|OFF)$")
  message(FATAL_ERROR "platform-unavailable verification input is malformed")
endif()

get_filename_component(root "${ROOT}" REALPATH)
get_filename_component(build "${BUILD}" REALPATH)
execute_process(
  COMMAND sh "${root}/tools/internal/state/root"
    "${root}" platform-unavailable
  RESULT_VARIABLE root_result
  OUTPUT_VARIABLE expected_build
  ERROR_VARIABLE root_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT root_result EQUAL 0)
  message(FATAL_ERROR
    "platform-unavailable root lookup failed: ${root_error}")
endif()
if(NOT build STREQUAL expected_build)
  message(FATAL_ERROR
    "platform-unavailable verification does not own build tree: ${build}")
endif()

set(cache "${build}/CMakeCache.txt")
set(commands_path "${build}/compile_commands.json")
foreach(required_file IN ITEMS "${cache}" "${commands_path}")
  if(NOT EXISTS "${required_file}" OR IS_DIRECTORY "${required_file}")
    message(FATAL_ERROR
      "platform-unavailable verification input is missing: ${required_file}")
  endif()
endforeach()

function(require_cache_bool name expected)
  file(STRINGS "${cache}" rows REGEX "^${name}:BOOL=")
  list(LENGTH rows count)
  if(NOT count EQUAL 1 OR NOT rows STREQUAL "${name}:BOOL=${expected}")
    message(FATAL_ERROR
      "platform-unavailable cache expected ${name}:BOOL=${expected}, got ${rows}")
  endif()
endfunction()

require_cache_bool(RUND_FORCE_UNAVAILABLE_PLATFORM ON)
require_cache_bool(RUND_ENABLE_VULKAN "${VULKAN_ENABLED}")
require_cache_bool(RUND_TEST_NODE ON)
require_cache_bool(RUND_STRICT_WARNINGS ON)

file(STRINGS "${cache}" focus_rows
  REGEX "^RUND_NODE_FOCUSED_CASE:[A-Z]+=")
list(LENGTH focus_rows focus_count)
if(NOT focus_count EQUAL 1 OR
   NOT focus_rows MATCHES
     "^RUND_NODE_FOCUSED_CASE:[A-Z]+=runtime\\.platform-adapter$")
  message(FATAL_ERROR
    "platform-unavailable cache lost the exact focused case: ${focus_rows}")
endif()

function(fragment_sources fragment expression out)
  file(READ "${fragment}" contents)
  string(REGEX MATCHALL "${expression}" sources "${contents}")
  list(REMOVE_DUPLICATES sources)
  list(SORT sources)
  if(NOT sources)
    message(FATAL_ERROR "platform source fragment has no matching sources")
  endif()
  set(${out} "${sources}" PARENT_SCOPE)
endfunction()

fragment_sources(
  "${root}/node/cmake/node/sources/platform.cmake"
  "src/runtime/platform/unavailable/[A-Za-z0-9_./-]+\\.cpp"
  expected_platform_sources)
fragment_sources(
  "${root}/node/cmake/node/sources/accel/vulkan.cmake"
  "src/accel/vulkan/[A-Za-z0-9_./-]+\\.cpp"
  expected_vulkan_sources)

file(READ "${commands_path}" commands_json)
string(JSON command_count LENGTH "${commands_json}")
if(command_count EQUAL 0)
  message(FATAL_ERROR "platform-unavailable compile database is empty")
endif()
math(EXPR last_command "${command_count} - 1")

set(actual_platform_sources)
set(actual_vulkan_sources)
set(platform_object_count 0)
set(vulkan_object_count 0)
set(runtime_test_entry_count 0)
set(runtime_test_object_count 0)
set(vulkan_sdk_definition false)
set(glslang_definition false)
set(spirv_definition false)

foreach(index RANGE 0 ${last_command})
  string(JSON source GET "${commands_json}" ${index} file)
  string(JSON command GET "${commands_json}" ${index} command)
  string(JSON output ERROR_VARIABLE output_error
    GET "${commands_json}" ${index} output)
  if(NOT output_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
      "platform-unavailable compile entry ${index} has no object output")
  endif()

  if(command MATCHES "RUND_NODE_HAVE_VULKAN_SDK")
    set(vulkan_sdk_definition true)
  endif()
  if(command MATCHES "RUND_NODE_[A-Z0-9_]*GLSLANG[A-Z0-9_]*")
    set(glslang_definition true)
  endif()
  if(command MATCHES "RUND_NODE_[A-Z0-9_]*SPIRV[A-Z0-9_]*")
    set(spirv_definition true)
  endif()

  if(source STREQUAL
     "${root}/node/tests/contract/runtime/platform/adapter.cpp")
    math(EXPR runtime_test_entry_count "${runtime_test_entry_count} + 1")
    if(NOT output MATCHES
       "/node/CMakeFiles/node-runtime\\.dir/tests/contract/runtime/platform/adapter\\.cpp\\.o$")
      message(FATAL_ERROR
        "platform-adapter case is compiled by a non-canonical target: ${output}")
    endif()
    if(NOT command MATCHES "RUND_NODE_PLATFORM_UNAVAILABLE=1")
      message(FATAL_ERROR
        "platform-adapter case would not execute its unavailable-owner body")
    endif()
    if(EXISTS "${output}" AND NOT IS_DIRECTORY "${output}")
      math(EXPR runtime_test_object_count
        "${runtime_test_object_count} + 1")
    endif()
  endif()

  set(owner)
  if(output MATCHES
     "/node/CMakeFiles/node-object-platform\\.dir/")
    set(owner platform)
  elseif(output MATCHES
     "/node/CMakeFiles/node-object-accel-vulkan\\.dir/")
    set(owner vulkan)
  endif()
  if(NOT owner)
    continue()
  endif()

  if(NOT command MATCHES "RUND_NODE_PLATFORM_UNAVAILABLE=1")
    message(FATAL_ERROR
      "${owner} object is missing unavailable-platform selection: ${source}")
  endif()
  file(RELATIVE_PATH relative_source "${root}/node" "${source}")
  if(relative_source MATCHES "^\\.\\./")
    message(FATAL_ERROR
      "${owner} object source is outside Node ownership: ${source}")
  endif()

  if(owner STREQUAL platform)
    list(APPEND actual_platform_sources "${relative_source}")
    if(EXISTS "${output}" AND NOT IS_DIRECTORY "${output}")
      math(EXPR platform_object_count "${platform_object_count} + 1")
    endif()
  else()
    list(APPEND actual_vulkan_sources "${relative_source}")
    if(EXISTS "${output}" AND NOT IS_DIRECTORY "${output}")
      math(EXPR vulkan_object_count "${vulkan_object_count} + 1")
    endif()
  endif()
endforeach()

list(REMOVE_DUPLICATES actual_platform_sources)
list(REMOVE_DUPLICATES actual_vulkan_sources)
list(SORT actual_platform_sources)
list(SORT actual_vulkan_sources)
if(NOT actual_platform_sources STREQUAL expected_platform_sources)
  message(FATAL_ERROR
    "selected platform object sources are not the unavailable owner\n"
    "expected=${expected_platform_sources}\nactual=${actual_platform_sources}")
endif()
if(NOT actual_vulkan_sources STREQUAL expected_vulkan_sources)
  message(FATAL_ERROR
    "Vulkan object target does not own the canonical Vulkan source fragment\n"
    "expected=${expected_vulkan_sources}\nactual=${actual_vulkan_sources}")
endif()

list(LENGTH expected_platform_sources expected_platform_count)
list(LENGTH expected_vulkan_sources expected_vulkan_count)
if(NOT runtime_test_entry_count EQUAL 1)
  message(FATAL_ERROR
    "platform-adapter exact test must have one compile owner: "
    "${runtime_test_entry_count}")
endif()
if(OBJECTS_REQUIRED)
  if(NOT platform_object_count EQUAL expected_platform_count)
    message(FATAL_ERROR
      "platform object owner was not completely built: "
      "${platform_object_count}/${expected_platform_count}")
  endif()
  if(NOT vulkan_object_count EQUAL expected_vulkan_count)
    message(FATAL_ERROR
      "Vulkan object owner was not completely built: "
      "${vulkan_object_count}/${expected_vulkan_count}")
  endif()
  if(NOT runtime_test_object_count EQUAL 1)
    message(FATAL_ERROR
      "platform-adapter exact test object was not built")
  endif()
endif()

if(NOT VULKAN_ENABLED)
  if(vulkan_sdk_definition OR glslang_definition OR spirv_definition)
    message(FATAL_ERROR
      "Vulkan OFF retained SDK, glslang, or SPIR-V compile definitions")
  endif()
endif()

file(APPEND "${REPORT}"
  "phase\t${PHASE}\n"
  "vulkan_enabled\t${VULKAN_ENABLED}\n"
  "objects_required\t${OBJECTS_REQUIRED}\n"
  "platform_source_count\t${expected_platform_count}\n"
  "platform_object_count\t${platform_object_count}\n"
  "vulkan_source_count\t${expected_vulkan_count}\n"
  "vulkan_object_count\t${vulkan_object_count}\n"
  "runtime_test_entry_count\t${runtime_test_entry_count}\n"
  "runtime_test_object_count\t${runtime_test_object_count}\n"
  "vulkan_sdk_definition\t${vulkan_sdk_definition}\n"
  "glslang_definition\t${glslang_definition}\n"
  "spirv_definition\t${spirv_definition}\n")
