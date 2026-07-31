cmake_minimum_required(VERSION 3.20)
project(rund_package_consumer LANGUAGES CXX)

if(NOT DEFINED RUND_EXPECTED_PACKAGE_DIR OR
   RUND_EXPECTED_PACKAGE_DIR STREQUAL "")
  message(FATAL_ERROR
    "package consumer requires the freshly installed package directory")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR
    "positive installed package consumer must use the producer Release build type")
endif()
find_package(runD 1.0.1 EXACT CONFIG REQUIRED)

file(REAL_PATH "${RUND_EXPECTED_PACKAGE_DIR}" expected_runD_dir)
file(REAL_PATH "${runD_DIR}" resolved_runD_dir)
if(NOT resolved_runD_dir STREQUAL expected_runD_dir)
  message(FATAL_ERROR
    "consumer resolved runD outside the fresh install prefix\n"
    "expected: ${expected_runD_dir}\n"
    "actual:   ${resolved_runD_dir}")
endif()

get_filename_component(runD_install_prefix "${resolved_runD_dir}/../../.."
                       ABSOLUTE)

if(NOT DEFINED runD_VERSION OR NOT runD_VERSION STREQUAL "1.0.1")
  message(FATAL_ERROR
    "installed package resolved a non-current SDK version: ${runD_VERSION}")
endif()

if(NOT TARGET runD::sdk)
  message(FATAL_ERROR "installed package is missing runD::sdk")
endif()
get_target_property(runD_sdk_imported runD::sdk IMPORTED)
if(NOT runD_sdk_imported)
  message(FATAL_ERROR "installed runD::sdk is not an imported package target")
endif()
get_target_property(runD_sdk_compile_features
                    runD::sdk INTERFACE_COMPILE_FEATURES)
if(NOT runD_sdk_compile_features STREQUAL "cxx_std_20")
  message(FATAL_ERROR
    "installed runD::sdk compile feature disagrees: ${runD_sdk_compile_features}")
endif()
get_target_property(runD_sdk_compile_options
                    runD::sdk INTERFACE_COMPILE_OPTIONS)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(runD_expected_compile_options -fno-fast-math -ffp-contract=off)
elseif(MSVC)
  set(runD_expected_compile_options /fp:strict)
else()
  message(FATAL_ERROR
    "installed runD::sdk has no numeric policy for ${CMAKE_CXX_COMPILER_ID}")
endif()
if(NOT "${runD_sdk_compile_options}" STREQUAL
   "${runD_expected_compile_options}")
  message(FATAL_ERROR
    "installed runD::sdk compile options disagree: ${runD_sdk_compile_options}")
endif()
get_target_property(runD_sdk_prefix runD::sdk RUND_PACKAGE_PREFIX)
if(NOT runD_sdk_prefix STREQUAL runD_install_prefix)
  message(FATAL_ERROR
    "installed runD::sdk prefix identity disagrees: ${runD_sdk_prefix}")
endif()
get_target_property(runD_sdk_includes runD::sdk
                    INTERFACE_INCLUDE_DIRECTORIES)
if(NOT runD_sdk_includes STREQUAL "${runD_install_prefix}/include")
  message(FATAL_ERROR
    "installed runD::sdk include boundary disagrees: ${runD_sdk_includes}")
endif()
get_target_property(runD_sdk_links runD::sdk INTERFACE_LINK_LIBRARIES)
get_target_property(runD_expected_links runD::sdk RUND_PACKAGE_LINK_CLOSURE)
if(runD_expected_links STREQUAL "runD_expected_links-NOTFOUND" OR
   runD_expected_links STREQUAL "")
  message(FATAL_ERROR "installed runD::sdk has no canonical link closure")
endif()
set(runD_required_archives
  "${runD_install_prefix}/lib/libcluster.a"
  "${runD_install_prefix}/lib/libnode.a"
  "${runD_install_prefix}/lib/libkernel.a")
foreach(runD_archive IN LISTS runD_required_archives)
  if(NOT EXISTS "${runD_archive}")
    message(FATAL_ERROR "installed runD archive is missing: ${runD_archive}")
  endif()
endforeach()
list(SUBLIST runD_sdk_links 0 3 runD_archive_prefix)
if(NOT "${runD_archive_prefix}" STREQUAL "${runD_required_archives}")
  message(FATAL_ERROR
    "installed runD archive order disagrees: ${runD_archive_prefix}")
endif()
foreach(runD_external_dependency IN ITEMS
    runD_METAL_FRAMEWORK
    runD_FOUNDATION_FRAMEWORK)
  if(DEFINED ${runD_external_dependency})
    if(NOT EXISTS "${${runD_external_dependency}}")
      message(FATAL_ERROR
        "installed package resolved a missing dependency: ${runD_external_dependency}")
    endif()
    list(FIND runD_sdk_links "${${runD_external_dependency}}"
         runD_dependency_index)
    if(runD_dependency_index EQUAL -1)
      message(FATAL_ERROR
        "installed runD::sdk omits dependency: ${runD_external_dependency}")
    endif()
  endif()
endforeach()
if(TARGET Vulkan::Vulkan)
  list(FIND runD_sdk_links Vulkan::Vulkan runD_vulkan_index)
  if(runD_vulkan_index EQUAL -1)
    message(FATAL_ERROR "installed runD::sdk omits Vulkan")
  endif()
endif()
if(NOT CMAKE_DL_LIBS STREQUAL "")
  list(FIND runD_sdk_links "${CMAKE_DL_LIBS}" runD_dl_index)
  if(runD_dl_index EQUAL -1)
    message(FATAL_ERROR "installed runD::sdk omits dynamic-loader support")
  endif()
endif()
set(runD_unique_links ${runD_sdk_links})
list(REMOVE_DUPLICATES runD_unique_links)
list(LENGTH runD_sdk_links runD_link_count)
list(LENGTH runD_unique_links runD_unique_link_count)
if(NOT runD_link_count EQUAL runD_unique_link_count)
  message(FATAL_ERROR
    "installed runD::sdk repeats a link dependency: ${runD_sdk_links}")
endif()
if(NOT "${runD_sdk_links}" STREQUAL "${runD_expected_links}")
  message(FATAL_ERROR
    "installed runD::sdk link closure disagrees\n"
    "expected: ${runD_expected_links}\n"
    "actual:   ${runD_sdk_links}")
endif()
get_property(runD_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
list(FILTER runD_imported_targets INCLUDE REGEX "^runD::")
if(NOT "${runD_imported_targets}" STREQUAL "runD::sdk")
  message(FATAL_ERROR
    "installed package target surface disagrees: ${runD_imported_targets}")
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES runD::sdk)
foreach(runD_probe IN ITEMS
    RUND_PACKAGE_HAS_PRODUCT_SURFACE
    RUND_PACKAGE_HAS_SESSION_BOUNDARY
    RUND_PACKAGE_ACCEPTS_MUTABLE_ELEMENT_FUNCTION
    RUND_PACKAGE_ACCEPTS_POINTER_ELEMENT_CAPTURE
    RUND_PACKAGE_ACCEPTS_REFERENCE_ELEMENT_CAPTURE
    RUND_PACKAGE_ACCEPTS_MUTABLE_GLOBAL_ELEMENT_READ
    RUND_PACKAGE_ACCEPTS_BRANCHED_MUTABLE_GLOBAL_READ)
  unset(${runD_probe} CACHE)
endforeach()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/contract/surface.cpp"
     runD_surface_contract)
check_cxx_source_compiles(
  "${runD_surface_contract}"
  RUND_PACKAGE_HAS_PRODUCT_SURFACE)
if(NOT RUND_PACKAGE_HAS_PRODUCT_SURFACE)
  message(FATAL_ERROR
    "installed package violates the current product surface contract")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/contract/session.cpp"
     runD_session_contract)
check_cxx_source_compiles(
  "${runD_session_contract}"
  RUND_PACKAGE_HAS_SESSION_BOUNDARY)
if(NOT RUND_PACKAGE_HAS_SESSION_BOUNDARY)
  message(FATAL_ERROR
    "installed Session entry imports the Compute execution surface")
endif()
check_cxx_source_compiles(
  "#include <rund/compute.hpp>\n#include <cstdint>\n#include <utility>\nint main() { auto flow = rund::compute::on(rund::compute::Target::cpu(1u)).map<std::int32_t>(\"mutable\", 4u, [state = 0](auto value) mutable { ++state; return value + state; }); auto program = std::move(flow).compile(); (void)program; }"
  RUND_PACKAGE_ACCEPTS_MUTABLE_ELEMENT_FUNCTION)
if(RUND_PACKAGE_ACCEPTS_MUTABLE_ELEMENT_FUNCTION)
  message(FATAL_ERROR
    "installed package accepts a mutable Compute element function")
endif()
check_cxx_source_compiles(
  "#include <rund/compute.hpp>\n#include <cstdint>\n#include <utility>\nint main() { std::int32_t state = 1; auto flow = rund::compute::on(rund::compute::Target::cpu(1u)).map<std::int32_t>(\"pointer\", 4u, [pointer = &state](auto value) { return value + *pointer; }); auto program = std::move(flow).compile(); (void)program; }"
  RUND_PACKAGE_ACCEPTS_POINTER_ELEMENT_CAPTURE)
if(RUND_PACKAGE_ACCEPTS_POINTER_ELEMENT_CAPTURE)
  message(FATAL_ERROR
    "installed package accepts a pointer element capture")
endif()
check_cxx_source_compiles(
  "#include <rund/compute.hpp>\n#include <cstdint>\n#include <utility>\nint main() { std::int32_t state = 1; auto flow = rund::compute::on(rund::compute::Target::cpu(1u)).map<std::int32_t>(\"reference\", 4u, [&state](auto value) { return value + state; }); auto program = std::move(flow).compile(); (void)program; }"
  RUND_PACKAGE_ACCEPTS_REFERENCE_ELEMENT_CAPTURE)
if(RUND_PACKAGE_ACCEPTS_REFERENCE_ELEMENT_CAPTURE)
  message(FATAL_ERROR
    "installed package accepts a reference element capture")
endif()
check_cxx_source_compiles(
  "#include <rund/compute.hpp>\n#include <cstdint>\n#include <utility>\nstd::int32_t mutable_host = 1; int main() { auto flow = rund::compute::on(rund::compute::Target::cpu(1u)).map<std::int32_t>(\"global\", 4u, [](auto value) { return value + mutable_host; }); auto program = std::move(flow).compile(); (void)program; }"
  RUND_PACKAGE_ACCEPTS_MUTABLE_GLOBAL_ELEMENT_READ)
if(RUND_PACKAGE_ACCEPTS_MUTABLE_GLOBAL_ELEMENT_READ)
  message(FATAL_ERROR
    "installed package accepts a mutable global element read")
endif()
check_cxx_source_compiles(
  "#include <rund/compute.hpp>\n#include <cstdint>\n#include <utility>\nstd::int32_t mutable_host = 1; int main() { auto fn = rund::compute::capture([](auto value, auto selector) { return rund::compute::select(selector != 0, value + mutable_host, value); }, std::int32_t{1}); auto flow = rund::compute::on(rund::compute::Target::cpu(1u)).map<std::int32_t>(\"branched-global\", 4u, fn); auto program = std::move(flow).compile(); (void)program; }"
  RUND_PACKAGE_ACCEPTS_BRANCHED_MUTABLE_GLOBAL_READ)
if(RUND_PACKAGE_ACCEPTS_BRANCHED_MUTABLE_GLOBAL_READ)
  message(FATAL_ERROR
    "installed package accepts a capture-controlled mutable global read")
endif()
unset(CMAKE_REQUIRED_LIBRARIES)

include(cmake/targets.cmake)
include(cmake/private.cmake)
include(cmake/public.cmake)

include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/sdk/surface.cmake")
rund_read_sdk_surface(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/surface/headers.tsv"
  rund_direct_headers rund_private_files rund_private_trees)
set(rund_header_dir "${CMAKE_CURRENT_BINARY_DIR}/header")
file(MAKE_DIRECTORY "${rund_header_dir}")
set(rund_header_sources)
set(rund_header_index 0)
foreach(rund_header IN LISTS rund_direct_headers)
  math(EXPR rund_header_index "${rund_header_index} + 1")
  set(rund_header_source
      "${rund_header_dir}/${rund_header_index}.cpp")
  file(CONFIGURE
       OUTPUT "${rund_header_source}"
       CONTENT "#include <${rund_header}>\n"
       @ONLY
       NEWLINE_STYLE UNIX)
  list(APPEND rund_header_sources "${rund_header_source}")
endforeach()
add_library(rund_package_header_consumer OBJECT
            ${rund_header_sources}
            umbrella.cpp)
target_link_libraries(rund_package_header_consumer PRIVATE runD::sdk)

function(rund_add_doc_snippet target source mode)
  if(NOT IS_ABSOLUTE "${source}" OR NOT EXISTS "${source}" OR
     IS_DIRECTORY "${source}")
    message(FATAL_ERROR
      "documentation snippet source is unavailable: ${source}")
  endif()
  if(mode STREQUAL "run")
    add_executable("${target}" "${source}")
    rund_register_consumer("${target}" 0 general)
  elseif(mode STREQUAL "compile")
    add_library("${target}" OBJECT "${source}")
  else()
    message(FATAL_ERROR "documentation snippet mode is invalid: ${mode}")
  endif()
  target_link_libraries("${target}" PRIVATE runD::sdk)
  set_property(GLOBAL APPEND PROPERTY RUND_DOC_SNIPPET_TARGETS "${target}")
endfunction()

if(NOT DEFINED RUND_DOC_SNIPPET_TARGETS OR
   RUND_DOC_SNIPPET_TARGETS STREQUAL "" OR
   NOT EXISTS "${RUND_DOC_SNIPPET_TARGETS}")
  message(FATAL_ERROR
    "installed package consumer requires classified documentation snippets")
endif()
include("${RUND_DOC_SNIPPET_TARGETS}")
get_property(rund_doc_snippet_targets GLOBAL PROPERTY RUND_DOC_SNIPPET_TARGETS)
get_property(rund_package_targets GLOBAL PROPERTY RUND_PACKAGE_TARGETS)
get_property(rund_package_runs GLOBAL PROPERTY RUND_PACKAGE_RUNS)
if("${rund_package_targets}" STREQUAL "" OR
   "${rund_package_runs}" STREQUAL "")
  message(FATAL_ERROR "installed package consumer registry is empty")
endif()
list(REMOVE_DUPLICATES rund_package_targets)
set(rund_package_run_manifest "${CMAKE_BINARY_DIR}/runs.tsv")
file(WRITE "${rund_package_run_manifest}" "")
foreach(rund_package_run IN LISTS rund_package_runs)
  string(REPLACE "|" "\t" rund_package_row "${rund_package_run}")
  file(APPEND "${rund_package_run_manifest}" "${rund_package_row}\n")
endforeach()

add_custom_target(rund_package_consumers
  DEPENDS
    rund_package_header_consumer
    ${rund_package_targets}
    ${rund_doc_snippet_targets})
