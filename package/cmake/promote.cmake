cmake_minimum_required(VERSION 3.20)

foreach(required_var IN ITEMS CANDIDATE PREFIX RANLIB)
  if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "package promotion requires ${required_var}")
  endif()
  get_filename_component(${required_var} "${${required_var}}" ABSOLUTE)
endforeach()

foreach(archive IN ITEMS libcluster.a libnode.a libkernel.a)
  execute_process(
    COMMAND "${RANLIB}" -D "${CANDIDATE}/lib/${archive}"
    RESULT_VARIABLE normalize_result
    OUTPUT_VARIABLE normalize_stdout
    ERROR_VARIABLE normalize_stderr)
  if(NOT normalize_result STREQUAL "0")
    message(FATAL_ERROR
      "failed to normalize package archive ${archive}\n"
      "${normalize_stdout}${normalize_stderr}")
  endif()
endforeach()

if(CANDIDATE STREQUAL PREFIX)
  message(FATAL_ERROR "package candidate and staged prefix must differ")
endif()
set(candidate_boundary "${CANDIDATE}/")
set(prefix_boundary "${PREFIX}/")
string(FIND "${candidate_boundary}" "${prefix_boundary}" prefix_contains_candidate)
string(FIND "${prefix_boundary}" "${candidate_boundary}" candidate_contains_prefix)
if(prefix_contains_candidate EQUAL 0 OR candidate_contains_prefix EQUAL 0)
  message(FATAL_ERROR "package candidate and staged prefix must not overlap")
endif()

foreach(required_path IN ITEMS
    include
    lib/libcluster.a
    lib/libnode.a
    lib/libkernel.a
    lib/cmake/runD/runDConfig.cmake
    lib/cmake/runD/runDConfigVersion.cmake
    lib/cmake/runD/runDTargets.cmake
    share/runD/LICENSE
    share/runD/licenses/xxhash/LICENSE
    share/runD/sdk-headers.tsv)
  if(NOT EXISTS "${CANDIDATE}/${required_path}")
    message(FATAL_ERROR
      "package candidate is missing required path: ${required_path}")
  endif()
endforeach()

function(rund_package_inventory root output)
  if(NOT IS_DIRECTORY "${root}")
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  if(IS_SYMLINK "${root}")
    message(FATAL_ERROR "package prefix root is a symlink: ${root}")
  endif()
  file(GLOB_RECURSE paths
    LIST_DIRECTORIES false
    RELATIVE "${root}"
    "${root}/*")
  list(SORT paths)
  set(rows)
  foreach(path IN LISTS paths)
    set(file "${root}/${path}")
    if(IS_SYMLINK "${file}")
      message(FATAL_ERROR "package file is a symlink: ${file}")
    endif()
    file(SIZE "${file}" size)
    file(SHA256 "${file}" sha256)
    list(APPEND rows "${path}\t${size}\t${sha256}")
  endforeach()
  set(${output} "${rows}" PARENT_SCOPE)
endfunction()

rund_package_inventory("${CANDIDATE}" candidate_inventory)
rund_package_inventory("${PREFIX}" prefix_inventory)
if("${candidate_inventory}" STREQUAL "${prefix_inventory}")
  file(REMOVE_RECURSE "${CANDIDATE}")
  message(STATUS "Retained unchanged runD package staging prefix")
  return()
endif()

file(REMOVE_RECURSE "${PREFIX}")
file(RENAME "${CANDIDATE}" "${PREFIX}")
message(STATUS "Promoted changed runD package staging prefix")
