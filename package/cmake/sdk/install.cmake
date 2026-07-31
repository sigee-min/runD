cmake_minimum_required(VERSION 3.20)

set(source_root "@CMAKE_CURRENT_SOURCE_DIR@")
set(work_root "@CMAKE_CURRENT_BINARY_DIR@/package/sdk-headers")
set(compiler "@CMAKE_CXX_COMPILER@")
set(compiler_id "@CMAKE_CXX_COMPILER_ID@")
if(NOT compiler_id MATCHES "Clang|GNU")
  message(FATAL_ERROR
    "exact SDK header installation requires a Clang or GNU compiler")
endif()

include("${source_root}/package/cmake/sdk/surface.cmake")
rund_read_sdk_surface(
  "${source_root}/package/docs/surface/headers.tsv"
  direct_headers private_files private_trees)

set(include_roots
  "${source_root}/cluster/include"
  "${source_root}/math32/include"
  "${source_root}/math64/include"
  "${source_root}/node/include"
  "${source_root}/accel/include"
  "${source_root}/kernel/include")

function(resolve_surface_entry path expected_type output_var)
  set(matches)
  foreach(include_root IN LISTS include_roots)
    set(candidate "${include_root}/${path}")
    if(EXISTS "${candidate}")
      if(IS_SYMLINK "${candidate}")
        message(FATAL_ERROR "SDK surface entry is a symlink: ${path}")
      endif()
      if(expected_type STREQUAL "file" AND IS_DIRECTORY "${candidate}")
        message(FATAL_ERROR "SDK surface file is a directory: ${path}")
      elseif(expected_type STREQUAL "tree" AND NOT IS_DIRECTORY "${candidate}")
        message(FATAL_ERROR "SDK surface tree is not a directory: ${path}")
      endif()
      list(APPEND matches "${candidate}")
    endif()
  endforeach()
  list(LENGTH matches match_count)
  if(NOT match_count EQUAL 1)
    message(FATAL_ERROR
      "SDK surface entry must have exactly one source owner: "
      "${path} (${match_count})")
  endif()
  list(GET matches 0 match)
  set(${output_var} "${match}" PARENT_SCOPE)
endfunction()

foreach(path IN LISTS direct_headers private_files)
  resolve_surface_entry("${path}" file resolved_entry)
endforeach()
foreach(path IN LISTS private_trees)
  resolve_surface_entry("${path}" tree resolved_entry)
endforeach()

file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}/probe")
set(include_flags)
foreach(include_root IN LISTS include_roots)
  list(APPEND include_flags "-I${include_root}")
endforeach()
set(toolchain_flags)
if(NOT "@CMAKE_CXX_COMPILER_TARGET@" STREQUAL "")
  list(APPEND toolchain_flags "--target=@CMAKE_CXX_COMPILER_TARGET@")
endif()
if(NOT "@CMAKE_SYSROOT@" STREQUAL "")
  list(APPEND toolchain_flags "--sysroot=@CMAKE_SYSROOT@")
elseif(NOT "@CMAKE_OSX_SYSROOT@" STREQUAL "")
  list(APPEND toolchain_flags "-isysroot" "@CMAKE_OSX_SYSROOT@")
endif()

set(closure_pairs)
set(probe_index 0)
foreach(header IN LISTS direct_headers)
  math(EXPR probe_index "${probe_index} + 1")
  set(probe "${work_root}/probe/header-${probe_index}.cpp")
  file(WRITE "${probe}" "#include <${header}>\n")
  file(REAL_PATH "${probe}" probe_path)
  execute_process(
    COMMAND "${compiler}" ${toolchain_flags} -std=c++20
            -fno-fast-math -ffp-contract=off
            -MM -MT rund-sdk-header-probe "${probe}" ${include_flags}
    WORKING_DIRECTORY "${source_root}"
    RESULT_VARIABLE dependency_result
    OUTPUT_VARIABLE dependency_output
    ERROR_VARIABLE dependency_error)
  if(NOT dependency_result STREQUAL "0")
    message(FATAL_ERROR
      "SDK direct header dependency probe failed: ${header}\n"
      "${dependency_error}")
  endif()

  string(REPLACE "\\\n" " " dependency_text "${dependency_output}")
  string(REGEX REPLACE "^[^:]*:[ \t]*" "" dependency_text
    "${dependency_text}")
  separate_arguments(dependencies UNIX_COMMAND "${dependency_text}")
  foreach(dependency IN LISTS dependencies)
    file(REAL_PATH "${dependency}" dependency_path
         BASE_DIRECTORY "${source_root}")
    if(dependency_path STREQUAL probe_path)
      continue()
    endif()

    set(relative "")
    set(owner_count 0)
    foreach(include_root IN LISTS include_roots)
      file(RELATIVE_PATH candidate_relative "${include_root}"
           "${dependency_path}")
      if(NOT candidate_relative MATCHES "^\\.\\.(/|$)" AND
         NOT IS_ABSOLUTE "${candidate_relative}")
        math(EXPR owner_count "${owner_count} + 1")
        set(relative "${candidate_relative}")
      endif()
    endforeach()
    if(NOT owner_count EQUAL 1)
      message(FATAL_ERROR
        "SDK dependency must have exactly one include owner: "
        "${dependency_path} (${owner_count})")
    endif()
    if(IS_SYMLINK "${dependency_path}")
      message(FATAL_ERROR "SDK dependency is a symlink: ${relative}")
    endif()
    list(APPEND closure_pairs "${relative}\t${dependency_path}")
  endforeach()
endforeach()

list(REMOVE_DUPLICATES closure_pairs)
list(SORT closure_pairs)
set(inventory)
set(previous_relative "")
set(install_include "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/include")
foreach(pair IN LISTS closure_pairs)
  string(REPLACE "\t" ";" fields "${pair}")
  list(GET fields 0 relative)
  list(GET fields 1 source)
  if(relative STREQUAL previous_relative)
    message(FATAL_ERROR "SDK install-relative header collision: ${relative}")
  endif()
  set(previous_relative "${relative}")
  list(APPEND inventory "${relative}")
  get_filename_component(destination "${relative}" DIRECTORY)
  file(INSTALL "${source}" DESTINATION "${install_include}/${destination}"
       MESSAGE_NEVER)
endforeach()

rund_check_sdk_closure(
  inventory direct_headers private_files private_trees)
set(manifest "${work_root}/sdk-headers.tsv")
file(WRITE "${manifest}" "path\n")
foreach(path IN LISTS inventory)
  file(APPEND "${manifest}" "${path}\n")
endforeach()
rund_read_sdk_inventory("${manifest}" written_inventory)
if(NOT "${written_inventory}" STREQUAL "${inventory}")
  message(FATAL_ERROR "written SDK header inventory differs from closure")
endif()
file(INSTALL "${manifest}"
     DESTINATION "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/runD"
     MESSAGE_NEVER)
list(LENGTH inventory inventory_count)
message(STATUS "Installed exact runD SDK header closure: ${inventory_count} files")
