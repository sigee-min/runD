set(_rund_compiler_launcher "")
if(DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
  set(_rund_compiler_launcher "${CMAKE_CXX_COMPILER_LAUNCHER}")
elseif(DEFINED CMAKE_OBJCXX_COMPILER_LAUNCHER)
  set(_rund_compiler_launcher "${CMAKE_OBJCXX_COMPILER_LAUNCHER}")
else()
  find_program(RUND_CCACHE_EXECUTABLE NAMES ccache)
  mark_as_advanced(RUND_CCACHE_EXECUTABLE)
  if(RUND_CCACHE_EXECUTABLE)
    set(_rund_compiler_launcher
      "${CMAKE_COMMAND};-E;env;CCACHE_DIR=${CMAKE_SOURCE_DIR}/.cache/ccache;CCACHE_BASEDIR=${CMAKE_SOURCE_DIR};CCACHE_MAXSIZE=10G;${RUND_CCACHE_EXECUTABLE}"
    )
  endif()
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER AND _rund_compiler_launcher)
  set(CMAKE_CXX_COMPILER_LAUNCHER
    "${_rund_compiler_launcher}"
    CACHE STRING "C++ compiler launcher")
endif()
if(NOT DEFINED CMAKE_OBJCXX_COMPILER_LAUNCHER AND _rund_compiler_launcher)
  set(CMAKE_OBJCXX_COMPILER_LAUNCHER
    "${_rund_compiler_launcher}"
    CACHE STRING "Objective-C++ compiler launcher")
endif()
unset(_rund_compiler_launcher)

option(
  RUND_STRICT_WARNINGS
  "Compile repository-owned production and contract sources with strict warnings"
  ON)

set(
  RUND_INTERNAL_SANITIZER_PROFILE
  OFF
  CACHE BOOL
  "Use the repository sanitizer compile profile")
mark_as_advanced(RUND_INTERNAL_SANITIZER_PROFILE)

if(RUND_INTERNAL_SANITIZER_PROFILE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
    message(FATAL_ERROR
      "RUND_INTERNAL_SANITIZER_PROFILE has no policy for C++ compiler "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-O1>")
  if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-gline-tables-only>"
      "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-gno-inline-line-tables>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-g1>")
  endif()
endif()

if(RUND_STRICT_WARNINGS)
  if(MSVC)
    set(_rund_strict_warning_options
      /W4
      /WX
      /permissive-)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
    set(_rund_strict_warning_options
      -Wall
      -Wextra
      -Wpedantic
      -Werror)
  else()
    message(FATAL_ERROR
      "RUND_STRICT_WARNINGS has no policy for C++ compiler "
      "${CMAKE_CXX_COMPILER_ID}; add one before admitting this toolchain")
  endif()

  foreach(_rund_strict_warning_option IN LISTS _rund_strict_warning_options)
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:${_rund_strict_warning_option}>")
  endforeach()
  unset(_rund_strict_warning_option)
  unset(_rund_strict_warning_options)
endif()
