# Use the real root policy and Ninja generator; an argv-only fixture cannot
# prove that CMake attached the intended pool to executable link rules.
set(link_source "${fixture}/link-source")
file(MAKE_DIRECTORY "${link_source}")
file(WRITE "${link_source}/main.cpp" "int main() { return 0; }\n")
file(WRITE "${link_source}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.20)\n"
  "project(LinkPool LANGUAGES CXX)\n"
  "set(RUND_STRICT_WARNINGS OFF CACHE BOOL \"\" FORCE)\n"
  "set(CMAKE_CXX_COMPILER_LAUNCHER \"\")\n"
  "set(CMAKE_OBJCXX_COMPILER_LAUNCHER \"\")\n"
  "include(\"${ROOT}/cmake/root/compile.cmake\")\n"
  "add_executable(alpha main.cpp)\n"
  "add_executable(beta main.cpp)\n")

foreach(mode IN ITEMS default caller)
  set(link_build "${fixture}/link-${mode}")
  if(mode STREQUAL default)
    set(link_options
      "-DCMAKE_JOB_POOLS=caller_compile=3"
      "-DCMAKE_JOB_POOL_COMPILE=caller_compile")
    set(link_pool rund_link)
    set(link_depth 1)
  else()
    set(link_options
      "-DCMAKE_JOB_POOLS=caller_link=2"
      "-DCMAKE_JOB_POOL_LINK=caller_link")
    set(link_pool caller_link)
    set(link_depth 2)
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${link_source}" -B "${link_build}"
      -G Ninja ${link_options}
    RESULT_VARIABLE link_configured
    OUTPUT_VARIABLE link_output ERROR_VARIABLE link_error)
  if(NOT link_configured EQUAL 0)
    message(FATAL_ERROR "Link-pool configure failed: ${link_output}\n${link_error}")
  endif()
  file(READ "${link_build}/CMakeFiles/rules.ninja" link_rules)
  file(READ "${link_build}/build.ninja" link_graph)
  if(NOT link_rules MATCHES "pool ${link_pool}\n +depth = ${link_depth}\n")
    message(FATAL_ERROR "Ninja link-pool depth missing for ${mode}")
  endif()
  string(REGEX MATCHALL "pool = ${link_pool}\n" link_assignments "${link_graph}")
  list(LENGTH link_assignments link_count)
  if(NOT link_count EQUAL 2)
    message(FATAL_ERROR "Expected two bounded executable links, got ${link_count}")
  endif()
  if(mode STREQUAL default AND
     NOT link_rules MATCHES "pool caller_compile\n +depth = 3\n")
    message(FATAL_ERROR "Repository link pool discarded the caller's compiler pool")
  endif()
endforeach()
