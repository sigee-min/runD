@PACKAGE_INIT@

set(runD_VERSION "@RUND_PACKAGE_VERSION@")
set(runD_FOUND TRUE)

# runD exports one SDK target and no CMake components.  Resolve a required
# component request before dependency discovery or target import so a rejected
# lookup cannot leave runD::sdk behind in the caller's directory.
foreach(_runD_component IN LISTS runD_FIND_COMPONENTS)
  set("runD_${_runD_component}_FOUND" FALSE)
endforeach()
check_required_components(runD)
if(DEFINED runD_FOUND AND NOT runD_FOUND)
  set(runD_NOT_FOUND_MESSAGE
    "runD does not provide CMake components; requested: ${runD_FIND_COMPONENTS}")
  unset(_runD_component)
  return()
endif()
unset(_runD_component)

@RUND_PACKAGE_CONFIG_FIND_DEPENDENCIES@

include("${CMAKE_CURRENT_LIST_DIR}/runDTargets.cmake")
