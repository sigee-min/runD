if(TARGET runD::sdk)
  get_target_property(_runD_existing_prefix
    runD::sdk RUND_PACKAGE_PREFIX)
  if(_runD_existing_prefix STREQUAL "${PACKAGE_PREFIX_DIR}")
    unset(_runD_existing_prefix)
    return()
  endif()
  message(FATAL_ERROR
    "runD::sdk already exists and is not owned by ${PACKAGE_PREFIX_DIR}")
endif()

set(_runD_links
  "${PACKAGE_PREFIX_DIR}/lib/libcluster.a"
  "${PACKAGE_PREFIX_DIR}/lib/libnode.a"
  "${PACKAGE_PREFIX_DIR}/lib/libkernel.a"
)

if(DEFINED runD_METAL_FRAMEWORK)
  list(APPEND _runD_links "${runD_METAL_FRAMEWORK}")
endif()
if(DEFINED runD_FOUNDATION_FRAMEWORK)
  list(APPEND _runD_links "${runD_FOUNDATION_FRAMEWORK}")
endif()
if(TARGET Vulkan::Vulkan)
  list(APPEND _runD_links Vulkan::Vulkan)
endif()
if(NOT "@CMAKE_DL_LIBS@" STREQUAL "")
  list(APPEND _runD_links "@CMAKE_DL_LIBS@")
endif()

add_library(runD::sdk INTERFACE IMPORTED)
set_target_properties(runD::sdk PROPERTIES
  RUND_PACKAGE_PREFIX "${PACKAGE_PREFIX_DIR}"
  RUND_PACKAGE_LINK_CLOSURE "${_runD_links}"
  INTERFACE_COMPILE_FEATURES cxx_std_20
  INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include"
  INTERFACE_LINK_LIBRARIES "${_runD_links}"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_property(TARGET runD::sdk PROPERTY INTERFACE_COMPILE_OPTIONS
    "-fno-fast-math;-ffp-contract=off")
elseif(MSVC)
  set_property(TARGET runD::sdk PROPERTY INTERFACE_COMPILE_OPTIONS "/fp:strict")
endif()

unset(_runD_links)
