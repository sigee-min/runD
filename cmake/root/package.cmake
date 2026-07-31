# Root-owned package install and export targets.
set(RUND_PACKAGE_VERSION "1.0.1")

set(RUND_SDK_INSTALL_SCRIPT
    "${CMAKE_CURRENT_BINARY_DIR}/package/sdk/install.cmake")
get_filename_component(RUND_SDK_INSTALL_SCRIPT_DIR
                       "${RUND_SDK_INSTALL_SCRIPT}" DIRECTORY)
file(MAKE_DIRECTORY "${RUND_SDK_INSTALL_SCRIPT_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/package/cmake/sdk/install.cmake"
  "${RUND_SDK_INSTALL_SCRIPT}"
  @ONLY)
install(SCRIPT "${RUND_SDK_INSTALL_SCRIPT}")

install(
  TARGETS kernel node cluster
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/runDConfigVersion.cmake"
  VERSION "${RUND_PACKAGE_VERSION}"
  COMPATIBILITY ExactVersion
)

set(RUND_PACKAGE_CONFIG_FIND_DEPENDENCIES "")
if(RUND_NODE_PACKAGE_NEEDS_METAL_FRAMEWORKS)
  string(APPEND RUND_PACKAGE_CONFIG_FIND_DEPENDENCIES
    "find_library(runD_METAL_FRAMEWORK NAMES Metal REQUIRED)\n"
    "find_library(runD_FOUNDATION_FRAMEWORK NAMES Foundation REQUIRED)\n"
  )
endif()
if(RUND_NODE_PACKAGE_NEEDS_VULKAN_TARGET)
  string(APPEND RUND_PACKAGE_CONFIG_FIND_DEPENDENCIES
    "include(CMakeFindDependencyMacro)\n"
    "find_dependency(Vulkan)\n"
  )
endif()
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/runDConfig.cmake"
  INSTALL_DESTINATION lib/cmake/runD
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/targets.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/runDTargets.cmake"
  @ONLY
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/runDConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/runDConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/runDTargets.cmake"
  DESTINATION lib/cmake/runD
)

install(
  FILES LICENSE
  DESTINATION share/runD
)

install(
  FILES node/src/host/vendor/xxhash/LICENSE
  DESTINATION share/runD/licenses/xxhash
)

set(RUND_PACKAGE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/runD-install")
if(RUND_ENABLE_PACKAGE_RELEASE_TESTS)
  set(RUND_PACKAGE_INSTALL_CANDIDATE
      "${CMAKE_BINARY_DIR}/runD-install.next")
  add_custom_target(runD_package_install
    COMMAND ${CMAKE_COMMAND} -E remove_directory
            ${RUND_PACKAGE_INSTALL_CANDIDATE}
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}
            --prefix ${RUND_PACKAGE_INSTALL_CANDIDATE}
    COMMAND ${CMAKE_COMMAND}
            -D "CANDIDATE=${RUND_PACKAGE_INSTALL_CANDIDATE}"
            -D "PREFIX=${RUND_PACKAGE_INSTALL_PREFIX}"
            -D "RANLIB=${CMAKE_RANLIB}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/package/cmake/promote.cmake"
    DEPENDS kernel node cluster
    COMMENT "Installing runD package staging prefix"
    VERBATIM)
endif()
