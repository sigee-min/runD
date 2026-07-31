set(RUND_NODE_METAL_OBJCXX_SOURCES)
foreach(source IN LISTS RUND_NODE_COMPONENT_ACCEL_METAL_SOURCES)
  if(source MATCHES "\\.mm$")
    list(APPEND RUND_NODE_METAL_OBJCXX_SOURCES "${source}")
  endif()
endforeach()

if(APPLE)
  list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew")
  find_library(RUND_NODE_METAL_FRAMEWORK Metal)
  find_library(RUND_NODE_FOUNDATION_FRAMEWORK Foundation)
  if(RUND_NODE_METAL_FRAMEWORK AND RUND_NODE_FOUNDATION_FRAMEWORK)
    set(RUND_NODE_HAVE_METAL_SDK TRUE)
    set(RUND_NODE_PACKAGE_NEEDS_METAL_FRAMEWORKS TRUE PARENT_SCOPE)
    enable_language(OBJCXX)
    set_source_files_properties(${RUND_NODE_METAL_OBJCXX_SOURCES} PROPERTIES
      LANGUAGE OBJCXX
      COMPILE_OPTIONS "-fobjc-arc"
    )
  else()
    set_source_files_properties(${RUND_NODE_METAL_OBJCXX_SOURCES} PROPERTIES
      LANGUAGE CXX)
  endif()
else()
  set_source_files_properties(${RUND_NODE_METAL_OBJCXX_SOURCES} PROPERTIES
    LANGUAGE CXX)
endif()

if(RUND_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
endif()
if(RUND_ENABLE_VULKAN AND TARGET Vulkan::Vulkan)
  set(RUND_NODE_HAVE_VULKAN_SDK TRUE)
  set(RUND_NODE_VULKAN_TARGET Vulkan::Vulkan)
  set(RUND_NODE_PACKAGE_NEEDS_VULKAN_TARGET TRUE PARENT_SCOPE)
endif()
if(RUND_ENABLE_VULKAN AND UNIX)
  find_program(RUND_NODE_GLSLANG_VALIDATOR
    NAMES glslangValidator
    HINTS /opt/homebrew/bin
  )
  find_program(RUND_NODE_SPIRV_VAL
    NAMES spirv-val
    HINTS /opt/homebrew/bin
  )
endif()

function(rund_node_object_context target component)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/accel/include"
    "${CMAKE_SOURCE_DIR}/kernel/include"
    "${CMAKE_SOURCE_DIR}/math32/include"
    "${CMAKE_SOURCE_DIR}/math64/include")
  target_compile_features(${target} PRIVATE cxx_std_20)
  set(object_links math32 math64)
  target_compile_definitions(${target} PRIVATE RUND_NODE_INTERNAL_BUILD=1)
  if(RUND_NODE_USE_UNAVAILABLE_PLATFORM)
    target_compile_definitions(${target} PRIVATE
      RUND_NODE_PLATFORM_UNAVAILABLE=1)
  endif()
  if(component STREQUAL "ACCEL_PICK")
    rund_node_project_native_components(
      catalog_native "${RUND_NODE_FOCUSED_BACKEND}")
    foreach(native IN LISTS catalog_native)
      if(native STREQUAL "ACCEL_METAL")
        target_compile_definitions(${target} PRIVATE
          RUND_NODE_CATALOG_METAL=1)
      elseif(native STREQUAL "ACCEL_VULKAN")
        target_compile_definitions(${target} PRIVATE
          RUND_NODE_CATALOG_VULKAN=1)
      else()
        message(FATAL_ERROR
          "Unknown focused catalog component: ${native}")
      endif()
    endforeach()
  endif()
  list(FIND RUND_NODE_NATIVE_COMPONENTS "${component}"
    native_component_index)
  if(native_component_index EQUAL -1)
    target_link_libraries(${target} PRIVATE ${object_links})
    set_target_properties(${target} PROPERTIES
      RUND_NODE_OBJECT_LINKS "${object_links}")
    return()
  endif()
  if(component STREQUAL "ACCEL_METAL")
    if(RUND_NODE_HAVE_METAL_SDK)
      target_compile_definitions(${target} PRIVATE RUND_NODE_HAVE_METAL_SDK=1)
      list(APPEND object_links
        "${RUND_NODE_METAL_FRAMEWORK}"
        "${RUND_NODE_FOUNDATION_FRAMEWORK}")
    endif()
  elseif(component STREQUAL "ACCEL_VULKAN")
    if(RUND_NODE_HAVE_VULKAN_SDK)
      target_compile_definitions(${target} PRIVATE RUND_NODE_HAVE_VULKAN_SDK=1)
      list(APPEND object_links ${RUND_NODE_VULKAN_TARGET})
    endif()
    if(RUND_ENABLE_VULKAN AND RUND_NODE_GLSLANG_VALIDATOR)
      target_compile_definitions(${target} PRIVATE
        RUND_NODE_HAVE_GLSLANG_VALIDATOR=1
        RUND_NODE_GLSLANG_VALIDATOR_PATH="${RUND_NODE_GLSLANG_VALIDATOR}")
    endif()
    if(RUND_ENABLE_VULKAN AND RUND_NODE_SPIRV_VAL)
      target_compile_definitions(${target} PRIVATE
        RUND_NODE_HAVE_SPIRV_VAL=1
        RUND_NODE_SPIRV_VAL_PATH="${RUND_NODE_SPIRV_VAL}")
    endif()
  else()
    message(FATAL_ERROR "Unknown Node native component context: ${component}")
  endif()
  target_link_libraries(${target} PRIVATE ${object_links})
  set_target_properties(${target} PROPERTIES
    RUND_NODE_OBJECT_LINKS "${object_links}")
endfunction()

function(rund_node_archive_base_context target)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()

function(rund_node_native_link_dependencies out)
  set(dependencies)
  foreach(component IN LISTS ARGN)
    if(component STREQUAL "ACCEL_METAL")
      if(RUND_NODE_HAVE_METAL_SDK)
        list(APPEND dependencies
          "${RUND_NODE_METAL_FRAMEWORK}"
          "${RUND_NODE_FOUNDATION_FRAMEWORK}")
      endif()
    elseif(component STREQUAL "ACCEL_VULKAN")
      if(RUND_NODE_HAVE_VULKAN_SDK)
        list(APPEND dependencies ${RUND_NODE_VULKAN_TARGET})
      endif()
    else()
      message(FATAL_ERROR "Unknown Node native link component: ${component}")
    endif()
  endforeach()
  set(${out} "${dependencies}" PARENT_SCOPE)
endfunction()

function(rund_node_accel_link_dependencies out)
  rund_node_native_link_dependencies(dependencies
    ACCEL_METAL
    ACCEL_VULKAN)
  set(${out} "${dependencies}" PARENT_SCOPE)
endfunction()

function(rund_node_installed_archive_context target)
  rund_node_archive_base_context(${target})
  target_link_libraries(${target} PUBLIC accel kernel ${CMAKE_DL_LIBS})
  target_link_libraries(${target} PRIVATE math32 math64)
  rund_node_accel_link_dependencies(accel_dependencies)
  if(accel_dependencies)
    target_link_libraries(${target} PRIVATE ${accel_dependencies})
  endif()
endfunction()

function(rund_node_internal_archive_context target root)
  rund_node_archive_base_context(${target})
  foreach(kernel_view IN ITEMS compute dispatch execution)
    if(NOT TARGET kernel-closure-${kernel_view})
      message(FATAL_ERROR
        "Node internal archives require Kernel ${kernel_view} view")
    endif()
  endforeach()
  set(public_dependencies)
  set(private_dependencies)
  if(root STREQUAL "CPU_SIMD")
    list(APPEND public_dependencies kernel-closure-compute)
  elseif(root STREQUAL "WORKER_BACKEND")
    list(APPEND public_dependencies kernel-closure-dispatch)
  elseif(root STREQUAL "CPU_COMPUTE")
    list(APPEND public_dependencies
      kernel-closure-compute
      kernel-closure-dispatch
      kernel-closure-execution)
  elseif(root STREQUAL "RUNTIME_BASE")
    list(APPEND public_dependencies kernel)
  elseif(root STREQUAL "CPU_ACCEL")
    # CPU_ACCEL reaches the exact Kernel compute view through CPU_SIMD.
    # It owns no public Accel or full Kernel transition.
  elseif(root STREQUAL "ACCEL_EXECUTION")
    # The internal execution closure owns implementation and native runtime
    # edges only. Public Accel and full Kernel remain installed-SDK concerns.
    if(CMAKE_DL_LIBS)
      list(APPEND public_dependencies ${CMAKE_DL_LIBS})
    endif()
    rund_node_scc_direct_components(accel_components ACCEL_EXECUTION)
    set(native_components ${accel_components})
    list(FILTER native_components INCLUDE REGEX "^ACCEL_(METAL|VULKAN)$")
    rund_node_native_link_dependencies(
      private_dependencies ${native_components})
  elseif(root STREQUAL "COMPUTE_EXECUTION")
    # CPU_COMPUTE owns the complete disjoint Kernel view union. This native
    # adapter SCC adds no second Kernel edge.
  elseif(root STREQUAL "CPU_RUNTIME_PRODUCT" OR
         root STREQUAL "RUNTIME_PRODUCT" OR root STREQUAL "NUMERIC" OR
         root STREQUAL "TELEMETRY")
    # Product joins have only SCC edges. Numeric carries no out-of-line
    # external dependency.
  elseif(NOT root STREQUAL "NUMERIC" AND NOT root STREQUAL "TELEMETRY")
    message(FATAL_ERROR "Node archive ${target} has unknown SCC root ${root}")
  endif()
  if(public_dependencies)
    target_link_libraries(${target} PUBLIC ${public_dependencies})
  endif()
  if(private_dependencies)
    target_link_libraries(${target} PRIVATE ${private_dependencies})
  endif()
  set(external_dependencies
    ${public_dependencies}
    ${private_dependencies})
  set_target_properties(${target} PROPERTIES
    RUND_NODE_EXTERNAL_LINKS "${external_dependencies}")
endfunction()

# Product translation units compile only here.  Archives below are link-shape
# owners composed from these objects and contain no independently compiled
# implementation source.
foreach(component IN LISTS RUND_NODE_COMPONENTS)
  set(name "${RUND_NODE_COMPONENT_${component}_NAME}")
  set(sources ${RUND_NODE_COMPONENT_${component}_SOURCES})
  if(NOT sources)
    message(FATAL_ERROR
      "Node component ${component} has no object sources")
  endif()
  set(target "node-object-${name}")
  add_library(${target} OBJECT ${sources})
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  rund_node_object_context(${target} "${component}")
  set("RUND_NODE_COMPONENT_${component}_TARGET" "${target}")
endforeach()

set(rund_node_catalog_target
  "${RUND_NODE_COMPONENT_ACCEL_PICK_TARGET}")
get_target_property(rund_node_catalog_definitions
  ${rund_node_catalog_target} COMPILE_DEFINITIONS)
if(rund_node_catalog_definitions MATCHES "-NOTFOUND$")
  set(rund_node_catalog_definitions)
endif()
list(FILTER rund_node_catalog_definitions INCLUDE REGEX
  "^RUND_NODE_CATALOG_(METAL|VULKAN)=1$")
rund_node_project_native_components(
  rund_node_catalog_components "${RUND_NODE_FOCUSED_BACKEND}")
set(rund_node_expected_catalog_definitions)
foreach(component IN LISTS rund_node_catalog_components)
  if(component STREQUAL "ACCEL_METAL")
    list(APPEND rund_node_expected_catalog_definitions
      RUND_NODE_CATALOG_METAL=1)
  elseif(component STREQUAL "ACCEL_VULKAN")
    list(APPEND rund_node_expected_catalog_definitions
      RUND_NODE_CATALOG_VULKAN=1)
  endif()
endforeach()
list(SORT rund_node_catalog_definitions)
list(SORT rund_node_expected_catalog_definitions)
if(NOT "${rund_node_catalog_definitions}" STREQUAL
   "${rund_node_expected_catalog_definitions}")
  message(FATAL_ERROR
    "Node picker catalog does not match the focused native component projection")
endif()

function(rund_node_archive_objects target root scope)
  if(scope STREQUAL "direct")
    rund_node_scc_direct_components(components "${root}")
  elseif(scope STREQUAL "closure")
    rund_node_scc_components(components "${root}")
  else()
    message(FATAL_ERROR
      "Node archive ${target} has unknown object scope ${scope}")
  endif()
  if(NOT components)
    message(FATAL_ERROR
      "Node archive ${target} has empty ${scope} SCC object set ${root}")
  endif()
  set(object_count 0)
  foreach(component IN LISTS components)
    set(sources ${RUND_NODE_COMPONENT_${component}_SOURCES})
    set(object_target "${RUND_NODE_COMPONENT_${component}_TARGET}")
    if(NOT object_target OR NOT TARGET ${object_target})
      message(FATAL_ERROR
        "Node SCC ${root} has missing object component ${component}")
    endif()
    target_sources(${target} PRIVATE $<TARGET_OBJECTS:${object_target}>)
    math(EXPR object_count "${object_count} + 1")
  endforeach()
  if(object_count EQUAL 0)
    message(FATAL_ERROR "Node archive ${target} has no object components")
  endif()
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE CXX
    RUND_NODE_SCC_ROOT "${root}"
    RUND_NODE_ARCHIVE_SCOPE "${scope}"
    RUND_NODE_ARCHIVE_COMPONENTS "${components}")
endfunction()

function(rund_node_internal_scc_target out root)
  list(FIND RUND_NODE_SCCS "${root}" root_index)
  if(root_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node internal archive SCC: ${root}")
  endif()
  string(TOLOWER "${root}" name)
  string(REPLACE "_" "-" name "${name}")
  set(${out} "node-closure-${name}" PARENT_SCOPE)
endfunction()

function(rund_node_scc_link_target out root)
  if(root STREQUAL "-")
    set(target "")
  else()
    rund_node_internal_scc_target(target "${root}")
  endif()
  if(target AND NOT TARGET ${target})
    message(FATAL_ERROR
      "Node SCC ${root} has missing canonical archive target ${target}")
  endif()
  set(${out} "${target}" PARENT_SCOPE)
endfunction()

function(rund_node_link_scc_dependencies target root)
  set(dependency_targets)
  foreach(dependency IN LISTS RUND_NODE_SCC_${root}_DEPENDS)
    rund_node_internal_scc_target(dependency_target "${dependency}")
    if(NOT TARGET ${dependency_target})
      message(FATAL_ERROR
        "Node SCC ${root} has missing dependency target ${dependency_target}")
    endif()
    list(APPEND dependency_targets "${dependency_target}")
    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
      target_link_libraries(${target} INTERFACE ${dependency_target})
    else()
      target_link_libraries(${target} PUBLIC ${dependency_target})
    endif()
  endforeach()
  set_target_properties(${target} PROPERTIES
    RUND_NODE_SCC_DEPENDENCY_TARGETS "${dependency_targets}")
endfunction()

function(rund_node_require_archive_graph product_root)
  set(materialized_components)
  foreach(root IN LISTS RUND_NODE_SCCS)
    rund_node_internal_scc_target(target "${root}")
    get_target_property(scope ${target} RUND_NODE_ARCHIVE_SCOPE)
    get_target_property(actual_components
      ${target} RUND_NODE_ARCHIVE_COMPONENTS)
    if(actual_components MATCHES "-NOTFOUND$")
      set(actual_components)
    endif()
    rund_node_scc_direct_components(expected_components "${root}")
    get_target_property(target_type ${target} TYPE)
    if(expected_components AND NOT target_type STREQUAL "STATIC_LIBRARY")
      message(FATAL_ERROR
        "Node object SCC ${root} is not a static archive")
    elseif(NOT expected_components AND
           NOT target_type STREQUAL "INTERFACE_LIBRARY")
      message(FATAL_ERROR
        "Node dependency-only SCC ${root} is not an interface join")
    endif()
    if(NOT scope STREQUAL "direct" OR
       NOT "${actual_components}" STREQUAL "${expected_components}")
      message(FATAL_ERROR
        "Node internal archive ${target} materializes transitive SCC objects")
    endif()

    set(expected_sources)
    foreach(component IN LISTS expected_components)
      list(APPEND materialized_components "${component}")
      set(object_target "${RUND_NODE_COMPONENT_${component}_TARGET}")
      list(APPEND expected_sources "$<TARGET_OBJECTS:${object_target}>")
      get_target_property(object_links ${object_target} LINK_LIBRARIES)
      get_target_property(expected_object_links
        ${object_target} RUND_NODE_OBJECT_LINKS)
      if(NOT "${object_links}" STREQUAL "${expected_object_links}")
        message(FATAL_ERROR
          "Node component ${component} has non-canonical object links")
      endif()
    endforeach()
    get_target_property(actual_sources ${target} SOURCES)
    if(actual_sources MATCHES "-NOTFOUND$")
      set(actual_sources)
    endif()
    if(NOT "${actual_sources}" STREQUAL "${expected_sources}")
      message(FATAL_ERROR
        "Node internal archive ${target} has non-canonical object inputs")
    endif()

    set(expected_dependency_targets)
    foreach(dependency IN LISTS RUND_NODE_SCC_${root}_DEPENDS)
      rund_node_internal_scc_target(dependency_target "${dependency}")
      list(APPEND expected_dependency_targets "${dependency_target}")
    endforeach()
    get_target_property(actual_dependency_targets
      ${target} RUND_NODE_SCC_DEPENDENCY_TARGETS)
    if(actual_dependency_targets MATCHES "-NOTFOUND$")
      set(actual_dependency_targets)
    endif()
    if(NOT "${actual_dependency_targets}" STREQUAL
       "${expected_dependency_targets}")
      message(FATAL_ERROR
        "Node internal archive ${target} has a non-canonical SCC dependency")
    endif()

    if(target_type STREQUAL "INTERFACE_LIBRARY")
      get_target_property(link_libraries ${target} INTERFACE_LINK_LIBRARIES)
    else()
      get_target_property(link_libraries ${target} LINK_LIBRARIES)
    endif()
    if(link_libraries MATCHES "-NOTFOUND$")
      set(link_libraries)
    endif()
    set(actual_scc_links)
    set(actual_external_links)
    foreach(link IN LISTS link_libraries)
      if(link MATCHES "^node-closure-")
        list(APPEND actual_scc_links "${link}")
      else()
        list(APPEND actual_external_links "${link}")
      endif()
    endforeach()
    if(NOT "${actual_scc_links}" STREQUAL
       "${expected_dependency_targets}")
      message(FATAL_ERROR
        "Node internal archive ${target} does not link its direct SCC DAG")
    endif()
    get_target_property(expected_external_links
      ${target} RUND_NODE_EXTERNAL_LINKS)
    if(expected_external_links MATCHES "-NOTFOUND$")
      set(expected_external_links)
    endif()
    if(NOT "${actual_external_links}" STREQUAL
       "${expected_external_links}")
      message(FATAL_ERROR
        "Node internal archive ${target} has non-canonical external links")
    endif()
  endforeach()

  set(sorted_materialized_components ${materialized_components})
  rund_node_scc_components(sorted_product_components "${product_root}")
  list(SORT sorted_materialized_components)
  list(SORT sorted_product_components)
  if(NOT "${sorted_materialized_components}" STREQUAL
     "${sorted_product_components}")
    message(FATAL_ERROR
      "Node internal archives do not materialize each active component exactly once")
  endif()

  rund_node_scc_components(product_components "${product_root}")
  get_target_property(product_scope node RUND_NODE_ARCHIVE_SCOPE)
  get_target_property(product_archive_components
    node RUND_NODE_ARCHIVE_COMPONENTS)
  if(NOT product_scope STREQUAL "closure" OR
     NOT "${product_archive_components}" STREQUAL "${product_components}")
    message(FATAL_ERROR
      "Node archive is not the active product closure")
  endif()
  rund_node_scc_link_target(product_test_target "${product_root}")
  if(product_test_target STREQUAL "node" OR
     NOT product_test_target STREQUAL "node-closure-runtime-product")
    message(FATAL_ERROR
      "Node product contracts do not use the internal product SCC aggregate")
  endif()
endfunction()

add_library(node STATIC)
rund_node_profile_root(rund_node_product_root product)
rund_node_archive_objects(node "${rund_node_product_root}" closure)
rund_node_installed_archive_context(node)

foreach(root IN LISTS RUND_NODE_SCCS)
  rund_node_internal_scc_target(target "${root}")
  if(RUND_NODE_SCC_${root}_COMPONENTS)
    add_library(${target} STATIC EXCLUDE_FROM_ALL)
    rund_node_archive_objects(${target} "${root}" direct)
    rund_node_internal_archive_context(${target} "${root}")
  else()
    # A dependency-only SCC is an honest graph join, not an archive containing
    # a dummy source. It therefore adds zero compile and link objects.
    add_library(${target} INTERFACE EXCLUDE_FROM_ALL)
    set_target_properties(${target} PROPERTIES
      RUND_NODE_SCC_ROOT "${root}"
      RUND_NODE_ARCHIVE_SCOPE "direct"
      RUND_NODE_ARCHIVE_COMPONENTS "")
  endif()
endforeach()
foreach(root IN LISTS RUND_NODE_SCCS)
  rund_node_internal_scc_target(target "${root}")
  rund_node_link_scc_dependencies(${target} "${root}")
endforeach()
rund_node_require_archive_graph("${rund_node_product_root}")

function(rund_node_link_profiles_target out_target out_root)
  if(NOT ARGN)
    message(FATAL_ERROR "Node link closure has no registry profiles")
  endif()

  set(required_components)
  foreach(profile IN LISTS ARGN)
    rund_node_profile_components(profile_components "${profile}")
    list(APPEND required_components ${profile_components})
  endforeach()
  list(REMOVE_DUPLICATES required_components)
  if(NOT required_components)
    set(${out_target} "" PARENT_SCOPE)
    set(${out_root} "-" PARENT_SCOPE)
    return()
  endif()

  set(best_root)
  set(best_count)
  foreach(candidate IN LISTS RUND_NODE_SCCS)
    rund_node_scc_components(candidate_components "${candidate}")
    set(covers TRUE)
    foreach(required IN LISTS required_components)
      list(FIND candidate_components "${required}" component_index)
      if(component_index EQUAL -1)
        set(covers FALSE)
        break()
      endif()
    endforeach()
    if(covers)
      list(LENGTH candidate_components candidate_count)
      if(NOT best_root OR candidate_count LESS best_count)
        set(best_root "${candidate}")
        set(best_count "${candidate_count}")
      endif()
    endif()
  endforeach()
  if(NOT best_root)
    message(FATAL_ERROR
      "Node registry profiles have no closed SCC owner: ${ARGN}")
  endif()

  rund_node_scc_link_target(target "${best_root}")
  set(${out_target} "${target}" PARENT_SCOPE)
  set(${out_root} "${best_root}" PARENT_SCOPE)
endfunction()

# The registry has one unversioned name for each current execution closure.
rund_node_profile_root(rund_node_accel_root accel)
rund_node_scc_link_target(rund_node_accel_archive "${rund_node_accel_root}")
if(NOT rund_node_accel_archive STREQUAL "node-closure-accel-execution")
  message(FATAL_ERROR
    "Node Accel execution SCC has a non-canonical archive owner")
endif()
rund_node_profile_root(rund_node_compute_root compute)
rund_node_scc_link_target(rund_node_compute_archive "${rund_node_compute_root}")
if(NOT rund_node_compute_archive STREQUAL "node-closure-compute-execution")
  message(FATAL_ERROR
    "Node Compute execution SCC has a non-canonical archive owner")
endif()
