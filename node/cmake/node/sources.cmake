set(RUND_NODE_USE_UNAVAILABLE_PLATFORM FALSE)
if(RUND_FORCE_UNAVAILABLE_PLATFORM OR NOT UNIX)
  set(RUND_NODE_USE_UNAVAILABLE_PLATFORM TRUE)
endif()

# Each source fragment is loaded exactly once into one OBJECT component.  The
# final product and canonical closure archives consume those same objects;
# they never compile a second copy of a product translation unit.
set(RUND_NODE_COMPONENTS)

function(rund_node_component id name)
  if(NOT id MATCHES "^[A-Z][A-Z0-9_]*$" OR
     NOT name MATCHES "^[a-z][a-z0-9-]*$")
    message(FATAL_ERROR "Malformed Node component: ${id}|${name}")
  endif()
  list(FIND RUND_NODE_COMPONENTS "${id}" duplicate_component)
  if(NOT duplicate_component EQUAL -1)
    message(FATAL_ERROR "Duplicate Node component: ${id}")
  endif()

  set(fragments)
  foreach(argument IN LISTS ARGN)
    list(APPEND fragments "${argument}")
  endforeach()
  if(NOT fragments)
    message(FATAL_ERROR "Node component ${id} has no source fragments")
  endif()

  set(NODE_SOURCES)
  foreach(fragment IN LISTS fragments)
    set(path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/${fragment}")
    if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
      message(FATAL_ERROR "Missing Node source fragment: ${path}")
    endif()
    include("${path}")
  endforeach()
  if(NOT NODE_SOURCES)
    message(FATAL_ERROR "Node component ${id} has no sources")
  endif()

  set(RUND_NODE_COMPONENTS ${RUND_NODE_COMPONENTS} "${id}" PARENT_SCOPE)
  set("RUND_NODE_COMPONENT_${id}_NAME" "${name}" PARENT_SCOPE)
  set("RUND_NODE_COMPONENT_${id}_SOURCES" ${NODE_SOURCES} PARENT_SCOPE)
endfunction()

rund_node_component(NUMERIC numeric sources/evidence.cmake)
rund_node_component(TELEMETRY telemetry sources/telemetry.cmake)
rund_node_component(ACCEL_SIMD accel-simd sources/accel/simd.cmake)
rund_node_component(ACCEL_CORE accel-core sources/accel/core.cmake)
rund_node_component(ACCEL_CPU accel-cpu sources/accel/cpu.cmake)
rund_node_component(ACCEL_CONTEXT accel-context sources/accel/context.cmake)
rund_node_component(ACCEL_PICK accel-pick sources/accel/pick.cmake)
rund_node_component(ACCEL_FAKE accel-fake sources/accel/fake.cmake)
rund_node_component(ACCEL_METAL accel-metal sources/accel/metal.cmake)
rund_node_component(ACCEL_VULKAN accel-vulkan sources/accel/vulkan.cmake)
rund_node_component(COMPUTE_CPU compute-cpu sources/compute.cmake)
rund_node_component(COMPUTE_ACCEL compute-accel sources/compute/accel.cmake)
rund_node_component(WORKER_BACKEND worker-backend
  sources/worker/backend.cmake)
rund_node_component(HOST host sources/host.cmake)
rund_node_component(RUNTIME_BASE runtime-base sources/runtime.cmake)
rund_node_component(RUNTIME_COMPUTE runtime-compute
  sources/runtime/compute.cmake)
rund_node_component(SCHEDULER_CHANNEL scheduler-channel
  sources/scheduler/channel.cmake)
rund_node_component(SCHEDULER_CANCEL scheduler-cancel
  sources/scheduler/cancel.cmake)
rund_node_component(SCHEDULER_CORE scheduler-core
  sources/scheduler/core.cmake)
rund_node_component(SCHEDULER_TASK scheduler-task
  sources/scheduler/task.cmake)
rund_node_component(SCHEDULER_HOSTIO scheduler-hostio
  sources/scheduler/host/io.cmake)
rund_node_component(SCHEDULER_LANE scheduler-lane
  sources/scheduler/lane.cmake)
rund_node_component(SCHEDULER_REACTOR scheduler-reactor
  sources/scheduler/reactor.cmake)
rund_node_component(SCHEDULER_PROGRESS scheduler-progress
  sources/scheduler/progress.cmake)
rund_node_component(SCHEDULER_STATE scheduler-state
  sources/scheduler/state.cmake)
rund_node_component(PLATFORM platform sources/platform.cmake)

set(RUND_NODE_NATIVE_COMPONENTS
  ACCEL_METAL
  ACCEL_VULKAN)

# SIMD, CPU Accel, and native Accel are three closed execution layers. Compute
# uses only SIMD for its typed CPU path; native selection and adapters remain
# above CPU Accel and never enter a CPU-only exact build.
set(RUND_NODE_SCCS
  NUMERIC
  TELEMETRY
  WORKER_BACKEND
  CPU_SIMD
  CPU_ACCEL
  ACCEL_EXECUTION
  CPU_COMPUTE
  COMPUTE_EXECUTION
  RUNTIME_BASE
  CPU_RUNTIME_PRODUCT
  RUNTIME_PRODUCT)
set(RUND_NODE_SCC_NUMERIC_COMPONENTS NUMERIC)
set(RUND_NODE_SCC_NUMERIC_DEPENDS)
set(RUND_NODE_SCC_TELEMETRY_COMPONENTS TELEMETRY)
set(RUND_NODE_SCC_TELEMETRY_DEPENDS)
set(RUND_NODE_SCC_WORKER_BACKEND_COMPONENTS WORKER_BACKEND)
set(RUND_NODE_SCC_WORKER_BACKEND_DEPENDS)
set(RUND_NODE_SCC_CPU_SIMD_COMPONENTS ACCEL_SIMD)
set(RUND_NODE_SCC_CPU_SIMD_DEPENDS)
set(RUND_NODE_SCC_CPU_ACCEL_COMPONENTS
  ACCEL_CORE
  ACCEL_CPU
  ACCEL_CONTEXT)
set(RUND_NODE_SCC_CPU_ACCEL_DEPENDS CPU_SIMD)
set(RUND_NODE_SCC_ACCEL_EXECUTION_COMPONENTS
  ACCEL_PICK
  ACCEL_FAKE
  ACCEL_METAL
  ACCEL_VULKAN)
set(RUND_NODE_SCC_ACCEL_EXECUTION_DEPENDS CPU_ACCEL)
set(RUND_NODE_SCC_CPU_COMPUTE_COMPONENTS COMPUTE_CPU)
set(RUND_NODE_SCC_CPU_COMPUTE_DEPENDS CPU_SIMD TELEMETRY WORKER_BACKEND)
set(RUND_NODE_SCC_COMPUTE_EXECUTION_COMPONENTS COMPUTE_ACCEL)
set(RUND_NODE_SCC_COMPUTE_EXECUTION_DEPENDS
  CPU_COMPUTE
  ACCEL_EXECUTION)
set(RUND_NODE_SCC_RUNTIME_BASE_COMPONENTS
  HOST
  RUNTIME_BASE
  SCHEDULER_CHANNEL
  SCHEDULER_CANCEL
  SCHEDULER_CORE
  SCHEDULER_TASK
  SCHEDULER_HOSTIO
  SCHEDULER_LANE
  SCHEDULER_REACTOR
  SCHEDULER_PROGRESS
  SCHEDULER_STATE
  PLATFORM)
set(RUND_NODE_SCC_RUNTIME_BASE_DEPENDS NUMERIC TELEMETRY WORKER_BACKEND)
set(RUND_NODE_SCC_CPU_RUNTIME_PRODUCT_COMPONENTS RUNTIME_COMPUTE)
set(RUND_NODE_SCC_CPU_RUNTIME_PRODUCT_DEPENDS RUNTIME_BASE CPU_COMPUTE)
# The full product is a zero-object DAG join. Runtime Compute is materialized
# exactly once by CPU_RUNTIME_PRODUCT; the native accelerator branch joins it
# through COMPUTE_EXECUTION without compiling a mirror implementation.
set(RUND_NODE_SCC_RUNTIME_PRODUCT_COMPONENTS)
set(RUND_NODE_SCC_RUNTIME_PRODUCT_DEPENDS
  CPU_RUNTIME_PRODUCT
  COMPUTE_EXECUTION)

include("${CMAKE_CURRENT_LIST_DIR}/profiles.cmake")

function(rund_node_scc_direct_components out scc)
  set(components ${RUND_NODE_SCC_${scc}_COMPONENTS})
  if(scc STREQUAL "ACCEL_EXECUTION" AND
     NOT "${RUND_NODE_FOCUSED_BACKEND}" STREQUAL "")
    rund_node_project_native_components(
      selected_native "${RUND_NODE_FOCUSED_BACKEND}")
    list(REMOVE_ITEM components ${RUND_NODE_NATIVE_COMPONENTS})
    list(APPEND components ${selected_native})
  endif()
  set(${out} ${components} PARENT_SCOPE)
endfunction()

function(rund_node_scc_components out scc)
  set(chain ${ARGN})
  list(FIND RUND_NODE_SCCS "${scc}" scc_index)
  if(scc_index EQUAL -1)
    message(FATAL_ERROR "Unknown Node source SCC: ${scc}")
  endif()
  list(FIND chain "${scc}" cycle_index)
  if(NOT cycle_index EQUAL -1)
    message(FATAL_ERROR "Cyclic Node source SCC dependency: ${chain};${scc}")
  endif()
  list(APPEND chain "${scc}")

  rund_node_scc_direct_components(closure "${scc}")
  foreach(dependency IN LISTS RUND_NODE_SCC_${scc}_DEPENDS)
    rund_node_scc_components(dependency_components "${dependency}" ${chain})
    list(APPEND closure ${dependency_components})
  endforeach()
  list(REMOVE_DUPLICATES closure)
  set(${out} ${closure} PARENT_SCOPE)
endfunction()
foreach(row IN LISTS RUND_NODE_LINK_PROFILE_ROWS)
  string(REPLACE "|" ";" fields "${row}")
  list(GET fields 0 profile)
  list(GET fields 1 root)
  if(NOT root STREQUAL "-")
    list(FIND RUND_NODE_SCCS "${root}" root_index)
    if(root_index EQUAL -1)
      message(FATAL_ERROR
        "Node link profile ${profile} has unknown SCC root ${root}")
    endif()
  endif()
endforeach()

function(rund_node_profile_root out profile)
  set(found FALSE)
  set(root)
  foreach(row IN LISTS RUND_NODE_LINK_PROFILE_ROWS)
    string(REPLACE "|" ";" fields "${row}")
    list(GET fields 0 candidate)
    if(candidate STREQUAL profile)
      list(GET fields 1 root)
      set(found TRUE)
      break()
    endif()
  endforeach()
  if(NOT found)
    message(FATAL_ERROR "Unknown Node link profile: ${profile}")
  endif()
  set(${out} "${root}" PARENT_SCOPE)
endfunction()

function(rund_node_profile_components out profile)
  rund_node_profile_root(root "${profile}")
  if(root STREQUAL "-")
    set(${out} "" PARENT_SCOPE)
    return()
  endif()
  rund_node_scc_components(components "${root}")
  set(${out} ${components} PARENT_SCOPE)
endfunction()

# Small semantic contracts must stay independent of the product execution
# closure. This is a configure-time guard against turning one numeric evidence
# edit back into an Accel/Compute/Scheduler rebuild.
rund_node_profile_root(rund_node_numeric_profile_root numeric)
rund_node_profile_components(rund_node_numeric_profile_components numeric)
list(LENGTH RUND_NODE_COMPONENT_NUMERIC_SOURCES
  rund_node_numeric_source_count)
if(NOT rund_node_numeric_profile_root STREQUAL "NUMERIC" OR
   NOT "${rund_node_numeric_profile_components}" STREQUAL "NUMERIC" OR
   NOT rund_node_numeric_source_count EQUAL 1)
  message(FATAL_ERROR
    "Node numeric profile must own exactly its one-source NUMERIC closure")
endif()
rund_node_profile_root(rund_node_runtime_profile_root runtime)
if(NOT rund_node_runtime_profile_root STREQUAL "RUNTIME_BASE")
  message(FATAL_ERROR "Node runtime profile lost its RUNTIME_BASE closure")
endif()
rund_node_profile_components(rund_node_runtime_profile_components runtime)
list(FIND rund_node_runtime_profile_components WORKER_BACKEND
  rund_node_runtime_worker_backend_index)
if(rund_node_runtime_worker_backend_index EQUAL -1)
  message(FATAL_ERROR
    "Node runtime profile lost the shared worker backend owner")
endif()
set(rund_node_expected_worker_backend_sources
  src/runtime/backend.cpp
  src/runtime/reason.cpp)
list(SORT rund_node_expected_worker_backend_sources)
set(rund_node_actual_worker_backend_sources
  ${RUND_NODE_COMPONENT_WORKER_BACKEND_SOURCES})
list(SORT rund_node_actual_worker_backend_sources)
if(NOT "${rund_node_actual_worker_backend_sources}" STREQUAL
       "${rund_node_expected_worker_backend_sources}")
  message(FATAL_ERROR
    "Node WORKER_BACKEND must own backend selection and shared reason code")
endif()
function(rund_node_components_source_count out)
  set(source_count 0)
  foreach(component IN LISTS ARGN)
    list(LENGTH RUND_NODE_COMPONENT_${component}_SOURCES component_count)
    math(EXPR source_count "${source_count} + ${component_count}")
  endforeach()
  set(${out} "${source_count}" PARENT_SCOPE)
endfunction()

# This is a structural contract, not a link-error oracle: the Accel profile
# must remain a proper dependency of Compute and must never absorb the public
# Compute implementation again.
rund_node_profile_root(rund_node_accel_profile_root accel)
rund_node_profile_root(rund_node_compute_profile_root compute)
rund_node_profile_root(rund_node_cpu_simd_profile_root cpu-simd)
rund_node_profile_root(rund_node_cpu_accel_profile_root cpu-accel)
rund_node_profile_root(rund_node_cpu_compute_profile_root cpu-compute)
rund_node_profile_root(rund_node_cpu_product_profile_root cpu-product)
if(NOT rund_node_accel_profile_root STREQUAL "ACCEL_EXECUTION" OR
   NOT rund_node_compute_profile_root STREQUAL "COMPUTE_EXECUTION" OR
   NOT rund_node_cpu_simd_profile_root STREQUAL "CPU_SIMD" OR
   NOT rund_node_cpu_accel_profile_root STREQUAL "CPU_ACCEL" OR
   NOT rund_node_cpu_compute_profile_root STREQUAL "CPU_COMPUTE" OR
   NOT rund_node_cpu_product_profile_root STREQUAL "CPU_RUNTIME_PRODUCT")
  message(FATAL_ERROR "Node execution profiles lost their canonical SCC roots")
endif()
rund_node_profile_components(rund_node_accel_profile_components accel)
rund_node_profile_components(rund_node_compute_profile_components compute)
rund_node_profile_components(rund_node_cpu_simd_profile_components cpu-simd)
rund_node_profile_components(rund_node_cpu_accel_profile_components cpu-accel)
rund_node_profile_components(rund_node_cpu_compute_profile_components cpu-compute)
rund_node_profile_components(rund_node_cpu_product_profile_components cpu-product)
rund_node_profile_components(rund_node_product_profile_components product)
set(rund_node_sorted_product_profile_components
  ${rund_node_product_profile_components})
set(rund_node_sorted_all_components ${RUND_NODE_COMPONENTS})
if(NOT "${RUND_NODE_FOCUSED_BACKEND}" STREQUAL "")
  rund_node_project_native_components(
    rund_node_selected_native_components "${RUND_NODE_FOCUSED_BACKEND}")
  list(REMOVE_ITEM rund_node_sorted_all_components
    ${RUND_NODE_NATIVE_COMPONENTS})
  list(APPEND rund_node_sorted_all_components
    ${rund_node_selected_native_components})
endif()
list(SORT rund_node_sorted_product_profile_components)
list(SORT rund_node_sorted_all_components)
if(NOT "${rund_node_sorted_product_profile_components}" STREQUAL
       "${rund_node_sorted_all_components}")
  message(FATAL_ERROR
    "Node product profile does not close over its active production projection")
endif()
foreach(component IN ITEMS COMPUTE_CPU COMPUTE_ACCEL)
  list(FIND rund_node_accel_profile_components "${component}"
    rund_node_accel_compute_index)
  if(NOT rund_node_accel_compute_index EQUAL -1)
    message(FATAL_ERROR
      "Node Accel execution profile contains Compute component ${component}")
  endif()
endforeach()
foreach(component IN ITEMS COMPUTE_CPU COMPUTE_ACCEL)
  list(FIND rund_node_compute_profile_components "${component}"
    rund_node_compute_component_index)
  if(rund_node_compute_component_index EQUAL -1)
    message(FATAL_ERROR
      "Node Compute execution profile lost component ${component}")
  endif()
endforeach()
foreach(component IN LISTS rund_node_accel_profile_components)
  list(FIND rund_node_compute_profile_components "${component}"
    dependency_index)
  if(dependency_index EQUAL -1)
    message(FATAL_ERROR
      "Node Compute execution profile does not close over Accel component ${component}")
  endif()
endforeach()
if(NOT "${RUND_NODE_FOCUSED_BACKEND}" STREQUAL "")
  rund_node_project_native_components(
    rund_node_selected_native_components "${RUND_NODE_FOCUSED_BACKEND}")
  foreach(component IN LISTS RUND_NODE_NATIVE_COMPONENTS)
    list(FIND rund_node_accel_profile_components "${component}"
      actual_index)
    list(FIND rund_node_selected_native_components "${component}"
      expected_index)
    if((actual_index EQUAL -1 AND NOT expected_index EQUAL -1) OR
       (NOT actual_index EQUAL -1 AND expected_index EQUAL -1))
      message(FATAL_ERROR
        "Node focused backend ${RUND_NODE_FOCUSED_BACKEND} has invalid native component ${component}")
    endif()
  endforeach()
endif()
list(LENGTH RUND_NODE_COMPONENT_ACCEL_SIMD_SOURCES
  rund_node_cpu_simd_source_count)
if(NOT "${rund_node_cpu_simd_profile_components}" STREQUAL "ACCEL_SIMD" OR
   NOT rund_node_cpu_simd_source_count EQUAL 5)
  message(FATAL_ERROR
    "Node CPU SIMD profile must own exactly its five-source ACCEL_SIMD closure")
endif()
set(rund_node_expected_cpu_accel_components
  ACCEL_SIMD
  ACCEL_CORE
  ACCEL_CPU
  ACCEL_CONTEXT)
set(rund_node_sorted_cpu_accel_components
  ${rund_node_cpu_accel_profile_components})
list(SORT rund_node_expected_cpu_accel_components)
list(SORT rund_node_sorted_cpu_accel_components)
if(NOT "${rund_node_sorted_cpu_accel_components}" STREQUAL
       "${rund_node_expected_cpu_accel_components}")
  message(FATAL_ERROR
    "Node CPU Accel profile must own only SIMD, generic, CPU, and context components")
endif()
set(rund_node_expected_cpu_compute_components
  ACCEL_SIMD
  COMPUTE_CPU
  TELEMETRY
  WORKER_BACKEND)
set(rund_node_sorted_cpu_compute_components
  ${rund_node_cpu_compute_profile_components})
list(SORT rund_node_expected_cpu_compute_components)
list(SORT rund_node_sorted_cpu_compute_components)
if(NOT "${rund_node_sorted_cpu_compute_components}" STREQUAL
       "${rund_node_expected_cpu_compute_components}")
  message(FATAL_ERROR
    "Node CPU Compute profile must own only SIMD, telemetry, CPU Compute, and worker components")
endif()
foreach(component IN LISTS rund_node_cpu_simd_profile_components)
  list(FIND rund_node_cpu_compute_profile_components "${component}"
    dependency_index)
  if(dependency_index EQUAL -1)
    message(FATAL_ERROR
      "Node CPU Compute profile does not close over CPU SIMD component ${component}")
  endif()
endforeach()
foreach(component IN LISTS rund_node_cpu_compute_profile_components)
  list(FIND rund_node_cpu_product_profile_components "${component}"
    dependency_index)
  if(dependency_index EQUAL -1)
    message(FATAL_ERROR
      "Node CPU Runtime Product does not close over CPU Compute component ${component}")
  endif()
endforeach()
rund_node_components_source_count(rund_node_accel_profile_source_count
  ${rund_node_accel_profile_components})
rund_node_components_source_count(rund_node_compute_profile_source_count
  ${rund_node_compute_profile_components})
rund_node_components_source_count(rund_node_cpu_simd_profile_source_count
  ${rund_node_cpu_simd_profile_components})
rund_node_components_source_count(rund_node_cpu_compute_profile_source_count
  ${rund_node_cpu_compute_profile_components})
list(LENGTH RUND_NODE_COMPONENT_COMPUTE_CPU_SOURCES
  rund_node_compute_cpu_source_count)
list(LENGTH RUND_NODE_COMPONENT_COMPUTE_ACCEL_SOURCES
  rund_node_compute_accel_source_count)
list(LENGTH RUND_NODE_COMPONENT_WORKER_BACKEND_SOURCES
  rund_node_worker_backend_source_count)
list(LENGTH RUND_NODE_COMPONENT_TELEMETRY_SOURCES
  rund_node_telemetry_source_count)
math(EXPR rund_node_expected_compute_profile_source_count
  "${rund_node_accel_profile_source_count} + ${rund_node_compute_cpu_source_count} + ${rund_node_compute_accel_source_count} + ${rund_node_telemetry_source_count} + ${rund_node_worker_backend_source_count}")
math(EXPR rund_node_expected_cpu_compute_profile_source_count
  "${rund_node_cpu_simd_profile_source_count} + ${rund_node_compute_cpu_source_count} + ${rund_node_telemetry_source_count} + ${rund_node_worker_backend_source_count}")
if(rund_node_accel_profile_source_count EQUAL 0 OR
   NOT rund_node_compute_profile_source_count EQUAL
       rund_node_expected_compute_profile_source_count OR
   NOT rund_node_cpu_compute_profile_source_count EQUAL
       rund_node_expected_cpu_compute_profile_source_count)
  message(FATAL_ERROR
    "Node Compute closures lost their exact CPU/full decomposition")
endif()

# Every selected-platform production source has one and only one component.
set(rund_node_owned_sources)
set(rund_node_owned_components)
foreach(component IN LISTS RUND_NODE_COMPONENTS)
  foreach(source IN LISTS RUND_NODE_COMPONENT_${component}_SOURCES)
    if(IS_ABSOLUTE "${source}")
      set(path "${source}")
    else()
      set(path "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endif()
    if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
      message(FATAL_ERROR
        "Node component ${component} owns missing source ${source}")
    endif()
    if(NOT source MATCHES "^src/.+\\.(cpp|mm)$")
      message(FATAL_ERROR
        "Node component ${component} owns invalid source ${source}")
    endif()
    list(FIND rund_node_owned_sources "${source}" source_index)
    if(NOT source_index EQUAL -1)
      list(GET rund_node_owned_components ${source_index} first_component)
      message(FATAL_ERROR
        "Node source ${source} has duplicate owners ${first_component} and ${component}")
    endif()
    list(APPEND rund_node_owned_sources "${source}")
    list(APPEND rund_node_owned_components "${component}")
  endforeach()
endforeach()

# GLOB is validation-only, never source admission authority.  Inactive platform
# alternatives are intentionally excluded from the selected build.
file(GLOB_RECURSE rund_node_live_sources
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm")
set(rund_node_expected_sources)
foreach(source IN LISTS rund_node_live_sources)
  if(source MATCHES "^src/runtime/platform/unavailable/" AND
     NOT RUND_NODE_USE_UNAVAILABLE_PLATFORM)
    continue()
  endif()
  if(source MATCHES "^src/runtime/platform/posix/" AND
     RUND_NODE_USE_UNAVAILABLE_PLATFORM)
    continue()
  endif()
  if(source MATCHES "^src/runtime/platform/linux/" AND
     (RUND_NODE_USE_UNAVAILABLE_PLATFORM OR
      NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    continue()
  endif()
  if(source MATCHES "^src/runtime/platform/mac/" AND
     (RUND_NODE_USE_UNAVAILABLE_PLATFORM OR
      NOT (APPLE OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")))
    continue()
  endif()
  if(source MATCHES "^src/runtime/platform/portable/" AND
     (RUND_NODE_USE_UNAVAILABLE_PLATFORM OR APPLE OR
      CMAKE_SYSTEM_NAME STREQUAL "FreeBSD" OR
      CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT UNIX))
    continue()
  endif()
  list(APPEND rund_node_expected_sources "${source}")
endforeach()
list(SORT rund_node_expected_sources)
set(rund_node_sorted_owned_sources ${rund_node_owned_sources})
list(SORT rund_node_sorted_owned_sources)
if(NOT "${rund_node_sorted_owned_sources}" STREQUAL
       "${rund_node_expected_sources}")
  set(missing ${rund_node_expected_sources})
  list(REMOVE_ITEM missing ${rund_node_sorted_owned_sources})
  set(extra ${rund_node_sorted_owned_sources})
  list(REMOVE_ITEM extra ${rund_node_expected_sources})
  message(FATAL_ERROR
    "Node product source ownership is not closed; missing=${missing}; extra=${extra}")
endif()

# Every component belongs to exactly one modeled SCC.
set(rund_node_scc_components)
foreach(scc IN LISTS RUND_NODE_SCCS)
  foreach(component IN LISTS RUND_NODE_SCC_${scc}_COMPONENTS)
    list(FIND RUND_NODE_COMPONENTS "${component}" component_index)
    if(component_index EQUAL -1)
      message(FATAL_ERROR "Node SCC ${scc} owns unknown component ${component}")
    endif()
    list(FIND rund_node_scc_components "${component}" duplicate_scc)
    if(NOT duplicate_scc EQUAL -1)
      message(FATAL_ERROR "Node component ${component} has duplicate SCC owners")
    endif()
    list(APPEND rund_node_scc_components "${component}")
  endforeach()
  rund_node_scc_components(closure_probe "${scc}")
endforeach()
set(rund_node_sorted_components ${RUND_NODE_COMPONENTS})
list(SORT rund_node_sorted_components)
list(SORT rund_node_scc_components)
if(NOT "${rund_node_sorted_components}" STREQUAL
       "${rund_node_scc_components}")
  message(FATAL_ERROR "Node component-to-SCC ownership is not closed")
endif()
