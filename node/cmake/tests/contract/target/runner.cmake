function(rund_node_dispatch_object out target cases)
  set(dispatch_target "${target}-dispatch")
  if(TARGET ${dispatch_target})
    message(FATAL_ERROR
      "Node contract dispatch owner already exists: ${dispatch_target}")
  endif()
  add_library(${dispatch_target} OBJECT tests/contract/dispatch.cpp)
  set_target_properties(${dispatch_target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_compile_features(${dispatch_target} PRIVATE cxx_std_20)
  target_compile_definitions(${dispatch_target} PRIVATE
    RUND_NODE_TEST_CASES=\"${cases}\")
  set(${out} "$<TARGET_OBJECTS:${dispatch_target}>" PARENT_SCOPE)
endfunction()

function(rund_node_runner_object out)
  if(NOT TARGET node-contract-runner)
    add_library(node-contract-runner OBJECT tests/contract/main.cpp)
    set_target_properties(node-contract-runner PROPERTIES
      EXCLUDE_FROM_ALL TRUE)
    rund_node_compile_context(node-contract-runner)
    if(RUND_NODE_HAVE_METAL_SDK)
      target_compile_definitions(node-contract-runner PRIVATE
        RUND_NODE_HAVE_METAL_SDK=1)
    endif()
    if(RUND_NODE_HAVE_VULKAN_SDK)
      target_compile_definitions(node-contract-runner PRIVATE
        RUND_NODE_HAVE_VULKAN_SDK=1)
    endif()
  endif()
  get_target_property(runner_type node-contract-runner TYPE)
  if(NOT runner_type STREQUAL "OBJECT_LIBRARY")
    message(FATAL_ERROR
      "Node contract runner must have one OBJECT compile owner")
  endif()
  get_target_property(runner_definitions
    node-contract-runner COMPILE_DEFINITIONS)
  foreach(definition IN LISTS runner_definitions)
    if(definition MATCHES "^RUND_NODE_TEST_CASES")
      message(FATAL_ERROR
        "Node contract runner must not consume a target case table")
    endif()
  endforeach()
  set(${out} "$<TARGET_OBJECTS:node-contract-runner>" PARENT_SCOPE)
endfunction()
