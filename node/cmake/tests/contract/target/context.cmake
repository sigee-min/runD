function(rund_node_compile_context target)
  cmake_parse_arguments(
    RUND_CONTEXT "CASE;PLATFORM;MATH" "LINK_ROOT" "" ${ARGN})
  get_property(context_set TARGET ${target}
    PROPERTY RUND_NODE_COMPILE_CONTEXT SET)
  if(context_set)
    message(FATAL_ERROR
      "Node compile context already exists: ${target}")
  endif()
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tools"
    "${CMAKE_CURRENT_SOURCE_DIR}/tooling"
    "${CMAKE_SOURCE_DIR}/accel/include"
    "${CMAKE_SOURCE_DIR}/kernel/include"
    "${CMAKE_SOURCE_DIR}/math32/include"
    "${CMAKE_SOURCE_DIR}/math64/include")
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_compile_definitions(${target} PRIVATE
    RUND_NODE_INTERNAL_BUILD=1
    RUND_NODE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  if(RUND_CONTEXT_CASE AND RUND_NODE_FOCUSED_BACKEND STREQUAL "cpu")
    target_compile_definitions(${target} PRIVATE
      RUND_NODE_TEST_BACKEND_CPU=1)
  endif()
  if(RUND_CONTEXT_PLATFORM AND RUND_NODE_USE_UNAVAILABLE_PLATFORM)
    target_compile_definitions(${target} PRIVATE
      RUND_NODE_PLATFORM_UNAVAILABLE=1)
  endif()

  set(needs_vulkan_sdk FALSE)
  if(RUND_CONTEXT_CASE)
    if(RUND_CONTEXT_LINK_ROOT STREQUAL "ACCEL_EXECUTION" OR
       RUND_CONTEXT_LINK_ROOT STREQUAL "COMPUTE_EXECUTION" OR
       RUND_CONTEXT_LINK_ROOT STREQUAL "RUNTIME_PRODUCT")
      set(needs_vulkan_sdk TRUE)
    endif()
    if(NOT RUND_NODE_FOCUSED_BACKEND STREQUAL "" AND
       NOT RUND_NODE_FOCUSED_BACKEND STREQUAL "vulkan")
      set(needs_vulkan_sdk FALSE)
    endif()
    if(needs_vulkan_sdk AND RUND_NODE_HAVE_VULKAN_SDK)
      target_compile_definitions(${target} PRIVATE
        RUND_NODE_HAVE_VULKAN_SDK=1)
      if(RUND_NODE_GLSLANG_VALIDATOR)
        target_compile_definitions(${target} PRIVATE
          "RUND_NODE_TEST_GLSLANG_VALIDATOR_PATH=\"${RUND_NODE_GLSLANG_VALIDATOR}\"")
      endif()
      # Production Accel closure owns the Vulkan link. Contract sources need
      # only its header search path; linking the imported target here would
      # repeat the same native library in every Accel/Compute product command.
      target_include_directories(${target} SYSTEM PRIVATE
        "$<TARGET_PROPERTY:${RUND_NODE_VULKAN_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
  endif()

  if(RUND_CONTEXT_MATH)
    target_link_libraries(${target} PRIVATE math32 math64)
  endif()

  set_property(TARGET ${target} PROPERTY RUND_NODE_COMPILE_CONTEXT
    "case=${RUND_CONTEXT_CASE}|platform=${RUND_CONTEXT_PLATFORM}|math=${RUND_CONTEXT_MATH}|root=${RUND_CONTEXT_LINK_ROOT}|backend=${RUND_NODE_FOCUSED_BACKEND}|unavailable=${RUND_NODE_USE_UNAVAILABLE_PLATFORM}|vulkan=${needs_vulkan_sdk}")
endfunction()
