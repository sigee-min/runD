function(rund_register_consumer target expected resource)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "package consumer target is unavailable: ${target}")
  endif()
  if(NOT expected MATCHES "^[0-9]+$" OR
     NOT resource MATCHES "^(general|accel)$")
    message(FATAL_ERROR
      "package consumer execution contract is invalid: ${target}")
  endif()
  set_property(GLOBAL APPEND PROPERTY RUND_PACKAGE_TARGETS "${target}")
  set_property(GLOBAL APPEND PROPERTY RUND_PACKAGE_RUNS
               "${target}|${expected}|${resource}")
endfunction()

function(rund_add_consumer target source expected resource)
  add_executable("${target}" "${source}")
  target_link_libraries("${target}" PRIVATE runD::sdk)
  rund_register_consumer("${target}" "${expected}" "${resource}")
endfunction()

rund_add_consumer(rund_package_sdk_consumer sdk.cpp 0 general)
rund_add_consumer(rund_package_server_consumer contract/server.cpp 0 general)
rund_add_consumer(rund_package_compute_consumer compute.cpp 0 accel)
target_sources(rund_package_compute_consumer
  PRIVATE
    compute/flow/primitives.cpp
    compute/flow/primitives/basic.cpp
    compute/flow/primitives/collective.cpp
    compute/flow/primitives/compact.cpp
    compute/flow/primitives/filter.cpp
    compute/flow/primitives/group.cpp
    compute/flow/primitives/math.cpp
    compute/flow/primitives/program.cpp
    compute/flow/primitives/record.cpp
    compute/flow/primitives/surface.cpp)
rund_add_consumer(rund_package_pipeline_consumer example/pipeline.cpp 0 general)
rund_add_consumer(rund_package_device_program_consumer
                  example/device/program.cpp 0 accel)
target_sources(rund_package_device_program_consumer
  PRIVATE
    example/device/program/attribution.cpp
    example/device/program/execute.cpp
    example/device/program/failure.cpp
    example/device/program/profile.cpp
    support/allocation.cpp)
rund_add_consumer(rund_package_blackbox_consumer blackbox.cpp 0 general)
target_sources(rund_package_blackbox_consumer
  PRIVATE
    blackbox/cluster.cpp
    blackbox/math.cpp
    blackbox/network.cpp
    blackbox/replay.cpp)
rund_add_consumer(rund_package_replay_consumer example/replay.cpp 0 general)
rund_add_consumer(rund_package_scenario_consumer example/scenario.cpp 0 general)
rund_add_consumer(rund_package_checkpoint_consumer example/checkpoint.cpp 0 general)
rund_add_consumer(rund_package_history_consumer example/history.cpp 0 general)
rund_add_consumer(rund_package_failure_consumer exit/failure.cpp 1 general)
rund_add_consumer(rund_package_assertion_consumer exit/assertion.cpp 2 general)
