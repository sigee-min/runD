set(KERNEL_CORE_TEST_SOURCES
  tests/contract/run/core.cpp
  tests/contract/core/checked.cpp
  tests/contract/orchestrator.cpp
  tests/contract/orchestrator/failure.cpp
)

set(KERNEL_EXECUTION_TEST_SOURCES
  tests/contract/run/execution.cpp
  tests/contract/dispatch.cpp
  tests/contract/dispatch/failure.cpp
  tests/contract/dispatch/partition.cpp
  tests/contract/dispatch/telemetry.cpp
  tests/contract/schedule.cpp
  tests/contract/schedule/capacity.cpp
  tests/contract/schedule/dispatch.cpp
  tests/contract/schedule/weighted.cpp
  tests/contract/workspace/balance.cpp
  tests/contract/workspace/capacity.cpp
  tests/contract/workspace/hints.cpp
  tests/contract/workspace/packets.cpp
  tests/contract/workspace/runner.cpp
  tests/contract/workspace/weighted.cpp
  tests/contract/workspace/weighted/alias.cpp
  tests/contract/workspace/weighted/basic.cpp
  tests/contract/workspace/weighted/range.cpp
  tests/contract/workspace/weighted/source.cpp
  tests/contract/workspace/weighted/ties.cpp
)

set(kernel_core_contract_sources
  ${KERNEL_CORE_TEST_SOURCES}
  ${KERNEL_EXECUTION_TEST_SOURCES})
list(LENGTH kernel_core_contract_sources kernel_core_contract_source_count)
list(REMOVE_DUPLICATES kernel_core_contract_sources)
list(LENGTH kernel_core_contract_sources
  kernel_core_contract_unique_source_count)
if(NOT kernel_core_contract_source_count EQUAL
   kernel_core_contract_unique_source_count)
  message(FATAL_ERROR
    "Kernel core and execution contracts must own disjoint sources")
endif()
unset(kernel_core_contract_sources)
unset(kernel_core_contract_source_count)
unset(kernel_core_contract_unique_source_count)
