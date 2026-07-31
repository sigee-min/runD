#pragma once

#include "capabilities.hpp"
#include "execute.hpp"

namespace kernel_contract_test {

inline rund::kernel::WorkerBackend MakeFakeBackend(FakePool* pool) {
  if (pool == nullptr) {
    return {};
  }
  return rund::kernel::WorkerBackend{
      .context = pool,
      .worker_count = FakeWorkerCount,
      .affinity_policy = FakeAffinityPolicy,
      .capabilities = FakeCapabilities,
      .is_nested = FakeIsNested,
      .execute_partitions = FakeExecutePartitions,
  };
}

} // namespace kernel_contract_test
