#pragma once

#include "stats.hpp"

#include <kernel/internal/workspace/schedule.hpp>
#include <kernel/schedule/workspace.hpp>

#include <vector>

namespace kernel_contract_test {

inline bool FakeExecutePartitions(void* context,
                                  const rund::kernel::Partition* partitions,
                                  const rund::kernel::u32 partition_count,
                                  const rund::kernel::WorkerTask task,
                                  rund::kernel::WorkerStats* out_stats) {
  FakePool* const pool = static_cast<FakePool*>(context);
  if (pool == nullptr || partitions == nullptr || !task) {
    return false;
  }
  std::vector<rund::kernel::u32> allocated_partitions_per_worker{};
  std::span<rund::kernel::u32> partitions_per_worker{};
  if (out_stats != nullptr &&
      !out_stats->partitions_per_worker_sink.empty() &&
      out_stats->partitions_per_worker_sink.size() >= pool->workers) {
    partitions_per_worker = out_stats->partitions_per_worker_sink.subspan(0u, pool->workers);
    for (rund::kernel::u32& count : partitions_per_worker) {
      count = 0u;
    }
  } else {
    allocated_partitions_per_worker.assign(pool->workers, 0u);
    partitions_per_worker = std::span<rund::kernel::u32>(allocated_partitions_per_worker.data(),
                                                   allocated_partitions_per_worker.size());
  }
  for (rund::kernel::u32 index = 0u; index < partition_count; ++index) {
    const rund::kernel::Partition& partition = partitions[index];
    if (partition.end <= partition.begin) {
      continue;
    }
    task.invoke(task.context, partition);
    if (!partitions_per_worker.empty()) {
      const rund::kernel::u32 execution_worker =
          index % static_cast<rund::kernel::u32>(partitions_per_worker.size());
      partitions_per_worker[execution_worker] += 1u;
    }
  }
  PublishFakeStats(pool->workers,
                   std::span<const rund::kernel::u32>(partitions_per_worker.data(),
                                                partitions_per_worker.size()),
                   out_stats);
  return true;
}

} // namespace kernel_contract_test
