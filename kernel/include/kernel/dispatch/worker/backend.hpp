#pragma once

#include <kernel/dispatch/worker/capabilities.hpp>
#include <kernel/dispatch/worker/stats.hpp>
#include <kernel/dispatch/worker/submission.hpp>
#include <kernel/dispatch/worker/task.hpp>

namespace rund::kernel {

struct WorkerBackend {
  void* context = nullptr;
  u32 (*worker_count)(void* context) = nullptr;
  WorkerAffinityPolicy (*affinity_policy)(void* context) = nullptr;
  WorkerBackendCapabilities (*capabilities)(void* context) = nullptr;
  bool (*is_nested)(void* context) = nullptr;
  bool (*execute_partitions)(void* context,
                             const Partition* partitions,
                             u32 partition_count,
                             WorkerTask task,
                             WorkerStats* out_stats) = nullptr;
  bool (*submit_partitions)(void* context,
                            const Partition* partitions,
                            u32 partition_count,
                            WorkerTask task,
                            WorkerStats* out_stats,
                            WorkerSubmission* submission) = nullptr;

  [[nodiscard]] explicit operator bool() const {
    return context != nullptr &&
           worker_count != nullptr &&
           affinity_policy != nullptr &&
           (execute_partitions != nullptr || submit_partitions != nullptr);
  }
};

} // namespace rund::kernel
