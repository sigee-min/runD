#pragma once

#include <kernel/dispatch/orchestrator/signal.hpp>
#include <kernel/dispatch/worker/backend.hpp>

#include <span>

namespace rund::kernel {

struct Workspace;

struct RunPreparedProgramRequest {
  Workspace *workspace = nullptr;
  WorkerBackend worker_backend{};
  void *context = nullptr;
  DispatchFn dispatch = nullptr;
  FailureSignal *failure_signal = nullptr;
  bool collect_worker_stats = true;
  std::span<u32> worker_stats_sink{};
  std::span<u64> worker_start_offset_ns_sink{};
  std::span<u64> worker_elapsed_ns_sink{};
  std::span<u64> worker_tail_wait_ns_sink{};
  u32 minimum_partition_count = 1;
  bool require_no_allocation = false;
};

} // namespace rund::kernel
