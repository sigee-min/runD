#include "../local.hpp"

#include <chrono>

namespace rund::kernel::internal {

Result Execute(const Plan& plan) {
  Result validation = dispatch::detail::ValidatePlan(plan);
  if (!validation.ok) {
    return validation;
  }

  if (!plan.worker_backend &&
      (plan.execution_width == 1u || (plan.partition_count == 1u && !plan.collect_worker_stats))) {
    const auto start = std::chrono::steady_clock::now();
    Result out = dispatch::detail::ExecuteSerial(plan, validation.telemetry);
    const auto end = std::chrono::steady_clock::now();
    out.telemetry.dispatch_cost_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    out.telemetry.dispatch_cost_measured = true;
    return out;
  }

  WorkerStats stats{};
  WorkerStats* const stats_sink =
      dispatch::detail::PrepareWorkerStats(plan.collect_worker_stats,
                                           plan.worker_stats_sink,
                                           plan.worker_start_offset_ns_sink,
                                           plan.worker_elapsed_ns_sink,
                                           plan.worker_tail_wait_ns_sink,
                                           stats);
  dispatch::detail::DispatchAdapter adapter{};
  const WorkerTask task =
      dispatch::detail::BuildDispatchTask(adapter,
                                          plan.context,
                                          plan.dispatch,
                                          plan.ordered_packet_indices,
                                          plan.ordered_packet_count);
  const auto start = std::chrono::steady_clock::now();
  const bool dispatched =
      dispatch::detail::ExecuteBackendPartitions(plan.worker_backend,
                                                 plan.partitions,
                                                 plan.partition_count,
                                                 task,
                                                 stats_sink);
  const auto end = std::chrono::steady_clock::now();
  dispatch::detail::RecordBackendDispatch(
      validation,
      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
      stats_sink);
  if (!dispatched) {
    validation.ok = false;
    validation.failure_reason = "dispatch_failed";
    return validation;
  }
  validation.failure_reason = "pass";
  return validation;
}

} // namespace rund::kernel::internal
