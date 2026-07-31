#pragma once

#include <kernel/program/executor/prepare/state.hpp>
#include <kernel/program/executor/prepare/validation.hpp>

#include <type_traits>

namespace rund::kernel {

template <std::size_t Rank>
template <typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] SkeletonResult
PreparedEach<Rank>::run(Callback &&callback) const {
  if (!valid) {
    return skeleton_detail::InvalidResult(reason);
  }
  if (units == 0u) {
    return skeleton_detail::PlanResult(plan_each(index_space));
  }
  if (const char *const ready_reason =
          executor_detail::ValidatePreparedEachProgram(*this);
      ready_reason != nullptr) {
    return skeleton_detail::InvalidResult(ready_reason);
  }

  FailureSignal failure_signal{};
  ResetFailureSignal(failure_signal);
  using CallbackObject = std::remove_reference_t<Callback>;
  skeleton_detail::ScheduledEachContext<Rank, CallbackObject> context{
      .index_space = index_space,
      .boundary_alignment = exec.boundary_alignment,
      .callback = &callback,
      .failure_signal = &failure_signal,
  };
  const RunResult run_result = RunPreparedProgram(RunPreparedProgramRequest{
      .workspace = exec.workspace,
      .worker_backend = exec.worker_backend,
      .context = &context,
      .dispatch = skeleton_detail::InvokeScheduledEach<Rank, CallbackObject>,
      .failure_signal = &failure_signal,
      .collect_worker_stats = exec.collect_worker_stats,
      .minimum_partition_count = 1u,
      .require_no_allocation = true,
  });
  if (!run_result.ok) {
    return skeleton_detail::InvalidResult(run_result.reason);
  }

  SkeletonResult result = skeleton_detail::PlanResult(plan_each(index_space));
  result.partition_boundary_checked = true;
  result.partition_boundary_aligned = true;
  result.boundary_alignment_units = exec.boundary_alignment.units;
  result.physical_tile_units =
      physical_tiling_enabled
          ? physical_tile_units
          : (partition_count == 0u
                 ? 0u
                 : (units + partition_count - 1u) / partition_count);
  return result;
}

template <std::size_t Rank>
template <typename Callback>
  requires std::is_invocable_r_v<void, Callback &, const Partition &>
[[nodiscard]] SkeletonResult
PreparedEach<Rank>::run_partitions(Callback &&callback) const {
  if (!valid) {
    return skeleton_detail::InvalidResult(reason);
  }
  if (units == 0u) {
    return skeleton_detail::PlanResult(plan_each(index_space));
  }
  if (const char *const ready_reason =
          executor_detail::ValidatePreparedEachProgram(*this);
      ready_reason != nullptr) {
    return skeleton_detail::InvalidResult(ready_reason);
  }

  FailureSignal failure_signal{};
  ResetFailureSignal(failure_signal);
  using CallbackObject = std::remove_reference_t<Callback>;
  struct PartitionContext {
    CallbackObject *callback = nullptr;
    FailureSignal *failure_signal = nullptr;
  } context{
      .callback = &callback,
      .failure_signal = &failure_signal,
  };
  const WorkerTask task{
      .context = &context,
      .invoke =
          [](void *const raw, const Partition &partition) {
            auto *const typed = static_cast<PartitionContext *>(raw);
            if (typed == nullptr || typed->callback == nullptr ||
                typed->failure_signal == nullptr ||
                HasFailure(*typed->failure_signal)) {
              return;
            }
            try {
              (*typed->callback)(partition);
            } catch (...) {
              MarkFailure(*typed->failure_signal,
                          "prepared_partition_callback_failed");
            }
          },
  };
  const RunResult run_result = RunPreparedProgram(RunPreparedProgramRequest{
      .workspace = exec.workspace,
      .worker_backend = exec.worker_backend,
      .context = task.context,
      .dispatch = task.invoke,
      .failure_signal = &failure_signal,
      .collect_worker_stats = exec.collect_worker_stats,
      .minimum_partition_count = 1u,
      .require_no_allocation = true,
  });
  if (!run_result.ok) {
    return skeleton_detail::InvalidResult(run_result.reason);
  }

  SkeletonResult result = skeleton_detail::PlanResult(plan_each(index_space));
  result.partition_boundary_checked = true;
  result.partition_boundary_aligned = true;
  result.boundary_alignment_units = exec.boundary_alignment.units;
  result.physical_tile_units =
      physical_tiling_enabled
          ? physical_tile_units
          : (partition_count == 0u
                 ? 0u
                 : (units + partition_count - 1u) / partition_count);
  return result;
}

} // namespace rund::kernel
