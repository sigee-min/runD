#pragma once

#include <kernel/program/executor/factory.hpp>
#include <kernel/program/executor/prepare.hpp>

#include <utility>

namespace rund::kernel {

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(const Executor& exec,
                                         const Space<Rank>& index_space,
                                         Callback&& callback) {
  const PreparedEach<Rank> prepared = exec.prepare(index_space);
  return prepared.run(std::forward<Callback>(callback));
}

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(const SerialPolicy policy,
                                         const Space<Rank>& index_space,
                                         Callback&& callback) {
  if (!policy.valid) {
    return skeleton_detail::InvalidResult(policy.reason);
  }
  return each(index_space, std::forward<Callback>(callback));
}

template <std::size_t Rank, typename Callback>
  requires skeleton_detail::DirectCallback<Callback>
[[nodiscard]] inline SkeletonResult each(const ParallelPolicy policy,
                                         const Space<Rank>& index_space,
                                         Callback&& callback) {
  if (!policy.valid) {
    return skeleton_detail::InvalidResult(policy.reason);
  }
  const ParallelRuntime runtime =
      executor_detail::AcquireParallelRuntime(policy.use_runtime_default_workers
                                                  ? 0u
                                                  : policy.workers);
  if (!runtime.valid) {
    return skeleton_detail::InvalidResult(runtime.reason);
  }
  if (runtime.workspace == nullptr) {
    return skeleton_detail::InvalidResult("parallel_runtime_workspace_missing");
  }
  if (!runtime.worker_backend) {
    return skeleton_detail::InvalidResult("parallel_runtime_backend_invalid");
  }
  const Executor exec = executor(*runtime.workspace,
                                 runtime.worker_backend,
                                 runtime.workers,
                                 policy.boundary_alignment,
                                 policy.physical_tile_policy);
  return each(exec, index_space, std::forward<Callback>(callback));
}

} // namespace rund::kernel
