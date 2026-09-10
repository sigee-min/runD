#include "internal.hpp"

#include <array>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck ClaimPreparedKernelPipelineStream(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelPipeline> pipelines,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation,
    PreparedResidencyStreamControl &control) noexcept {
  if (pipelines.empty() || pipelines.size() > ResidencyWindowCapacity ||
      plan_identity == 0u || token == 0u || generation == 0u) {
    return {false, "accel_kernel_run_invalid"};
  }
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  std::array<PreparedKernelPipeline, ResidencyWindowCapacity> owners{};
  std::size_t count = 0u;
  for (const PreparedKernelPipeline &pipeline : pipelines) {
    auto *const state =
        static_cast<prepared::PipelineState *>(pipeline.owner.get());
    if (!pipeline.ok || state == nullptr ||
        !prepared::ValidPipeline(context, *state)) {
      return {false, "accel_kernel_run_invalid"};
    }
    bool found = false;
    for (std::size_t index = 0u; index < count; ++index) {
      found = found || states[index] == state;
    }
    if (!found) {
      states[count] = state;
      owners[count] = pipeline;
      ++count;
    }
  }
  for (std::size_t left = 0u; left < count; ++left) {
    for (std::size_t right = left + 1u; right < count; ++right) {
      if (std::less<prepared::PipelineState *>{}(states[right], states[left])) {
        std::swap(states[left], states[right]);
        std::swap(owners[left], owners[right]);
      }
    }
  }
  std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> claims{};
  for (std::size_t index = 0u; index < count; ++index) {
    claims[index] = std::unique_lock<std::mutex>{states[index]->submission.mutex};
  }
  std::lock_guard stream_lock{control.gate};
  if (control.active) {
    return {false, "compute_pipeline_busy"};
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (states[index]->submission.active()) {
      return {false, "compute_pipeline_busy"};
    }
  }
  control.states = states;
  control.owners = owners;
  control.state_count = count;
  control.plan_identity = plan_identity;
  control.token = token;
  control.generation = generation;
  control.active = true;
  control.quarantined = false;
  for (std::size_t index = 0u; index < count; ++index) {
    states[index]->submission.owner = owners[index].owner;
    states[index]->submission.stream = &control;
  }
  return {true, "ok"};
}

rund::AccelCheck SubmitPreparedKernelPipelineStreamWindow(
    const rund::AccelContext &context,
    const PreparedResidencyWindowRequest &request,
    PreparedResidencyWindowControl &window,
    PreparedResidencyStreamControl &stream) noexcept {
  {
    std::scoped_lock lock{stream.gate, window.gate};
    if (!stream.active || stream.quarantined || window.active ||
        request.plan_identity != stream.plan_identity ||
        request.token != stream.token || request.generation != stream.generation) {
      return {false, "accel_kernel_run_invalid"};
    }
    window.stream = &stream;
  }
  const rund::AccelCheck submitted =
      SubmitPreparedKernelPipelineWindow(context, request, window);
  if (!submitted.ok) {
    std::lock_guard lock{window.gate};
    if (!window.active) {
      window.stream = nullptr;
    }
  }
  return submitted;
}

} // namespace rund::node::accel::detail
