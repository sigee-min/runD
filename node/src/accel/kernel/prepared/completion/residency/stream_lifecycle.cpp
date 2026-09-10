#include "internal.hpp"

#include <array>
#include <mutex>

namespace rund::node::accel::detail {

rund::AccelCheck ReleasePreparedKernelPipelineStream(
    PreparedResidencyStreamControl &control,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation, const bool quarantine) noexcept {
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  std::size_t count = 0u;
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.plan_identity != plan_identity ||
        control.token != token || control.generation != generation) {
      return {false, "accel_kernel_run_invalid"};
    }
    states = control.states;
    count = control.state_count;
  }
  std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> claims{};
  for (std::size_t index = 0u; index < count; ++index) {
    claims[index] =
        std::unique_lock<std::mutex>{states[index]->submission.mutex};
  }
  std::lock_guard lock{control.gate};
  if (!control.active || control.plan_identity != plan_identity ||
      control.token != token || control.generation != generation) {
    return {false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < count; ++index) {
    prepared::PipelineSubmission &submission = states[index]->submission;
    if (submission.stream != &control || submission.window != nullptr) {
      return {false, "compute_pipeline_busy"};
    }
  }
  const bool retain = quarantine || control.quarantined;
  for (std::size_t index = 0u; index < count; ++index) {
    prepared::PipelineSubmission &submission = states[index]->submission;
    submission.stream = nullptr;
    submission.quarantined = retain;
    if (!retain) {
      submission.owner.reset();
    }
  }
  control.states.fill(nullptr);
  control.owners = {};
  control.state_count = 0u;
  control.plan_identity = 0u;
  control.token = 0u;
  control.generation = 0u;
  control.active = false;
  control.quarantined = retain;
  return {true, "ok"};
}

rund::AccelCheck QuarantinePreparedKernelPipelineStream(
    PreparedResidencyStreamControl &control,
    const std::uint64_t plan_identity, const std::uint64_t token,
    const std::uint64_t generation) noexcept {
  std::array<prepared::PipelineState *, ResidencyWindowCapacity> states{};
  std::size_t count = 0u;
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.plan_identity != plan_identity ||
        control.token != token || control.generation != generation) {
      return {false, "accel_kernel_run_invalid"};
    }
    states = control.states;
    count = control.state_count;
  }
  std::array<std::unique_lock<std::mutex>, ResidencyWindowCapacity> claims{};
  for (std::size_t index = 0u; index < count; ++index) {
    claims[index] =
        std::unique_lock<std::mutex>{states[index]->submission.mutex};
  }
  std::lock_guard lock{control.gate};
  if (!control.active || control.plan_identity != plan_identity ||
      control.token != token || control.generation != generation) {
    return {false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < count; ++index) {
    prepared::PipelineSubmission &submission = states[index]->submission;
    if (submission.stream == &control) {
      submission.stream = nullptr;
    }
    submission.window = nullptr;
    submission.quarantined = true;
    // `owner` deliberately remains strong for the quarantined native state.
  }
  control.states.fill(nullptr);
  control.owners = {};
  control.state_count = 0u;
  control.plan_identity = 0u;
  control.token = 0u;
  control.generation = 0u;
  control.active = false;
  control.quarantined = true;
  return {true, "ok"};
}

} // namespace rund::node::accel::detail
