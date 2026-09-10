#include <rund/compute/pipeline.hpp>

#include "../backend.hpp"
#include "../status.hpp"
#include "local.hpp"
#include "state.hpp"

#include <cstdint>

namespace rund::compute::detail {

Status seed_pipeline_generations(PipelineState &state,
                                 const std::uint64_t generation,
                                 const std::uint8_t parity,
                                 Location *const location) noexcept {
  if (generation > PipelineGenerationCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (parity > 1u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Empty and CPU Pipelines intentionally own no native generation control.
  if (state.device->backend == Backend::Cpu || state.active_step_count == 0u) {
    state.native_generation = generation;
    state.native_parity = parity;
    return Status::success();
  }
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr || ops->seed_pipeline_generation == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t current = static_cast<std::uint32_t>(generation);
  const std::uint32_t selected_seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const std::uint32_t other_seed = current;
  const std::uint32_t primary_seed =
      state.transactional && parity != 0u ? other_seed : selected_seed;
  const std::uint32_t alternate_seed =
      state.transactional && parity != 0u ? selected_seed : other_seed;
  const rund::AccelCheck primary =
      ops->seed_pipeline_generation(state.prepared, primary_seed);
  const rund::AccelCheck alternate =
      primary.ok && state.transactional
          ? ops->seed_pipeline_generation(state.alternate_prepared,
                                          alternate_seed)
          : primary;
  if (!alternate.ok) {
    if (location != nullptr) {
      location->native_reason_key = alternate.reason;
    }
    return Status::fail(
        project_reason(alternate.reason, Reason::PipelineInvalid));
  }
  state.native_generation = generation;
  state.native_parity = parity;
  return Status::success();
}

bool rebase_failed_pipeline_generation(PipelineState &state,
                                       const bool submitted,
                                       const Reason failure) noexcept {
  if (!submitted || state.device->backend == Backend::Cpu ||
      state.active_step_count == 0u || failure == Reason::DeviceLost) {
    return true;
  }
  const DeviceOps *const ops = state.device->ops;
  const node::accel::detail::PreparedKernelPipeline &selected =
      state.transactional && state.attempt.parity != 0u
          ? state.alternate_prepared
          : state.prepared;
  const std::uint32_t current =
      static_cast<std::uint32_t>(state.attempt.generation);
  const std::uint32_t seed =
      state.transactional ? current - std::uint32_t{1u} : current;
  const bool rebased = ops != nullptr &&
                       ops->seed_pipeline_generation != nullptr &&
                       ops->seed_pipeline_generation(selected, seed).ok;
  if (rebased) {
    state.native_generation = state.attempt.generation;
    state.native_parity = state.attempt.parity;
  }
  state.control_poisoned = !rebased;
  return rebased;
}

} // namespace rund::compute::detail
