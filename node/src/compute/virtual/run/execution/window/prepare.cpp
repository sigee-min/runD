#include "../../execution.hpp"
#include "../../../../../accel/kernel/residency/window.hpp"

#include "../../cache.hpp"

#include "../../../../backend.hpp"
#include "../../../../pipeline/local.hpp"
#include "../../../../pipeline/state.hpp"

#include <array>
#include <mutex>
#include <span>

namespace rund::compute::detail {

Status prepare_virtual_execution_window(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    VirtualExecutionWindowPrepared &prepared) noexcept {
  prepared = {};
  const residency::execution::SealResult sealed =
      seal_virtual_execution(state, run);
  if (!sealed || sealed.plan.epoch_count() < 2u || state.pipeline == nullptr ||
      state.alternate_pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.pipeline->device != state.alternate_pipeline->device ||
      state.pipeline->device->ops == nullptr ||
      state.pipeline->device->ops->residency.residency_window_capability ==
          nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  std::array<node::accel::detail::PreparedKernelPipeline,
             node::accel::detail::ResidencyWindowCapacity>
      native{};
  std::size_t native_count = 0u;
  const std::array<std::shared_ptr<PipelineState>,
                   residency::execution::BankCapacity>
      pipelines{state.pipeline, state.alternate_pipeline};
  for (std::size_t bank = 0u; bank < pipelines.size(); ++bank) {
    std::lock_guard pipeline_lock{pipelines[bank]->gate};
    std::lock_guard publication_lock{pipelines[bank]->publication->gate};
    native[native_count++] = pipelines[bank]->prepared;
    if (pipelines[bank]->transactional) {
      native[native_count++] = pipelines[bank]->alternate_prepared;
    }
  }
  bool ready = false;
  const Status capability =
      state.pipeline->device->ops->residency.residency_window_capability(
          *state.pipeline->device,
          std::span<const node::accel::detail::PreparedKernelPipeline>{
              native.data(), native_count},
          ready);
  if (!capability || !ready) {
    return capability ? Status::fail(Reason::BackendUnsupported) : capability;
  }
  // A whole window is queued before recurrent Host service. A physical upload
  // or download submitted to that same native queue behind a future ready
  // wait would form a cycle (GPU waits ready; Host waits transfer fence).
  // The mutation-free backend readiness query above observes a pending
  // view-unavailable fault without consuming it; only after it succeeds do we
  // authenticate both exact Buffer owners. Thus rejected windows leave the
  // fault for the rolling physical fallback that must consume it.
  for (const std::shared_ptr<PipelineState> &pipeline : pipelines) {
    if (pipeline == nullptr || !residency_input_view(*pipeline, run) ||
        !residency_output_view(*pipeline)) {
      return Status::fail(Reason::BackendUnsupported);
    }
  }
  prepared.plan = sealed.plan;
  prepared.pipelines = pipelines;
  return Status::success();
}

} // namespace rund::compute::detail
