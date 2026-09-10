#include "readiness.hpp"

#include "../../backend.hpp"
#include "../../device/residency/execution/model.hpp"
#include "../state.hpp"

#include <array>
#include <mutex>

namespace rund::compute::detail {

Status
ready_virtual_residency_window(const VirtualPipelineState &state) noexcept {
  if (state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.alternate_pipeline->device != state.pipeline->device ||
      state.pipeline->device->ops == nullptr ||
      state.pipeline->device->ops->residency.residency_window_capability ==
          nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  std::array<node::accel::detail::PreparedKernelPipeline,
             residency::execution::BankCapacity>
      native{};
  const std::array<std::shared_ptr<PipelineState>,
                   residency::execution::BankCapacity>
      pipelines{state.pipeline, state.alternate_pipeline};
  for (std::size_t bank = 0u; bank < pipelines.size(); ++bank) {
    std::lock_guard pipeline_lock{pipelines[bank]->gate};
    std::lock_guard publication_lock{pipelines[bank]->publication->gate};
    native[bank] = pipelines[bank]->prepared;
  }
  bool ready = false;
  const Status capability =
      state.pipeline->device->ops->residency.residency_window_capability(
          *state.pipeline->device,
          std::span<const node::accel::detail::PreparedKernelPipeline>{native},
          ready);
  if (!capability) {
    return capability;
  }
  return ready ? Status::success() : Status::fail(Reason::BackendUnsupported);
}

} // namespace rund::compute::detail
