#include "model.hpp"

#include <limits>

namespace rund::compute::detail {

Status validate_pipeline_transfer_ready(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr ||
      state.publication == nullptr || state.preparing ||
      state.phase != PipelinePhase::Ready) {
    return Status::fail(state.phase == PipelinePhase::Poisoned
                            ? Reason::PipelinePoisoned
                            : (state.phase == PipelinePhase::Running
                                   ? Reason::PipelineBusy
                                   : Reason::PipelineInvalid));
  }
  return state.publication->device_lost ? Status::fail(Reason::DeviceLost)
                                        : Status::success();
}

bool add_pipeline_transfer_bytes(const std::size_t bytes,
                                 std::uint64_t &total) noexcept {
  if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
    return false;
  }
  total += bytes;
  return true;
}

} // namespace rund::compute::detail
