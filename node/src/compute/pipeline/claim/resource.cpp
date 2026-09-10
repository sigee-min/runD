#include "../claim.hpp"

#include <mutex>

namespace rund::compute::detail {

Status
validate_pipeline_resource_device(const PipelineState &state,
                                  const PipelineResource &resource) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  return resource.buffer == nullptr || resource.buffer->device != state.device
             ? Status::fail(Reason::BindingDeviceMismatch)
             : Status::success();
}

Status validate_pipeline_resources(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  std::lock_guard lock{state.device->claims->gate};
  for (const PipelineResource &resource : state.resources) {
    const Status device = validate_pipeline_resource_device(state, resource);
    if (!device) {
      return device;
    }
    if (resource.buffer->poisoned) {
      return Status::fail(Reason::BufferPoisoned);
    }
  }
  return Status::success();
}

Status acquire_pipeline_claims(PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const std::span<const BufferClaim> claims =
      state.transactional && state.attempt.parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  return acquire_claims(*state.device, claims);
}

} // namespace rund::compute::detail
