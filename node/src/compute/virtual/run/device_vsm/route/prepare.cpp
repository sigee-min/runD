#include "../route.hpp"
#include "../../../../backend.hpp"

#include "../internal.hpp"
#include "../operations.hpp"


namespace rund::compute::detail {

VirtualDeviceVsmPostStage prepare_virtual_device_vsm_route(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    const VirtualDeviceVsmCandidate candidate) noexcept {
  VirtualDeviceVsmPostStage result{};
  result.proof = candidate.proof;
  if (!candidate.device_vsm() || state.pipeline == nullptr ||
      state.pipeline->device == nullptr) {
    return result;
  }
  const DeviceOps *const ops = state.pipeline->device->ops;
  if (ops == nullptr ||
      ops->virtual_execution.prepare_virtual_device_vsm_product == nullptr) {
    result.status = Status::fail(Reason::BackendUnsupported);
    return result;
  }
  if (!candidate.proof.valid()) {
    result.status = Status::fail(Reason::BackendUnsupported);
    return result;
  }
  result.prepared = {};
  result.status = ops->virtual_execution.prepare_virtual_device_vsm_product(
      state, run, candidate.proof, result.prepared);
  if (result.status && result.prepared) {
    if (result.prepared.proof != candidate.proof) {
      result.status = Status::fail(Reason::PipelineInvalid);
    } else {
      result.phase = VirtualDeviceVsmPostPhase::Ready;
    }
  } else if (result.status) {
    result.status = Status::fail(Reason::PipelineInvalid);
  }
  return result;
}

Status
rearm_virtual_device_vsm_route(VirtualDeviceVsmPostStage &post) noexcept {
  if (!post.ready()) {
    return post.status ? Status::fail(Reason::PipelineInvalid) : post.status;
  }
  return device_vsm_product_detail::rearm_prepared(post.prepared);
}

} // namespace rund::compute::detail
