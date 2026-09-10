#include "window.hpp"

#include "../validation.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck QueryPreparedKernelPipelineDeviceVsmWindowCapability(
    const PreparedKernelPipeline &primary, const PreparedKernelPipeline &peer,
    const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain,
    const rund::kernel::WindowElement element) noexcept {
  if (!primary.ok || !peer.ok || primary.owner == nullptr ||
      peer.owner == nullptr || scalar != rund::kernel::ComputeScalar::Lane32 ||
      (domain != rund::kernel::ComputeDomain::I32 &&
       domain != rund::kernel::ComputeDomain::U32) ||
      element != rund::kernel::WindowElement::U32) {
    return {false, "device_vsm_window_capability_invalid"};
  }
  const auto *const left =
      static_cast<const prepared::PipelineState *>(primary.owner.get());
  const auto *const right =
      static_cast<const prepared::PipelineState *>(peer.owner.get());
  device_vsm_window_projection::WindowAuthority first{};
  const char *reason = nullptr;
  if (left == nullptr || right == nullptr ||
      !device_vsm_window_projection::exact_window_pipeline(*left, first,
                                                           reason)) {
    return {false,
            reason == nullptr ? "device_vsm_window_pipeline_invalid" : reason};
  }
  if (!device_vsm_window_projection::same_window_pipeline(*right, first)) {
    return {false, "device_vsm_window_state_mismatch"};
  }
  if (!device_vsm_window_ring_fusion_valid(first.fusion) ||
      (domain == rund::kernel::ComputeDomain::I32 && first.fusion.active()) ||
      first.step == nullptr || first.step->planned == nullptr ||
      first.active == nullptr || first.active->plan.element != element ||
      first.active->plan.domain != domain ||
      first.active->plan.boundary != rund::kernel::WindowBoundary::Clamp ||
      first.step->planned->plan.scalar != scalar) {
    return {false, "device_vsm_window_fusion_invalid"};
  }
  return {true, "ok"};
}

} // namespace rund::node::accel::detail
