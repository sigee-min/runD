#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck ValidateVulkanMapPrepare(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings) {
  if (!ComputePlanHeaderValid(plan, rund::kernel::ComputeApi::Vulkan)) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!FrozenCapsAdmitPlan(adapter.caps, plan)) {
    SetVulkanLastError(adapter, "compute_backend_mismatch");
    return rund::AccelCheck{false, "compute_backend_mismatch"};
  }
  const auto admission = rund::kernel::compute_lowering_detail::AdmitRetained(
      plan, artifact, nullptr);
  if (!admission.ok) {
    const char *const reason =
        artifact.kind == rund::kernel::LoweringArtifactKind::VulkanSource
            ? "compute_artifact_mismatch"
            : "compute_artifact_non_executable";
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  if (admission.parse_count != 0u || admission.emission_count != 0u) {
    SetVulkanLastError(adapter, "compute_artifact_mismatch");
    return rund::AccelCheck{false, "compute_artifact_mismatch"};
  }
  const rund::kernel::BindingValidation binding =
      BindingPlanCheck(plan, bindings, artifact.metadata,
                       PlanBindingInputMode::MapResidentViews);
  if (!binding.ok) {
    SetVulkanLastError(adapter, binding.reason);
    return rund::AccelCheck{false, binding.reason};
  }
  if (!RuntimeWindowsMatchPlan(plan, windows, window_count, bindings) ||
      !bindings.has_resident_output()) {
    SetVulkanLastError(adapter, "compute_dispatch_count_mismatch");
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  const auto aligned_windows =
      [&](const rund::kernel::ResidentBindingRange &refs) {
        const std::uint64_t alignment = adapter.storage_align;
        if (alignment == 0u) {
          return false;
        }
        for (std::uint64_t index = 0u; index < refs.count; ++index) {
          const rund::kernel::ResidentBufferRef *const ref = refs.ref(index);
          if (ref == nullptr || ref->stride_bytes == 0u) {
            return false;
          }
          const std::uint64_t bias = ref->offset_bytes % alignment;
          for (std::uint64_t window = 0u; window < window_count; ++window) {
            if (windows[window].begin_sequence >
                    std::numeric_limits<std::uint64_t>::max() /
                        ref->stride_bytes ||
                ref->offset_bytes >
                    std::numeric_limits<std::uint64_t>::max() -
                        windows[window].begin_sequence * ref->stride_bytes ||
                (ref->offset_bytes +
                 windows[window].begin_sequence * ref->stride_bytes) %
                        alignment !=
                    bias) {
              return false;
            }
          }
        }
        return true;
      };
  if (!aligned_windows(bindings.resident_inputs) ||
      !aligned_windows(bindings.resident_outputs)) {
    SetVulkanLastError(adapter, "compute_resident_stride_invalid");
    return rund::AccelCheck{false, "compute_resident_stride_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
