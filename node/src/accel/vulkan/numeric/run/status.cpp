#include "../resource.hpp"

#include "../../collective/finish.hpp"
#include "../../kernel/ops/status.hpp"
#include "../../runtime/timestamp.hpp"

#include <cstdint>
#include <limits>
#include <memory>

namespace rund::node::accel::detail {

rund::AccelCheck
DescribeVulkanNumericPipelineStatus(const std::shared_ptr<void> &prepared,
                                    VulkanPipelineStatusSource &source) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state = static_cast<VulkanNumericPrepared *>(prepared.get());
  source = {};
  if (state == nullptr || !state->pipeline_private) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (state->status_kind == VulkanNumericStatusKind::None) {
    return rund::AccelCheck{true, "ok"};
  }
  if (state->status == nullptr || state->status->buffer == VK_NULL_HANDLE ||
      state->status_count == 0u ||
      state->status_count > std::numeric_limits<std::uint32_t>::max() ||
      state->status_binding.buffer != state->status ||
      state->status_count >
          state->status_binding.range / sizeof(std::uint32_t)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  source.raw = state->status;
  source.offset = state->status_binding.offset;
  source.count = static_cast<std::uint32_t>(state->status_count);
  source.rule = VulkanPipelineStatusRule::Exact;
  source.success = 0u;
  const auto set = [&](const std::uint32_t index, const std::uint32_t raw,
                       const rund::compute::Reason reason) {
    source.raw_values[index] = raw;
    source.reasons[index] = static_cast<std::uint32_t>(reason);
  };
  if (state->status_kind == VulkanNumericStatusKind::Spectrum) {
    source.mapping_count = 2u;
    set(0u, 1u, rund::compute::Reason::SpectrumNonConvergence);
    set(1u, 2u, rund::compute::Reason::SpectrumScalingInvalid);
  } else {
    source.mapping_count = 4u;
    const bool factor = state->status_kind == VulkanNumericStatusKind::Factor;
    set(0u, 1u,
        factor ? rund::compute::Reason::FactorSingular
               : rund::compute::Reason::SolveSingular);
    set(1u, 2u,
        factor ? rund::compute::Reason::FactorNotPositiveDefinite
               : rund::compute::Reason::SolveNotPositiveDefinite);
    set(2u, 3u,
        factor ? rund::compute::Reason::FactorPivotUnderflow
               : rund::compute::Reason::SolvePivotUnderflow);
    set(3u, 4u,
        factor ? rund::compute::Reason::FactorScalingInvalid
               : rund::compute::Reason::SolveScalingInvalid);
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)prepared;
  (void)source;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck FinishVulkanNumeric(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &prepared) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const state = static_cast<VulkanNumericPrepared *>(prepared.get());
  if (state == nullptr || state->adapter != &adapter) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  rund::AccelCheck check =
      state->status == nullptr
          ? rund::AccelCheck{true, "ok"}
          : StatusCheck(adapter, state->status_readback, state->status_count);
  if (!check.ok) {
    return check;
  }
  const rund::AccelCheck accepted =
      AcceptVulkanDispatches(adapter, state->dispatches);
  if (!accepted.ok) {
    return accepted;
  }
  return check;
#else
  (void)adapter;
  (void)prepared;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
