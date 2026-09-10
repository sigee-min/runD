#include "../map.hpp"

#include "../../../../ops/status.hpp"
#include "../../sliding/internal.hpp"

#include <cstdint>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool gate_result(VulkanResidencySlidingGate &gate,
                 const std::uint64_t descriptor, bool &known_failure,
                 const char *&reason) noexcept {
  using namespace vulkan_sliding_detail;
  known_failure = false;
  reason = "accel_kernel_pipeline_invalid";
  if (gate.descriptor.mapped == nullptr ||
      gate.descriptor.bytes < sizeof(VulkanResidencySlidingPayload)) {
    return false;
  }
  gate.gpu_result_read_count.fetch_add(1u, std::memory_order_relaxed);
  const auto *const payload =
      static_cast<const VulkanResidencySlidingPayload *>(
          gate.descriptor.mapped);
  const std::uint64_t observed =
      static_cast<std::uint64_t>(
          payload->words[VulkanResidencySlidingObservedGenerationWord]) |
      (static_cast<std::uint64_t>(
           payload->words[VulkanResidencySlidingObservedGenerationWord + 1u])
       << 32u);
  if (payload->words[VulkanResidencySlidingAcceptedWord] == 1u &&
      payload->words[VulkanResidencySlidingReasonWord] == 0u &&
      observed == descriptor) {
    return true;
  }
  const std::uint32_t code = payload->words[VulkanResidencySlidingReasonWord];
  // 0xffffffff is the private authentication marker.  Every other nonzero
  // generated result must be a canonical runD Reason, never an ad-hoc 1/2.
  if (code != 0u && code != 0xffffffffu && CanonicalReasonStatus(code)) {
    known_failure = true;
    reason = CanonicalReasonText(code);
  }
  return false;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
