#include "local.hpp"
#include <kernel/program/compute/segmented/scan/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoSegmentedScanPlan(const rund::kernel::SegmentedScanDesc &desc,
                        const rund::kernel::ComputeDomain domain,
                        const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::SegmentedScanHash hash =
      rund::kernel::HashSegmentedScan(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::SegmentedScanElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *AcquireSegmentedScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::ComputeDomain domain, const std::string_view phase) {
  const rund::kernel::ComputePlan pseudo =
      PseudoSegmentedScanPlan(desc, domain, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = VulkanSegmentedScanSource(desc.element, domain, phase);
  artifact.ok = true;
  artifact.reason = "ok";
  const std::uint32_t push_bytes =
      phase == "prefix" ? 0u : kSegmentedScanPushBytes;
  return AcquireVulkanCollectivePipeline(adapter, kSegmentedScanDescriptorCount,
                                         push_bytes, pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
