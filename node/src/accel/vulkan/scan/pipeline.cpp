#include "../collective/pipeline.hpp"
#include "local.hpp"
#include "pipeline.hpp"
#include "source.hpp"
#include <kernel/program/compute/scan/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::ScanDesc &desc,
           const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::ScanHash hash = rund::kernel::HashScan(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::ScanElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ComputeDomain domain, const VulkanScanStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoPlan(desc, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  const bool inclusive = desc.op == rund::kernel::ScanOp::InclusiveSum;
  artifact.source_text =
      VulkanScanSource(desc.element, domain, stage, inclusive);
  artifact.ok = true;
  artifact.reason = "ok";
  const std::uint32_t push_bytes =
      stage == VulkanScanStage::Prefix ? 0u : kScanPushBytes;
  return AcquireVulkanCollectivePipeline(adapter, kScanDescriptorCount,
                                         push_bytes, pseudo, artifact);
}

#endif

} // namespace rund::node::accel::detail
