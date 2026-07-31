#include "local.hpp"
#include <kernel/program/compute/compact/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoCompactPlan(const rund::kernel::CompactDesc &desc,
                  const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::CompactHash hash = rund::kernel::HashCompact(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireCompactPipeline(VulkanAdapter &adapter,
                       const rund::kernel::CompactDesc &desc,
                       const CompactStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoCompactPlan(desc, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = VulkanCompactSource(stage);
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter, kCompactDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
