#include "local.hpp"
#include <kernel/program/compute/partition/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPartitionPlan(const rund::kernel::PartitionDesc &desc,
                    const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::PartitionHash hash = rund::kernel::HashPartition(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.value_bytes == sizeof(rund::kernel::u64)
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquirePartitionPipeline(VulkanAdapter &adapter,
                         const rund::kernel::PartitionDesc &desc,
                         const PartitionStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoPartitionPlan(desc, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text =
      VulkanPartitionSource(stage, desc.flag_bytes, desc.value_bytes);
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter,
                                         stage == PartitionStage::Classify
                                             ? kPartitionClassifyDescriptorCount
                                             : kPartitionScatterDescriptorCount,
                                         0u, pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
