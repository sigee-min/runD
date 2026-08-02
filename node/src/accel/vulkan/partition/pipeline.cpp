#include "local.hpp"
#include "../kernel/source_recipe.hpp"
#include <kernel/program/compute/partition/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPartitionPlan(const rund::kernel::PartitionDesc &desc,
                    const rund::kernel::ComputeApi api,
                    const PartitionStage stage) noexcept {
  // Element count is runtime data. Normalize it out of the shader identity.
  const rund::kernel::PartitionHash hash = rund::kernel::HashPartition(
      rund::kernel::PartitionDesc{.flag_bytes = desc.flag_bytes,
                                  .value_bytes = desc.value_bytes});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage) + 1u;
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5041525449540000ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
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
      PseudoPartitionPlan(desc, rund::kernel::ComputeApi::Vulkan, stage);
  std::string source =
      VulkanPartitionSource(stage, desc.flag_bytes, desc.value_bytes);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter,
                                         stage == PartitionStage::Classify
                                             ? kPartitionClassifyDescriptorCount
                                             : kPartitionScatterDescriptorCount,
                                         0u, pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
