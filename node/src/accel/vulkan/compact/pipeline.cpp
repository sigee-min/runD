#include "../kernel/artifact.hpp"
#include "local.hpp"
#include <kernel/program/compute/compact/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoCompactPlan(const rund::kernel::CompactDesc &desc,
                  const rund::kernel::ComputeApi api,
                  const CompactStage stage) noexcept {
  const rund::kernel::CompactHash hash = rund::kernel::HashCompact(desc);
  const std::uint64_t salt = static_cast<std::uint64_t>(stage) + 1u;
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x434f4d5041435400ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
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
      PseudoCompactPlan(desc, rund::kernel::ComputeApi::Vulkan, stage);
  std::string source = VulkanCompactSource(stage);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kCompactDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
