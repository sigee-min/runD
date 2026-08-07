#include "../kernel/artifact.hpp"
#include "local.hpp"
#include <kernel/program/compute/scatter/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoScatterPlan(const rund::kernel::ScatterDesc &desc,
                  const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::ScatterHash hash = rund::kernel::HashScatter(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::ScatterElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireScatterPipeline(VulkanAdapter &adapter,
                       const rund::kernel::ScatterDesc &desc) {
  const rund::kernel::ComputePlan pseudo =
      PseudoScatterPlan(desc, rund::kernel::ComputeApi::Vulkan);
  std::string source = VulkanScatterSource(desc.element);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kScatterDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
