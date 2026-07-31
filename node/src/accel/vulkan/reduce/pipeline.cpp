#include "local.hpp"
#include <kernel/program/compute/reduce/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoReducePlan(const rund::kernel::ReduceDesc &desc,
                 const rund::kernel::ComputeDomain domain,
                 const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::ReduceHash hash = rund::kernel::HashReduce(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::ReduceElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireReducePipeline(VulkanAdapter &adapter,
                      const rund::kernel::ReduceDesc &desc,
                      const rund::kernel::ComputeDomain domain) {
  const rund::kernel::ComputePlan pseudo =
      PseudoReducePlan(desc, domain, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text =
      VulkanReduceSource(desc.op, desc.element, desc.block_size, domain);
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter, kReduceDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
