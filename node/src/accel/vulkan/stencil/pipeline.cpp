#include "local.hpp"
#include <kernel/program/compute/stencil/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoStencilPlan(const rund::kernel::StencilDesc &desc,
                  const rund::kernel::ComputeDomain domain,
                  const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::StencilHash hash = rund::kernel::HashStencil(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::StencilElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireStencilPipeline(VulkanAdapter &adapter,
                       const rund::kernel::StencilDesc &desc,
                       const rund::kernel::ComputeDomain domain) {
  const rund::kernel::ComputePlan pseudo =
      PseudoStencilPlan(desc, domain, rund::kernel::ComputeApi::Vulkan);
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = VulkanStencilSource(desc.op, desc.element, domain);
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter, kStencilDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif
} // namespace rund::node::accel::detail
