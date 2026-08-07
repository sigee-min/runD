#include "../../domain.hpp"
#include "../kernel/artifact.hpp"
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
  const bool wide = desc.element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      desc.op != rund::kernel::StencilOp::Sum && IsSignedDomain(domain);
  const rund::kernel::ComputeDomain executable_domain =
      signed_extrema ? (wide ? rund::kernel::ComputeDomain::I64
                             : rund::kernel::ComputeDomain::I32)
                     : (wide ? rund::kernel::ComputeDomain::U64
                             : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::StencilElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      // The complete source has only signed-extrema and lane-width branches.
      // Normalize domains that compile to identical text so exact full-source
      // reuse remains visible in the complete ArtifactKey tuple.
      .domain = executable_domain,
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
  std::string source = VulkanStencilSource(desc.op, desc.element, domain);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kStencilDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif
} // namespace rund::node::accel::detail
