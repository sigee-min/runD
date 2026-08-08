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
  rund::kernel::StencilDesc source_identity = desc;
  source_identity.element_count = 0u;
  source_identity.radius = 0u;
  const rund::kernel::StencilHash hash =
      rund::kernel::HashStencil(source_identity);
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
      // Count and radius are runtime params. The complete source has only
      // operation, signed-extrema, and lane-width branches, so its pseudo
      // identity normalizes every non-source field before cache admission.
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
  if (!StencilVulkanDispatchFits(desc.element_count,
                                 adapter.max_dispatch_groups)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return nullptr;
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(adapter.physical_device, &properties);
  if (properties.limits.maxComputeWorkGroupInvocations <
          kStencilPhysicalGroupWidth ||
      properties.limits.maxComputeWorkGroupSize[0] <
          kStencilPhysicalGroupWidth ||
      properties.limits.maxComputeSharedMemorySize < kStencilSharedBytesMax) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return nullptr;
  }
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
