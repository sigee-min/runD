#include "../../domain.hpp"
#include "../kernel/artifact.hpp"
#include "local.hpp"
#include <kernel/program/compute/stencil/identity.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoStencilPlan(const rund::kernel::StencilDesc &desc,
                  const rund::kernel::ComputeDomain domain,
                  const rund::kernel::ComputeApi api,
                  const StencilGpuShape shape) noexcept {
  rund::kernel::StencilDesc source_identity = desc;
  source_identity.element_count = 0u;
  source_identity.radius = 0u;
  const rund::kernel::StencilHash hash =
      rund::kernel::HashStencil(source_identity);
  const std::uint64_t physical = shape.identity();
  const bool wide = desc.element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      desc.op != rund::kernel::StencilOp::Sum && IsSignedDomain(domain);
  const rund::kernel::ComputeDomain executable_domain =
      signed_extrema ? (wide ? rund::kernel::ComputeDomain::I64
                             : rund::kernel::ComputeDomain::I32)
                     : (wide ? rund::kernel::ComputeDomain::U64
                             : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (physical * UINT64_C(0x9e3779b97f4a7c15)),
      .op_hash_lo = hash.lo ^ physical,
      .api = api,
      .scalar = desc.element == rund::kernel::StencilElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      // Count and radius remain runtime params, but their capability-derived
      // physical result is source identity.  HashStencil stays the semantic
      // authority; width/radius-cap are mixed only into this backend key.
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

StencilGpuShape SelectVulkanStencilGpuShape(
    const VulkanAdapter &adapter, const rund::kernel::u64 element_count,
    const rund::kernel::u64 radius,
    const rund::kernel::StencilElement element) noexcept {
  if (adapter.physical_device == VK_NULL_HANDLE ||
      element_count > std::numeric_limits<rund::kernel::u32>::max() ||
      (element != rund::kernel::StencilElement::U32 &&
       element != rund::kernel::StencilElement::U64)) {
    return {};
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(adapter.physical_device, &properties);
  const rund::kernel::u32 maximum_width =
      std::min(properties.limits.maxComputeWorkGroupInvocations,
               properties.limits.maxComputeWorkGroupSize[0]);
  const rund::kernel::u64 maximum_groups = std::min<rund::kernel::u64>(
      adapter.max_dispatch_groups,
      properties.limits.maxComputeWorkGroupCount[0]);
  return SelectStencilGpuShape(
      element_count, radius, StencilElementBytes(element),
      StencilGpuCapabilities{
          .maximum_workgroup_width = maximum_width,
          .shared_memory_occupancy_budget = kStencilSharedMemoryOccupancyBudget,
          .shared_memory_limit = properties.limits.maxComputeSharedMemorySize,
          .maximum_group_count = maximum_groups,
      });
}

bool VulkanStencilPipelineMatches(
    const VulkanAdapter &adapter,
    const VulkanCollectivePipeline *const pipeline,
    const rund::kernel::StencilDesc &desc,
    const rund::kernel::ComputeDomain domain,
    const StencilGpuShape shape) noexcept {
  if (pipeline == nullptr || adapter.device == VK_NULL_HANDLE ||
      pipeline->device != adapter.device || !shape.valid() ||
      shape != SelectVulkanStencilGpuShape(adapter, desc.element_count,
                                           desc.radius, desc.element) ||
      pipeline->descriptor_count != kStencilDescriptorCount ||
      pipeline->push_bytes != 0u ||
      pipeline->specialization != VulkanSpecialization{} ||
      pipeline->pipeline == VK_NULL_HANDLE ||
      pipeline->pipeline_layout == VK_NULL_HANDLE ||
      pipeline->descriptor_set_layout == VK_NULL_HANDLE ||
      pipeline->source.empty()) {
    return false;
  }
  const rund::kernel::ComputePlan pseudo =
      PseudoStencilPlan(desc, domain, rund::kernel::ComputeApi::Vulkan, shape);
  const rund::kernel::ArtifactKey expected_key{
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = pseudo.scalar,
      .domain = pseudo.domain,
      .variant = rund::kernel::LoweringArtifactVariant::Canonical,
      .fixed_format = pseudo.fixed_format,
      .op_hash_hi = pseudo.op_hash_hi,
      .op_hash_lo = pseudo.op_hash_lo,
      .canonical_ir_hash_hi = pseudo.op_hash_hi,
      .canonical_ir_hash_lo = pseudo.op_hash_lo,
  };
  return pipeline->key == expected_key &&
         VulkanStencilSourceMatches(desc.op, desc.element, domain, shape,
                                    pipeline->source, pipeline->source_hash);
}

VulkanCollectivePipeline *AcquireStencilPipeline(
    VulkanAdapter &adapter, const rund::kernel::StencilDesc &desc,
    const rund::kernel::ComputeDomain domain, const StencilGpuShape shape) {
  if (!shape.valid() ||
      !StencilVulkanDispatchFits(desc.element_count,
                                 adapter.max_dispatch_groups, shape)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return nullptr;
  }
  if (shape != SelectVulkanStencilGpuShape(adapter, desc.element_count,
                                           desc.radius, desc.element)) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return nullptr;
  }
  const rund::kernel::ComputePlan pseudo =
      PseudoStencilPlan(desc, domain, rund::kernel::ComputeApi::Vulkan, shape);
  std::string source =
      VulkanStencilSource(desc.op, desc.element, domain, shape);
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
