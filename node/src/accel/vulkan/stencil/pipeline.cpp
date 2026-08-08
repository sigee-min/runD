#include "../../domain.hpp"
#include "../kernel/artifact.hpp"
#include "local.hpp"
#include <kernel/program/compute/stencil/identity.hpp>
#include <kernel/program/compute/stencil/plan.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoStencilPlan(const rund::kernel::StencilDesc &desc,
                  const rund::kernel::ComputeDomain domain,
                  const rund::kernel::ComputeApi api,
                  const RangeAggregatePlan &range) noexcept {
  rund::kernel::StencilDesc source_identity = desc;
  source_identity.element_count = 0u;
  source_identity.radius = 0u;
  const rund::kernel::StencilHash hash =
      rund::kernel::HashStencil(source_identity);
  const RangeAggregateIdentity physical = range.source_identity();
  const bool wide = desc.element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      desc.op != rund::kernel::StencilOp::Sum && IsSignedDomain(domain);
  const rund::kernel::ComputeDomain executable_domain =
      signed_extrema ? (wide ? rund::kernel::ComputeDomain::I64
                             : rund::kernel::ComputeDomain::I32)
                     : (wide ? rund::kernel::ComputeDomain::U64
                             : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi =
          hash.hi ^ (physical.hi * UINT64_C(0x9e3779b97f4a7c15)) ^ physical.lo,
      .op_hash_lo =
          hash.lo ^ physical.lo ^ (physical.hi * UINT64_C(0x94d049bb133111eb)),
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

bool VulkanStencilPipelineMatches(
    const VulkanAdapter &adapter,
    const VulkanCollectivePipeline *const pipeline,
    const rund::kernel::StencilDesc &desc,
    const rund::kernel::ComputeDomain domain,
    const RangeAggregatePlan &range) noexcept {
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
  const std::uint32_t descriptor_count = StencilRangeDescriptorCount(range);
  if (pipeline == nullptr || adapter.device == VK_NULL_HANDLE ||
      pipeline->device != adapter.device || !shape.valid() ||
      !StencilRangeAggregatePlanMatches(rund::kernel::PlanStencil(desc), domain,
                                        range) ||
      pipeline->descriptor_count != descriptor_count ||
      pipeline->push_bytes != 0u ||
      pipeline->specialization != VulkanSpecialization{} ||
      pipeline->pipeline == VK_NULL_HANDLE ||
      pipeline->pipeline_layout == VK_NULL_HANDLE ||
      pipeline->descriptor_set_layout == VK_NULL_HANDLE ||
      pipeline->source.empty()) {
    return false;
  }
  const rund::kernel::ComputePlan pseudo =
      PseudoStencilPlan(desc, domain, rund::kernel::ComputeApi::Vulkan, range);
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
         VulkanStencilSourceMatches(desc.op, desc.element, domain, shape, range,
                                    pipeline->source, pipeline->source_hash);
}

VulkanCollectivePipeline *AcquireStencilPipeline(
    VulkanAdapter &adapter, const rund::kernel::StencilDesc &desc,
    const rund::kernel::ComputeDomain domain, const RangeAggregatePlan &range) {
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
  const std::uint32_t descriptor_count = StencilRangeDescriptorCount(range);
  if (!shape.valid() ||
      !StencilVulkanDispatchFits(desc.element_count,
                                 adapter.max_dispatch_groups, shape)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return nullptr;
  }
  if (!StencilRangeAggregatePlanMatches(rund::kernel::PlanStencil(desc), domain,
                                        range)) {
    SetVulkanLastError(adapter, "accel_vulkan_pipeline_unavailable");
    return nullptr;
  }
  const rund::kernel::ComputePlan pseudo =
      PseudoStencilPlan(desc, domain, rund::kernel::ComputeApi::Vulkan, range);
  std::string source =
      VulkanStencilSource(desc.op, desc.element, domain, shape, range);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, descriptor_count, 0u, pseudo,
                                         artifact);
}
#endif

RangeAggregateCapabilities
VulkanRangeAggregateCapabilities(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || adapter->physical_device == VK_NULL_HANDLE) {
    return RangeAggregateCapabilities::unavailable();
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(adapter->physical_device, &properties);
  const rund::kernel::u32 maximum_width =
      std::min(properties.limits.maxComputeWorkGroupInvocations,
               properties.limits.maxComputeWorkGroupSize[0]);
  const rund::kernel::u64 maximum_groups = std::min<rund::kernel::u64>(
      adapter->max_dispatch_groups,
      properties.limits.maxComputeWorkGroupCount[0]);
  std::uint8_t widths = 0u;
  for (const rund::kernel::u32 width : kRangeAggregateWorkgroupWidths) {
    if (width <= maximum_width) {
      widths |= width == 64u    ? kRangeAggregateWidth64Bit
                : width == 128u ? kRangeAggregateWidth128Bit
                                : kRangeAggregateWidth256Bit;
    }
  }
  const std::optional<RangeAggregateCapabilities> capabilities =
      RangeAggregateCapabilities::gpu(
          RangeAggregateSourceVariant::Vulkan, widths, maximum_width,
          kRangeAggregateSharedMemoryReserve,
          properties.limits.maxComputeSharedMemorySize, maximum_groups,
          RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
              RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo) |
              RangeAggregateSupportBit(
                  RangeAggregateSupport::PrefixDifference) |
              RangeAggregateSupportBit(
                  RangeAggregateSupport::BlockPrefixSuffix));
  return capabilities.value_or(RangeAggregateCapabilities::unavailable());
#else
  (void)pick;
  return RangeAggregateCapabilities::unavailable();
#endif
}
} // namespace rund::node::accel::detail
