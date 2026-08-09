#include "../kernel/artifact.hpp"
#include "local.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoRangePlan(const RangeExec &execution) noexcept {
  const RangeIdentity identity = execution.source_identity();
  const bool wide = execution.wide_elements();
  const bool signed_values = execution.signed_values();
  const rund::kernel::ComputeDomain executable_domain =
      signed_values ? (wide ? rund::kernel::ComputeDomain::I64
                            : rund::kernel::ComputeDomain::I32)
                    : (wide ? rund::kernel::ComputeDomain::U64
                            : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = identity.hi,
      .op_hash_lo = identity.lo,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = wide ? rund::kernel::ComputeScalar::Lane64
                     : rund::kernel::ComputeScalar::Lane32,
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

bool VulkanRangePipelineMatches(const VulkanAdapter &adapter,
                                const VulkanCollectivePipeline *const pipeline,
                                const RangeExec &execution) noexcept {
  const std::uint32_t descriptor_count = execution.descriptor_count();
  if (pipeline == nullptr || adapter.device == VK_NULL_HANDLE ||
      pipeline->device != adapter.device ||
      pipeline->descriptor_count != descriptor_count ||
      pipeline->push_bytes != 0u ||
      pipeline->specialization != VulkanSpecialization{} ||
      pipeline->pipeline == VK_NULL_HANDLE ||
      pipeline->pipeline_layout == VK_NULL_HANDLE ||
      pipeline->descriptor_set_layout == VK_NULL_HANDLE ||
      pipeline->source.empty()) {
    return false;
  }
  const rund::kernel::ComputePlan pseudo = PseudoRangePlan(execution);
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
         VulkanRangeSourceMatches(execution, pipeline->source,
                                  pipeline->source_hash);
}

VulkanCollectivePipeline *
AcquireVulkanRangePipeline(VulkanAdapter &adapter, const RangeExec &execution) {
  if (!execution.vulkan_dispatch_fits(adapter.max_dispatch_groups)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return nullptr;
  }
  const rund::kernel::ComputePlan pseudo = PseudoRangePlan(execution);
  std::string source = VulkanRangeSource(execution);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, execution.descriptor_count(),
                                         0u, pseudo, artifact);
}
#endif

RangeCaps VulkanRangeCaps(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || adapter->physical_device == VK_NULL_HANDLE) {
    return RangeCaps::unavailable();
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
  for (const rund::kernel::u32 width : kRangeWidths) {
    if (width <= maximum_width) {
      widths |= width == 64u    ? kRangeWidth64Bit
                : width == 128u ? kRangeWidth128Bit
                                : kRangeWidth256Bit;
    }
  }
  const std::optional<RangeCaps> capabilities = RangeCaps::gpu(
      RangeSource::Vulkan, widths, maximum_width, kRangeSharedReserve,
      properties.limits.maxComputeSharedMemorySize, maximum_groups,
      std::numeric_limits<rund::kernel::u32>::max(), adapter->storage_limit,
      RangeSupportBit(RangeSupport::Direct) |
          RangeSupportBit(RangeSupport::SharedHalo) |
          RangeSupportBit(RangeSupport::PrefixDifference) |
          RangeSupportBit(RangeSupport::BlockPrefixSuffix));
  return capabilities.value_or(RangeCaps::unavailable());
#else
  (void)pick;
  return RangeCaps::unavailable();
#endif
}
} // namespace rund::node::accel::detail
