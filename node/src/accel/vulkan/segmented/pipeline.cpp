#include "local.hpp"
#include "../../domain.hpp"
#include "../kernel/source_recipe.hpp"
#include <kernel/program/compute/segmented/scan/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoSegmentedScanPlan(const rund::kernel::SegmentedScanDesc &desc,
                        const rund::kernel::ComputeDomain domain,
                        const rund::kernel::ComputeApi api,
                        const VulkanSegmentedScanStage stage) noexcept {
  const rund::kernel::SegmentedScanHash hash =
      rund::kernel::HashSegmentedScan(rund::kernel::SegmentedScanDesc{
          .op = desc.op, .element = desc.element});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage) + 1u;
  const bool wide = desc.element == rund::kernel::SegmentedScanElement::U64;
  const rund::kernel::ComputeDomain executable_domain =
      IsSignedDomain(domain)
          ? (wide ? rund::kernel::ComputeDomain::I64
                  : rund::kernel::ComputeDomain::I32)
          : (wide ? rund::kernel::ComputeDomain::U64
                  : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5345475343410000ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = api,
      .scalar = desc.element == rund::kernel::SegmentedScanElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *AcquireSegmentedScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedScanStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoSegmentedScanPlan(desc, domain, rund::kernel::ComputeApi::Vulkan,
                              stage);
  std::string source = VulkanSegmentedScanSource(desc.element, domain, stage);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  const std::uint32_t push_bytes =
      stage == VulkanSegmentedScanStage::Prefix ? 0u
                                                : kSegmentedScanPushBytes;
  return AcquireVulkanCollectivePipeline(adapter, kSegmentedScanDescriptorCount,
                                         push_bytes, pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
