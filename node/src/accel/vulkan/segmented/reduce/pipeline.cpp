#include "model.hpp"

#include <kernel/program/compute/segmented/reduce/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::SegmentedReduceDesc &desc,
           const rund::kernel::ComputeDomain domain) noexcept {
  const rund::kernel::SegmentedReduceHash hash =
      rund::kernel::HashSegmentedReduce(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = desc.element == rund::kernel::ReduceElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = domain,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] VulkanCollectivePipeline *AcquireSource(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::ComputeDomain domain, const std::string &source) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = source;
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter,
                                         kVulkanSegmentedReduceBindings, 0u,
                                         PseudoPlan(desc, domain), artifact);
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanSegmentedIndex(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::ComputeDomain domain, const char *const source) {
  return source == nullptr
             ? nullptr
             : AcquireSource(adapter, desc, domain, std::string{source});
}

VulkanCollectivePipeline *
AcquireVulkanSegmentedReduce(VulkanAdapter &adapter,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             const rund::kernel::ComputeDomain domain) {
  return AcquireSource(adapter, desc, domain,
                       VulkanSegmentedReduceSource(plan, domain));
}

#endif

} // namespace rund::node::accel::detail
