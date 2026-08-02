#include "model.hpp"
#include "../../../domain.hpp"
#include "../../kernel/source_recipe.hpp"

#include <kernel/program/compute/segmented/reduce/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::SegmentedReduceDesc &desc,
           const rund::kernel::ComputeDomain domain,
           const VulkanSegmentedReduceStage stage) noexcept {
  const bool reduce = stage == VulkanSegmentedReduceStage::Reduce;
  const rund::kernel::SegmentedReduceDesc executable_desc =
      reduce ? rund::kernel::SegmentedReduceDesc{.op = desc.op,
                                                 .element = desc.element}
             : rund::kernel::SegmentedReduceDesc{};
  const rund::kernel::SegmentedReduceHash hash =
      rund::kernel::HashSegmentedReduce(executable_desc);
  const std::uint64_t salt = static_cast<std::uint64_t>(stage);
  const bool wide = reduce && desc.element == rund::kernel::ReduceElement::U64;
  const rund::kernel::ComputeDomain executable_domain =
      !reduce ? rund::kernel::ComputeDomain::U32
              : (IsSignedDomain(domain)
                     ? (wide ? rund::kernel::ComputeDomain::I64
                             : rund::kernel::ComputeDomain::I32)
                     : (wide ? rund::kernel::ComputeDomain::U64
                             : rund::kernel::ComputeDomain::U32));
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5345475245440000ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = wide
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] VulkanCollectivePipeline *AcquireSource(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage stage, std::string source) {
  const rund::kernel::ComputePlan pseudo = PseudoPlan(desc, domain, stage);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter,
                                         kVulkanSegmentedReduceBindings, 0u,
                                         pseudo, artifact);
}

} // namespace

VulkanCollectivePipeline *
AcquireVulkanSegmentedReducePipeline(
    VulkanAdapter &adapter, const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage stage) {
  return AcquireSource(
      adapter, desc, domain, stage,
      VulkanSegmentedReduceSource(plan, domain, stage));
}

#endif

} // namespace rund::node::accel::detail
