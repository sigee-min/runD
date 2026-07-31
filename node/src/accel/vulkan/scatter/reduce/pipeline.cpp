#include "model.hpp"

#include <kernel/program/compute/scatter/reduce/identity.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::ScatterReducePlan &plan,
           const VulkanScatterReduceStage stage) {
  const auto hash = rund::kernel::HashScatterReduce(
      rund::kernel::ScatterReduceDesc{.op = plan.op,
                                      .domain = plan.domain,
                                      .fixed_format = plan.fixed_format,
                                      .element_count = plan.element_count,
                                      .output_count = plan.output_count,
                                      .count_source = plan.count_source});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5343524455434500ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = plan.element_bytes == 8u ? rund::kernel::ComputeScalar::Lane64
                                         : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok"};
}

} // namespace

VulkanCollectivePipeline *
AcquireVulkanScatterReducePipeline(VulkanAdapter &adapter,
                                   const rund::kernel::ScatterReducePlan &plan,
                                   const VulkanScatterReduceStage stage) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = VulkanScatterReduceSource(plan, stage);
  artifact.ok = true;
  artifact.reason = "ok";
  return AcquireVulkanCollectivePipeline(adapter, kVulkanScatterReduceBindings,
                                         0u, PseudoPlan(plan, stage), artifact);
}

#endif

} // namespace rund::node::accel::detail
