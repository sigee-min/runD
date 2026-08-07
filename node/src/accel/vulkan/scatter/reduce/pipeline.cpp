#include "../../kernel/artifact.hpp"
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
                                      .element_count = 0u,
                                      .output_count = 0u});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5343524455434500ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = plan.element_bytes == 8u ? rund::kernel::ComputeScalar::Lane64
                                         : rund::kernel::ComputeScalar::Lane32,
      .domain = plan.domain,
      .fixed_format = plan.fixed_format,
      .ok = true,
      .reason = "ok"};
}

} // namespace

VulkanCollectivePipeline *
AcquireVulkanScatterReducePipeline(VulkanAdapter &adapter,
                                   const rund::kernel::ScatterReducePlan &plan,
                                   const VulkanScatterReduceStage stage) {
  const rund::kernel::ComputePlan pseudo = PseudoPlan(plan, stage);
  std::string source = VulkanScatterReduceSource(plan, stage);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kVulkanScatterReduceBindings,
                                         0u, pseudo, artifact);
}

#endif

} // namespace rund::node::accel::detail
