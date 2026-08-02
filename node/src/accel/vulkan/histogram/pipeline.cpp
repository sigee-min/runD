#include "local.hpp"
#include "../kernel/source_recipe.hpp"
#include <kernel/program/compute/histogram/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoHistogramPlan(const rund::kernel::HistogramDesc &desc,
                    const bool clear) noexcept {
  const rund::kernel::HistogramHash hash = rund::kernel::HashHistogram(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi = clear ? hash.hi ^ 0x9e3779b97f4a7c15ull : hash.hi,
      .op_hash_lo = clear ? hash.lo ^ 0x517cc1b727220a95ull : hash.lo,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireHistogramPipeline(VulkanAdapter &adapter,
                         const rund::kernel::HistogramDesc &desc,
                         const bool clear) {
  const rund::kernel::ComputePlan pseudo = PseudoHistogramPlan(desc, clear);
  std::string source = VulkanHistogramSource(clear);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kHistogramDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
