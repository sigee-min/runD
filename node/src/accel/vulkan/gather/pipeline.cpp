#include "../kernel/artifact.hpp"
#include "local.hpp"
#include <kernel/program/compute/gather/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoGatherPlan(const rund::kernel::GatherDesc &desc,
                 const rund::kernel::ComputeApi api,
                 const bool control) noexcept {
  const rund::kernel::GatherHash hash = rund::kernel::HashGather(desc);
  return rund::kernel::ComputePlan{
      .op_hash_hi =
          hash.hi ^ (control ? 0x4741544845524354ull : 0x474154484552434full),
      .op_hash_lo =
          hash.lo ^ (control ? 0x9e3779b97f4a7c15ull : 0xbf58476d1ce4e5b9ull),
      .api = api,
      .scalar = desc.element == rund::kernel::GatherElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireGatherPipeline(VulkanAdapter &adapter,
                      const rund::kernel::GatherDesc &desc,
                      const bool control) {
  const rund::kernel::ComputePlan pseudo =
      PseudoGatherPlan(desc, rund::kernel::ComputeApi::Vulkan, control);
  std::string source = VulkanGatherSource(desc.element, control);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kGatherDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
