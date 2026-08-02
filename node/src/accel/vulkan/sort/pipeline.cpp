#include "local/api.hpp"
#include "../kernel/source_recipe.hpp"
#include <kernel/program/compute/sort/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::SortDesc &desc,
           const rund::kernel::ComputeApi api,
           const SortStage stage) noexcept {
  // The shader consumes pass/count/value policy through Params and bindings.
  // Only key width and stage are compile-time executable semantics.
  const rund::kernel::SortHash hash = rund::kernel::HashSort(
      rund::kernel::SortDesc{.key = desc.key,
                             .value = rund::kernel::SortValue::U32,
                             .radix_bits = 8u,
                             .stable = true});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage) + 1u;
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x534f52542e535400ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = api,
      .scalar = desc.key == rund::kernel::SortKey::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireSortPipeline(VulkanAdapter &adapter, const rund::kernel::SortDesc &desc,
                    const SortStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoPlan(desc, rund::kernel::ComputeApi::Vulkan, stage);
  std::string source = VulkanSortSource(desc.key, stage);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  const std::uint32_t push_bytes =
      stage == SortStage::Classify || stage == SortStage::Scatter
          ? kSortPushBytes
          : 0u;
  return AcquireVulkanCollectivePipeline(adapter, kSortDescriptorCount,
                                         push_bytes, pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
