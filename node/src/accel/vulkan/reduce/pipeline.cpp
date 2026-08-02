#include "local.hpp"
#include "../../domain.hpp"
#include "../kernel/source_recipe.hpp"
#include <kernel/program/compute/reduce/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoReducePlan(const rund::kernel::ReduceDesc &desc,
                 const rund::kernel::ComputeDomain domain,
                 const rund::kernel::ComputeApi api) noexcept {
  const rund::kernel::ReduceHash hash = rund::kernel::HashReduce(
      rund::kernel::ReduceDesc{.op = desc.op,
                               .element = desc.element,
                               .block_size = desc.block_size});
  const bool wide = desc.element == rund::kernel::ReduceElement::U64;
  const rund::kernel::ComputeDomain executable_domain =
      IsSignedDomain(domain)
          ? (wide ? rund::kernel::ComputeDomain::I64
                  : rund::kernel::ComputeDomain::I32)
          : (wide ? rund::kernel::ComputeDomain::U64
                  : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi,
      .op_hash_lo = hash.lo,
      .api = api,
      .scalar = desc.element == rund::kernel::ReduceElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *
AcquireReducePipeline(VulkanAdapter &adapter,
                      const rund::kernel::ReduceDesc &desc,
                      const rund::kernel::ComputeDomain domain) {
  const rund::kernel::ComputePlan pseudo =
      PseudoReducePlan(desc, domain, rund::kernel::ComputeApi::Vulkan);
  std::string source =
      VulkanReduceSource(desc.op, desc.element, desc.block_size, domain);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact = VulkanBackendArtifact(
      pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kReduceDescriptorCount, 0u,
                                         pseudo, artifact);
}
#endif

} // namespace rund::node::accel::detail
