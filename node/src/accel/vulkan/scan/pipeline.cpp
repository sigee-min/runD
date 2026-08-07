#include "../collective/pipeline.hpp"
#include "../../domain.hpp"
#include "../kernel/artifact.hpp"
#include "local.hpp"
#include "pipeline.hpp"
#include "source.hpp"
#include <kernel/program/compute/scan/identity.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::kernel::ComputePlan
PseudoPlan(const rund::kernel::ScanDesc &desc,
           const rund::kernel::ComputeDomain domain,
           const rund::kernel::ComputeApi api,
           const VulkanScanStage stage) noexcept {
  // Runtime sizes and count-source placement do not change this shader. Keep
  // the cache identity at executable semantics so differently sized scans
  // share the same native pipeline.
  const rund::kernel::ScanOp executable_op =
      stage == VulkanScanStage::Block ? desc.op
                                      : rund::kernel::ScanOp::ExclusiveSum;
  const rund::kernel::ScanHash hash = rund::kernel::HashScan(
      rund::kernel::ScanDesc{.op = executable_op, .element = desc.element});
  const std::uint64_t salt = static_cast<std::uint64_t>(stage) + 1u;
  const bool wide = desc.element == rund::kernel::ScanElement::U64;
  const rund::kernel::ComputeDomain executable_domain =
      IsSignedDomain(domain) ? (wide ? rund::kernel::ComputeDomain::I64
                                     : rund::kernel::ComputeDomain::I32)
                             : (wide ? rund::kernel::ComputeDomain::U64
                                     : rund::kernel::ComputeDomain::U32);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash.hi ^ (0x5343414e2e535400ull + salt),
      .op_hash_lo = hash.lo ^ (salt * 0x9e3779b97f4a7c15ull),
      .api = api,
      .scalar = desc.element == rund::kernel::ScanElement::U64
                    ? rund::kernel::ComputeScalar::Lane64
                    : rund::kernel::ComputeScalar::Lane32,
      .domain = executable_domain,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanScanPipeline(
    VulkanAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ComputeDomain domain, const VulkanScanStage stage) {
  const rund::kernel::ComputePlan pseudo =
      PseudoPlan(desc, domain, rund::kernel::ComputeApi::Vulkan, stage);
  const bool inclusive = desc.op == rund::kernel::ScanOp::InclusiveSum;
  std::string source = VulkanScanSource(desc.element, domain, stage, inclusive);
  const std::uint64_t source_bytes = source.size();
  const rund::kernel::LoweringArtifact artifact =
      MakeVulkanBackendArtifact(pseudo, std::move(source), source_bytes);
  if (!artifact.ok) {
    return nullptr;
  }
  const std::uint32_t push_bytes =
      stage == VulkanScanStage::Prefix ? 0u : kScanPushBytes;
  return AcquireVulkanCollectivePipeline(adapter, kScanDescriptorCount,
                                         push_bytes, pseudo, artifact);
}

#endif

} // namespace rund::node::accel::detail
