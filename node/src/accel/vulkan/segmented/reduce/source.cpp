#include "model.hpp"

#include "source/classify.hpp"
#include "source/prefix.hpp"
#include "source/reduce.hpp"
#include "source/scatter.hpp"

#include "../../../domain.hpp"
#include "../../../kernel/backend/source/storage.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanSegmentedReduceSource(
    Sink &sink, const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage
        stage) noexcept(noexcept(sink.append(std::string_view{}))) {
  switch (stage) {
  case VulkanSegmentedReduceStage::Classify:
    return EmitVulkanSegmentedReduceClassifySource(sink);
  case VulkanSegmentedReduceStage::Prefix:
    return EmitVulkanSegmentedReducePrefixSource(sink);
  case VulkanSegmentedReduceStage::Scatter:
    return EmitVulkanSegmentedReduceScatterSource(sink);
  case VulkanSegmentedReduceStage::Reduce:
    return EmitVulkanSegmentedReduceReduceSource(sink, plan, domain);
  }
  return false;
}

} // namespace

std::string
VulkanSegmentedReduceSource(const rund::kernel::SegmentedReducePlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            const VulkanSegmentedReduceStage stage) {
  std::uint64_t exact_bytes = 0u;
  const auto emit =
      [&](auto &sink) noexcept(noexcept(
          EmitVulkanSegmentedReduceSource(sink, plan, domain, stage))) {
        return EmitVulkanSegmentedReduceSource(sink, plan, domain, stage);
      };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanSegmentedReduceSourceBytes(
    const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedReduceStage stage, std::uint64_t &bytes) noexcept {
  const auto emit =
      [&](auto &sink) noexcept(noexcept(
          EmitVulkanSegmentedReduceSource(sink, plan, domain, stage))) {
        return EmitVulkanSegmentedReduceSource(sink, plan, domain, stage);
      };
  return backend_source_recipe::bytes(emit, bytes);
}

#endif

} // namespace rund::node::accel::detail
