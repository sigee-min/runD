#include "local.hpp"

#include "source/block.hpp"
#include "source/offset.hpp"
#include "source/prefix.hpp"
#include "source/prelude.hpp"
#include "../kernel/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanSegmentedScanSource(
    Sink &sink, const rund::kernel::SegmentedScanElement element,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedScanStage stage)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  VulkanSourceTextSink source{sink};
  AppendSegmentedPrelude(source, element, domain,
                         stage != VulkanSegmentedScanStage::Prefix);
  switch (stage) {
  case VulkanSegmentedScanStage::Block:
    AppendSegmentedBlock(source, element);
    break;
  case VulkanSegmentedScanStage::Prefix:
    AppendSegmentedPrefix(source, element);
    break;
  case VulkanSegmentedScanStage::Offset:
    AppendSegmentedOffset(source, element);
    break;
  }
  source += "}\n";
  return source.ok();
}

} // namespace

std::string
VulkanSegmentedScanSource(const rund::kernel::SegmentedScanElement element,
                          const rund::kernel::ComputeDomain domain,
                          const VulkanSegmentedScanStage stage) {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanSegmentedScanSource(sink, element, domain,
                                                       stage))) {
    return EmitVulkanSegmentedScanSource(sink, element, domain, stage);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanSegmentedScanSourceBytes(
    const rund::kernel::SegmentedScanElement element,
    const rund::kernel::ComputeDomain domain,
    const VulkanSegmentedScanStage stage, std::uint64_t &bytes) noexcept {
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanSegmentedScanSource(sink, element, domain,
                                                       stage))) {
    return EmitVulkanSegmentedScanSource(sink, element, domain, stage);
  };
  return backend_source_recipe::bytes(emit, bytes);
}
#endif

} // namespace rund::node::accel::detail
