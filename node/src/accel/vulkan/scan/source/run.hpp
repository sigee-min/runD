#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "offset.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanScanSource(
    Sink &sink, const rund::kernel::ScanElement element,
    const rund::kernel::ComputeDomain domain, const VulkanScanStage stage,
    const bool inclusive) noexcept(noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  AppendVulkanScanPrelude(source, element, domain, kVulkanScanWidth,
                          stage != VulkanScanStage::Prefix);
  switch (stage) {
  case VulkanScanStage::Block:
    AppendVulkanScanBlock(source, element, inclusive);
    break;
  case VulkanScanStage::Prefix:
    AppendVulkanScanPrefix(source, element);
    break;
  case VulkanScanStage::Offset:
    AppendVulkanScanOffset(source, element);
    break;
  }
  return source.valid();
}

std::string VulkanScanSource(const rund::kernel::ScanElement element,
                             const rund::kernel::ComputeDomain domain,
                             const VulkanScanStage stage,
                             const bool inclusive) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitVulkanScanSource(sink, element, domain, stage, inclusive);
  });
}

bool VulkanScanSourceBytes(const rund::kernel::ScanElement element,
                           const rund::kernel::ComputeDomain domain,
                           const VulkanScanStage stage, const bool inclusive,
                           std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanScanSource(sink, element, domain, stage, inclusive);
      },
      bytes);
}

} // namespace rund::node::accel::detail
