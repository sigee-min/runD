#include "source/classify.hpp"
#include "source/dispatch.hpp"
#include "source/prefix.hpp"
#include "source/scatter.hpp"
namespace rund::node::accel::detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitVulkanSortSource(
    Sink &sink, const rund::kernel::SortKey key, const SortStage stage)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  const rund::kernel::u32 local_size =
      stage == SortStage::Dispatch ? 1u : kVulkanSortThreadCount;
  const bool chunked =
      stage == SortStage::Classify || stage == SortStage::Scatter;
  if (!AppendVulkanSortBaseSource(sink, key, local_size, chunked)) {
    return false;
  }
  const std::string_view key_type = VulkanSortKeyType(key);
  switch (stage) {
  case SortStage::Dispatch:
    return AppendVulkanSortDispatchSource(sink);
  case SortStage::Classify:
    return AppendVulkanSortClassifySource(sink, key_type);
  case SortStage::Prefix:
    return AppendVulkanSortPrefixSource(sink);
  case SortStage::Base:
    return AppendVulkanSortBaseOffsetSource(sink);
  case SortStage::Scatter:
    return AppendVulkanSortScatterSource(sink, key_type);
  }
  return false;
}

} // namespace

std::string VulkanSortSource(const rund::kernel::SortKey key,
                             const SortStage stage) {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanSortSource(sink, key, stage))) {
    return EmitVulkanSortSource(sink, key, stage);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool VulkanSortSourceBytes(const rund::kernel::SortKey key,
                           const SortStage stage,
                           std::uint64_t &bytes) noexcept {
  const auto emit = [&](auto &sink)
      noexcept(noexcept(EmitVulkanSortSource(sink, key, stage))) {
    return EmitVulkanSortSource(sink, key, stage);
  };
  return backend_source_recipe::bytes(emit, bytes);
}
#endif
} // namespace rund::node::accel::detail
