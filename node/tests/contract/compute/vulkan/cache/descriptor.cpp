#include "support.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/kernel/step/map/stride.hpp"
#include "../../../../../src/accel/vulkan/adapter/state.hpp"
#include "../../../../../src/accel/vulkan/descriptor.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <array>
#include <cstdint>
#endif

namespace rund::node::vulkan_cache_contract {

[[nodiscard]] int DescriptorRange() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  VulkanAdapter adapter{};
  adapter.storage_align = 16u;
  adapter.storage_limit = 128u;

  VkDescriptorBufferInfo info{
      .buffer = FakeHandle<VkBuffer>(1u), .offset = 16u, .range = 112u};
  if (!ValidStorage(adapter, info)) {
    return 30;
  }
  info.offset = 4u;
  if (ValidStorage(adapter, info) ||
      WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(2u),
                                      &info, 1u)) {
    return 31;
  }
  info.offset = 0u;
  info.range = 129u;
  if (ValidStorage(adapter, info) ||
      WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(2u),
                                      &info, 1u)) {
    return 32;
  }
  info.range = VK_WHOLE_SIZE;
  if (ValidStorage(adapter, info)) {
    return 33;
  }

  const VulkanBuffer owner{.buffer = FakeHandle<VkBuffer>(3u), .bytes = 64u};
  const VulkanStorageBinding overflow{&owner, 48u, 32u};
  if (WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(4u),
                                      &overflow, 1u)) {
    return 34;
  }

  const rund::kernel::ResidentBufferRef strided{
      .bytes = 400u,
      .offset_bytes = 4u,
      .element_bytes = 4u,
      .stride_bytes = 20u,
      .count = 20u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  StorageRange range{};
  if (PlanStorage(adapter, strided, 0u, strided.count, range)) {
    return 35;
  }
  std::uint64_t begin = 0u;
  std::uint64_t pages = 0u;
  constexpr std::array expected_base{0u, 144u, 272u};
  constexpr std::array expected_count{7u, 7u, 6u};
  constexpr std::array expected_bytes{128u, 124u, 116u};
  while (begin < strided.count) {
    if (pages >= expected_count.size() ||
        !PlanStoragePage(adapter, strided, begin, range) ||
        range.base != expected_base[pages] ||
        range.count != expected_count[pages] ||
        range.bytes != expected_bytes[pages] ||
        range.bytes > adapter.storage_limit) {
      return 36;
    }
    begin += range.count;
    ++pages;
  }
  if (begin != strided.count || pages != expected_count.size() ||
      PlanStoragePage(adapter, strided, strided.count, range) ||
      !PlanStorage(adapter, strided, 0u, expected_count[0], range)) {
    return 37;
  }
  return 0;
#else
  return 0;
#endif
}

} // namespace rund::node::vulkan_cache_contract
