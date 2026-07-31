#pragma once

#include <array>
#include <cstdint>
#include <limits>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr VkDeviceSize kVulkanTransferAlignment = 4u;

struct VulkanTransferRange final {
  VkDeviceSize offset = 0u;
  VkDeviceSize bytes = 0u;
  VkDeviceSize host_offset = 0u;
};

struct VulkanUploadPreservation final {
  std::array<VkBufferCopy, 2u> regions{};
  std::uint32_t region_count = 0u;
  VkDeviceSize padding_offset = 0u;
  VkDeviceSize padding_bytes = 0u;
};

[[nodiscard]] inline bool AlignVulkanTransferStorage(
    const std::uint64_t bytes, VkDeviceSize &storage) noexcept {
  constexpr std::uint64_t mask = kVulkanTransferAlignment - 1u;
  if (bytes == 0u || bytes > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  storage = static_cast<VkDeviceSize>((bytes + mask) & ~mask);
  return true;
}

[[nodiscard]] inline bool ResolveVulkanTransferRange(
    const std::uint64_t offset, const std::uint64_t bytes,
    const VkDeviceSize storage, VulkanTransferRange &range) noexcept {
  constexpr std::uint64_t mask = kVulkanTransferAlignment - 1u;
  if (bytes == 0u || offset > std::numeric_limits<std::uint64_t>::max() - bytes) {
    return false;
  }
  const std::uint64_t end = offset + bytes;
  if (end > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  const std::uint64_t aligned_offset = offset & ~mask;
  const std::uint64_t aligned_end = (end + mask) & ~mask;
  if (aligned_end > storage) {
    return false;
  }
  range = VulkanTransferRange{
      .offset = static_cast<VkDeviceSize>(aligned_offset),
      .bytes = static_cast<VkDeviceSize>(aligned_end - aligned_offset),
      .host_offset = static_cast<VkDeviceSize>(offset - aligned_offset),
  };
  return true;
}

[[nodiscard]] inline bool ResolveVulkanUploadPreservation(
    const VulkanTransferRange &range, const std::uint64_t offset,
    const std::uint64_t bytes, const std::uint64_t semantic_bytes,
    VulkanUploadPreservation &preservation) noexcept {
  preservation = {};
  if (bytes == 0u || offset > semantic_bytes ||
      bytes > semantic_bytes - offset ||
      range.offset > std::numeric_limits<VkDeviceSize>::max() - range.bytes) {
    return false;
  }
  const VkDeviceSize range_end = range.offset + range.bytes;
  const std::uint64_t request_end = offset + bytes;
  if (range.offset > offset || range_end < request_end ||
      range.host_offset != offset - range.offset) {
    return false;
  }

  const auto preserve_word = [&](const VkDeviceSize source_offset) {
    if (source_offset < range.offset || source_offset > range_end ||
        kVulkanTransferAlignment > range_end - source_offset) {
      return false;
    }
    for (std::uint32_t index = 0u; index < preservation.region_count; ++index) {
      if (preservation.regions[index].srcOffset == source_offset) {
        return true;
      }
    }
    if (preservation.region_count >= preservation.regions.size()) {
      return false;
    }
    preservation.regions[preservation.region_count++] = VkBufferCopy{
        .srcOffset = source_offset,
        .dstOffset = source_offset - range.offset,
        .size = kVulkanTransferAlignment,
    };
    return true;
  };

  if (range.offset < offset && !preserve_word(range.offset)) {
    return false;
  }
  if (request_end < semantic_bytes && request_end < range_end) {
    const VkDeviceSize suffix_word =
        static_cast<VkDeviceSize>(request_end) &
        ~(kVulkanTransferAlignment - 1u);
    if (!preserve_word(suffix_word)) {
      return false;
    }
  } else if (request_end == semantic_bytes && request_end < range_end) {
    preservation.padding_offset =
        static_cast<VkDeviceSize>(request_end) - range.offset;
    preservation.padding_bytes =
        range_end - static_cast<VkDeviceSize>(request_end);
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail
