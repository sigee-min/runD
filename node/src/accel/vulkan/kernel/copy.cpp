#include "copy.hpp"
#include "../descriptor.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool PlanVulkanCopyRange(const VulkanAdapter &adapter,
                         const rund::kernel::ResidentBufferRef &ref,
                         const VulkanBuffer *const buffer,
                         VulkanCopyRange &range) noexcept {
  constexpr std::uint64_t word = sizeof(std::uint32_t);
  constexpr std::uint64_t word_limit =
      std::numeric_limits<std::uint32_t>::max();
  if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE ||
      ref.count == 0u || (ref.element_bytes != 4u && ref.element_bytes != 8u) ||
      ref.stride_bytes < ref.element_bytes || (ref.offset_bytes % word) != 0u ||
      (ref.stride_bytes % word) != 0u) {
    return false;
  }
  StorageRange planned{};
  if (!PlanStorage(adapter, ref, 0u, ref.count, planned) ||
      planned.base > buffer->bytes ||
      planned.bytes > buffer->bytes - planned.base) {
    return false;
  }
  const std::uint64_t offset_words = planned.offset / word;
  const std::uint64_t stride_words = ref.stride_bytes / word;
  const std::uint64_t element_words = ref.element_bytes / word;
  if (offset_words > word_limit || stride_words == 0u ||
      element_words - 1u > word_limit - offset_words ||
      ref.count - 1u >
          (word_limit - offset_words - (element_words - 1u)) / stride_words) {
    return false;
  }
  range = VulkanCopyRange{
      .base = planned.base,
      .bytes = planned.bytes,
      .offset_words = offset_words,
      .stride_words = stride_words,
  };
  return true;
}

#endif

} // namespace rund::node::accel::detail
