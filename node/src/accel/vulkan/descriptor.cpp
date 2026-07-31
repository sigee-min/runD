#include "descriptor/create.hpp"
#include "descriptor/write.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool ValidStorage(const VulkanAdapter &adapter,
                  const VkDescriptorBufferInfo &info) noexcept {
  return adapter.storage_align != 0u && adapter.storage_limit != 0u &&
         info.buffer != VK_NULL_HANDLE && info.range != 0u &&
         info.range != VK_WHOLE_SIZE &&
         info.offset % adapter.storage_align == 0u &&
         info.range <= adapter.storage_limit;
}

bool PlanStorage(const VulkanAdapter &adapter,
                 const rund::kernel::ResidentBufferRef &ref,
                 const std::uint64_t begin, const std::uint64_t count,
                 StorageRange &range) noexcept {
  range = {};
  if (adapter.storage_align == 0u || adapter.storage_limit == 0u ||
      ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
      count == 0u || begin > ref.count || count > ref.count - begin ||
      !rund::kernel::checked::mul(begin, ref.stride_bytes) ||
      !rund::kernel::checked::mul(count - 1u, ref.stride_bytes)) {
    return false;
  }
  const std::uint64_t begin_bytes = begin * ref.stride_bytes;
  const std::uint64_t tail = (count - 1u) * ref.stride_bytes;
  if (!rund::kernel::checked::add(ref.offset_bytes, begin_bytes) ||
      !rund::kernel::checked::add(tail, ref.element_bytes)) {
    return false;
  }
  const std::uint64_t byte_offset = ref.offset_bytes + begin_bytes;
  const std::uint64_t span = tail + ref.element_bytes;
  if (byte_offset > ref.bytes || span > ref.bytes - byte_offset) {
    return false;
  }
  const std::uint64_t base = byte_offset - byte_offset % adapter.storage_align;
  const std::uint64_t prefix = byte_offset - base;
  if (!rund::kernel::checked::add(prefix, span) ||
      prefix + span > adapter.storage_limit ||
      base > std::numeric_limits<VkDeviceSize>::max() ||
      prefix + span > std::numeric_limits<VkDeviceSize>::max()) {
    return false;
  }
  range = StorageRange{
      .base = static_cast<VkDeviceSize>(base),
      .bytes = static_cast<VkDeviceSize>(prefix + span),
      .offset = prefix,
      .count = count,
  };
  return true;
}

bool PlanStoragePage(const VulkanAdapter &adapter,
                     const rund::kernel::ResidentBufferRef &ref,
                     const std::uint64_t begin, StorageRange &range) noexcept {
  range = {};
  if (adapter.storage_align == 0u || adapter.storage_limit == 0u ||
      ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
      begin >= ref.count ||
      !rund::kernel::checked::mul(begin, ref.stride_bytes) ||
      !rund::kernel::checked::add(ref.offset_bytes, begin * ref.stride_bytes)) {
    return false;
  }
  const std::uint64_t byte_offset = ref.offset_bytes + begin * ref.stride_bytes;
  if (byte_offset > ref.bytes || ref.element_bytes > ref.bytes - byte_offset) {
    return false;
  }
  const std::uint64_t base = byte_offset - byte_offset % adapter.storage_align;
  const std::uint64_t prefix = byte_offset - base;
  if (prefix > adapter.storage_limit ||
      ref.element_bytes > adapter.storage_limit - prefix) {
    return false;
  }
  const std::uint64_t available =
      adapter.storage_limit - prefix - ref.element_bytes;
  const std::uint64_t capacity = 1u + available / ref.stride_bytes;
  const std::uint64_t count = std::min(ref.count - begin, capacity);
  return PlanStorage(adapter, ref, begin, count, range);
}

VulkanStorageBinding
VulkanStorageBindingFor(const VulkanBuffer *const buffer,
                        const rund::kernel::ResidentBufferRef &ref) noexcept {
  constexpr rund::kernel::u64 max =
      std::numeric_limits<rund::kernel::u64>::max();
  if (buffer == nullptr || ref.count == 0u || ref.element_bytes == 0u ||
      ref.stride_bytes < ref.element_bytes ||
      ref.count - 1u > (max - ref.element_bytes) / ref.stride_bytes) {
    return {};
  }
  const rund::kernel::u64 bytes =
      (ref.count - 1u) * ref.stride_bytes + ref.element_bytes;
  if (ref.offset_bytes > ref.bytes || bytes > ref.bytes - ref.offset_bytes ||
      ref.offset_bytes > buffer->bytes ||
      bytes > buffer->bytes - ref.offset_bytes ||
      ref.offset_bytes > std::numeric_limits<VkDeviceSize>::max() ||
      bytes > std::numeric_limits<VkDeviceSize>::max()) {
    return {};
  }
  return VulkanStorageBinding{buffer,
                              static_cast<VkDeviceSize>(ref.offset_bytes),
                              static_cast<VkDeviceSize>(bytes)};
}

#endif

} // namespace rund::node::accel::detail
