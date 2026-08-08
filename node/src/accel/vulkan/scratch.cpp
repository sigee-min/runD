#include "scratch.hpp"

#include "buffer/resident/find.hpp"
#include "resident/access.hpp"

#include <algorithm>
#include <mutex>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

thread_local VulkanScratch *active_scratch{};

} // namespace

VulkanScratch::VulkanScratch(const rund::AccelDevice &pick,
                             const KernelScratchLayout &layout,
                             const RunBinds &binds) noexcept {
  if (!VulkanPickOwnsAdapter(pick) || !ValidKernelScratch(layout, binds)) {
    return;
  }
  adapter_ = static_cast<VulkanAdapter *>(pick.backend.context);
  if (adapter_ == nullptr || adapter_->storage_align == 0u) {
    return;
  }
  try {
    pages_.reserve(layout.size());
    VulkanResidentState &resident = VulkanResidents(*adapter_);
    std::lock_guard lock{resident.mutex};
    for (const KernelScratchPage page : layout) {
      const rund::kernel::ResidentBufferRef &ref = binds.refs()[page.slot];
      VulkanResidentBufferResult resolved =
          ResolveVulkanResidentBuffer(resident, ref, binds.handles()[page.slot],
                                      "compute_resident_id_invalid");
      if (!resolved.check.ok || resolved.device_buffer == nullptr ||
          ref.offset_bytes > resolved.device_buffer->bytes ||
          page.bytes > resolved.device_buffer->bytes - ref.offset_bytes ||
          ref.offset_bytes > std::numeric_limits<VkDeviceSize>::max() ||
          page.bytes > std::numeric_limits<VkDeviceSize>::max()) {
        pages_.clear();
        return;
      }
      pages_.push_back(Page{
          .buffer = resolved.device_buffer,
          .base = static_cast<VkDeviceSize>(ref.offset_bytes),
          .bytes = static_cast<VkDeviceSize>(page.bytes),
      });
    }
  } catch (const std::bad_alloc &) {
    pages_.clear();
    return;
  }
  valid_ = pages_.size() == layout.size();
}

bool VulkanScratch::valid() const noexcept { return valid_; }

bool VulkanScratch::used() const noexcept { return used_; }

bool VulkanScratch::active() const noexcept { return scratch::active(pages_); }

std::uint64_t VulkanScratch::page_bytes() const noexcept {
  std::uint64_t result = 0u;
  for (const Page &page : pages_) {
    result = std::max(result, static_cast<std::uint64_t>(page.bytes));
  }
  return result;
}

void VulkanScratch::reset() noexcept { scratch::reset(pages_); }

bool VulkanScratch::acquire(const VkDeviceSize bytes,
                            const VkBufferUsageFlags usage,
                            VulkanBuffer &buffer) noexcept {
  if (!valid_ || bytes == 0u || adapter_ == nullptr) {
    return false;
  }
  const scratch::Placement placed =
      scratch::fit(pages_, adapter_->storage_align, bytes);
  if (!placed.ok || placed.page >= pages_.size()) {
    return false;
  }
  Page &page = pages_[placed.page];
  if (page.buffer == nullptr || page.buffer->buffer == VK_NULL_HANDLE) {
    return false;
  }
  buffer = VulkanBuffer{
      .buffer = page.buffer->buffer,
      .memory = VK_NULL_HANDLE,
      .bytes = bytes,
      .usage = usage,
      .memory_flags = page.buffer->memory_flags,
      .mapped = nullptr,
      .offset = page.base + placed.offset,
      .memory_use = VulkanMemoryUse::Scratch,
      .memory_lease = false,
      .borrowed = true,
  };
  used_ = true;
  return true;
}

bool VulkanScratch::acquire(const KernelScratchPlacement &placement,
                            const VkBufferUsageFlags usage,
                            VulkanBuffer &buffer) noexcept {
  if (!valid_ || adapter_ == nullptr || placement.page >= pages_.size()) {
    return false;
  }
  Page &page = pages_[placement.page];
  if (page.buffer == nullptr || page.buffer->buffer == VK_NULL_HANDLE ||
      !scratch::valid(placement, adapter_->storage_align, page.bytes) ||
      placement.requirement.bytes > std::numeric_limits<VkDeviceSize>::max() ||
      placement.offset > std::numeric_limits<VkDeviceSize>::max()) {
    return false;
  }
  const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(placement.requirement.bytes);
  const VkDeviceSize offset = static_cast<VkDeviceSize>(placement.offset);
  page.used = std::max(page.used, offset + bytes);
  buffer = VulkanBuffer{
      .buffer = page.buffer->buffer,
      .memory = VK_NULL_HANDLE,
      .bytes = bytes,
      .usage = usage,
      .memory_flags = page.buffer->memory_flags,
      .mapped = nullptr,
      .offset = page.base + offset,
      .memory_use = VulkanMemoryUse::Scratch,
      .memory_lease = false,
      .borrowed = true,
  };
  used_ = true;
  return true;
}

VulkanScratchScope::VulkanScratchScope(VulkanScratch *const scratch) noexcept
    : prior_{active_scratch} {
  active_scratch = scratch;
}

VulkanScratchScope::~VulkanScratchScope() { active_scratch = prior_; }

VulkanScratch *ActiveVulkanScratch() noexcept { return active_scratch; }

#endif

} // namespace rund::node::accel::detail
