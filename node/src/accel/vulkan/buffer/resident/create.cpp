#include <accel/check.hpp>
#include <accel/device.hpp>

#include <rund/counter.hpp>
#include "../../../resident/ref.hpp"
#include "../../../resident/result.hpp"
#include "../../../resident/usage.hpp"
#include "../../../resident/validation.hpp"
#include "../../adapter/api.hpp"
#include "../../command.hpp"
#include "../../status.hpp"
#include "../create/api.hpp"
#include "../local.hpp"
#include "../transfer/range.hpp"
#include "pool.hpp"
#include "storage.hpp"

#include <limits>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanResidentBufferResult
CreateVulkanResidentBuffer(const rund::AccelDevice &pick,
                           const ResidentDesc &desc,
                           const bool zero_initialize) {
  if (!VulkanPickOwnsAdapter(pick)) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_buffer_backend_unavailable");
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  const char *const desc_reason = ResidentDescReason(desc);
  if (std::string_view{desc_reason} != "ok") {
    return RejectResident<VulkanResidentBufferResult>(desc_reason);
  }
  VulkanBuffer buffer{};
  VkDeviceSize storage_bytes = 0u;
  if (!AlignVulkanTransferStorage(desc.bytes, storage_bytes)) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_vulkan_memory_unavailable");
  }
  std::shared_ptr<VulkanResidentStorage> storage;
  std::shared_ptr<VulkanResidentOwner> owner;
  try {
    storage = std::make_shared<VulkanResidentStorage>();
    owner = std::make_shared<VulkanResidentOwner>();
  } catch (const std::bad_alloc &) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_vulkan_memory_unavailable");
  }
  bool reused = false;
  std::unique_lock adapter_lock{adapter->mutex};
  // Zeroed creation records a clear on the shared bounded command ring.
  // Backpressure before taking resident state preserves the adapter->resident
  // lock order and prevents transient queue pressure from rejecting a valid
  // buffer allocation. Full-overwrite creation records no command.
  if (zero_initialize) {
    WaitForVulkanCommandSlot(*adapter, adapter_lock);
  }
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard resident_lock{resident.mutex};
  if (resident.next_id == std::numeric_limits<std::uint64_t>::max()) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_vulkan_resident_id_unavailable");
  }
  reused = TakeVulkanResidentStorage(
      *adapter, storage_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer);
  if (reused) {
    ::rund::detail::counter::Accumulate(adapter->buffer_reuse_hit_count, 1u);
  } else if (!CreateFreshVulkanBuffer(*adapter, storage_bytes,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VulkanMemoryUse::Resident, buffer)) {
    return RejectResident<VulkanResidentBufferResult>(VulkanLastError(adapter));
  }
  // A pooled allocation is a physical storage optimization, not a logical
  // Buffer identity.  Public zero-initialized creation clears fresh and reused
  // storage before publishing a new owner.  Full-overwrite upload/private
  // intents skip this pass because their caller proves every logical byte is
  // replaced before observation.
  if (zero_initialize &&
      (!EnsureVulkanCommandResources(*adapter) ||
       !BeginVulkanCommand(*adapter) ||
       !ResetVulkanStatus(adapter->command_buffer, buffer, storage_bytes) ||
       !SubmitVulkanCommand(*adapter, false))) {
    RetireVulkanResidentStorage(*adapter, buffer);
    return RejectResident<VulkanResidentBufferResult>(VulkanLastError(adapter));
  }
  const std::uint64_t id = resident.next_id;
  storage->adapter = adapter;
  storage->buffer = buffer;
  buffer = {};
  owner->adapter = adapter;
  owner->adapter_owner = pick.owner;
  owner->storage = storage;
  owner->id = id;
  const auto discard = [&] {
    buffer = storage->buffer;
    storage->adapter = nullptr;
    storage->buffer = {};
    owner->adapter = nullptr;
    owner->adapter_owner.reset();
    owner->storage.reset();
    RetireVulkanResidentStorage(*adapter, buffer);
  };
  try {
    const auto [entry, inserted] = resident.buffers.emplace(
        id,
        VulkanResidentBuffer{ResidentEntry{.id = id,
                                           .bytes = desc.bytes,
                                           .element_bytes = desc.element_bytes,
                                           .stride_bytes = desc.stride_bytes,
                                           .count = desc.count,
                                           .usage = desc.usage,
                                           .read_capable = ReadCapable(desc),
                                           .write_capable = WriteCapable(desc),
                                           .owner = owner},
                             storage, &storage->buffer});
    if (!inserted) {
      discard();
      return RejectResident<VulkanResidentBufferResult>(
          "accel_vulkan_resident_id_unavailable");
    }
    (void)entry;
  } catch (const std::bad_alloc &) {
    discard();
    return RejectResident<VulkanResidentBufferResult>(
        "accel_vulkan_memory_unavailable");
  }
  ++resident.next_id;
  return VulkanResidentBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromDesc(id, desc),
      .handle = owner,
      .storage = storage,
      .device_buffer = &storage->buffer,
      .storage_bytes = storage->buffer.bytes,
      .storage_reused = reused,
  };
}
#endif

} // namespace rund::node::accel::detail
