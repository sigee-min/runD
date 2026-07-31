#include <accel/check.hpp>
#include <accel/device.hpp>

#include <rund/counter.hpp>
#include "../../../resident/ref.hpp"
#include "../../../resident/result.hpp"
#include "../../../resident/usage.hpp"
#include "../../../resident/validation.hpp"
#include "../../resident/access.hpp"
#include "../../resident/storage.hpp"
#include "../local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
MetalResidentBufferResult
CreateMetalResidentBuffer(const rund::AccelDevice &pick,
                          const ResidentDesc &desc,
                          const bool zero_initialize) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->device == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  const char *const desc_reason = ResidentDescReason(desc);
  if (std::string_view{desc_reason} != "ok") {
    return RejectResident<MetalResidentBufferResult>(desc_reason);
  }
  NSUInteger length = 0u;
  if (!ToNSUInteger(desc.bytes, length)) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_resident_bytes_invalid");
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  if (device == nil) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_device_unavailable");
  }
  id<MTLBuffer> metal_buffer =
      [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (metal_buffer == nil) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_buffer_unavailable");
  }
  if (zero_initialize) {
    void *const contents = [metal_buffer contents];
    if (length != 0u && contents == nullptr) {
      return RejectResident<MetalResidentBufferResult>(
          "accel_metal_buffer_unavailable");
    }
    if (length != 0u) {
      std::memset(contents, 0, length);
    }
  }
  std::shared_ptr<void> handle =
      RetainMetalObject((__bridge void *)metal_buffer);
  if (handle == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_buffer_unavailable");
  }
  std::shared_ptr<MetalResidentOwner> owner;
  try {
    owner = std::make_shared<MetalResidentOwner>();
  } catch (const std::bad_alloc &) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_buffer_unavailable");
  }
  std::unique_lock adapter_lock{adapter->mutex};
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard resident_lock{resident.mutex};
  if (resident.next_id == std::numeric_limits<std::uint64_t>::max()) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_id_unavailable");
  }
  const std::uint64_t id = resident.next_id;
  owner->adapter = adapter;
  owner->adapter_owner = pick.owner;
  owner->buffer = std::move(handle);
  owner->id = id;
  try {
    const auto [entry, inserted] = resident.buffers.emplace(
        id,
        MetalResidentBuffer{ResidentEntry{.id = id,
                                          .bytes = desc.bytes,
                                          .element_bytes = desc.element_bytes,
                                          .stride_bytes = desc.stride_bytes,
                                          .count = desc.count,
                                          .usage = desc.usage,
                                          .read_capable = ReadCapable(desc),
                                          .write_capable = WriteCapable(desc),
                                          .owner = owner},
                            owner->buffer});
    if (!inserted) {
      owner->adapter = nullptr;
      owner.reset();
      return RejectResident<MetalResidentBufferResult>(
          "accel_metal_resident_id_unavailable");
    }
    (void)entry;
  } catch (const std::bad_alloc &) {
    owner->adapter = nullptr;
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_buffer_unavailable");
  }
  ++resident.next_id;
  ::rund::detail::counter::Accumulate(adapter->stats.buffer_allocation_count,
                                      1u);
  return MetalResidentBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromDesc(id, desc),
      .handle = owner,
      .device_buffer = owner->buffer,
  };
}
#else
MetalResidentBufferResult CreateMetalResidentBuffer(const rund::AccelDevice &,
                                                    const ResidentDesc &,
                                                    const bool) {
  return RejectResident<MetalResidentBufferResult>("accel_metal_unavailable");
}
#endif

} // namespace rund::node::accel::detail
