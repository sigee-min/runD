#pragma once

#include <accel/check.hpp>

#include "../../../resident/ref.hpp"
#include "../../../resident/result.hpp"
#include "../../../resident/slot.hpp"
#include "../../../resident/validation.hpp"
#include "../../resident/state.hpp"
#include "../../resident/storage.hpp"
#include "../owner.hpp"
#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalResidentMatch {
  MetalResidentBuffer *entry = nullptr;
  std::shared_ptr<void> owner{};
  const char *reason = nullptr;
};

[[nodiscard]] inline MetalResidentMatch MatchMetalResidentBuffer(
    MetalResidentBuffer &entry, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const bool allow_stride = false) {
  std::shared_ptr<void> owner = entry.owner.lock();
  const char *reason = "ok";
  if (!ResidentRefFits(entry, owner, ref, handle,
                       "accel_metal_resident_id_unavailable", reason,
                       allow_stride)) {
    return MetalResidentMatch{.entry = nullptr, .reason = reason};
  }
  return MetalResidentMatch{
      .entry = &entry, .owner = std::move(owner), .reason = "ok"};
}

[[nodiscard]] inline MetalResidentMatch FindMetalResidentBuffer(
    MetalResidentState &resident, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const char *const missing_reason,
    const bool allow_stride = false) {
  const auto found = resident.buffers.find(ref.id);
  return found == resident.buffers.end()
             ? MetalResidentMatch{.entry = nullptr, .reason = missing_reason}
             : MatchMetalResidentBuffer(found->second, ref, handle,
                                        allow_stride);
}

[[nodiscard]] inline MetalResidentBufferResult
MetalResidentResult(MetalResidentBuffer &entry, std::shared_ptr<void> owner) {
  std::shared_ptr<void> device_buffer = entry.device_buffer.lock();
  if (owner == nullptr || device_buffer == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_buffer_unavailable");
  }
  return MetalResidentBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromResident(entry),
      .handle = owner,
      .device_buffer = device_buffer,
  };
}

[[nodiscard]] inline MetalResidentBufferResult ResolveMetalResidentBuffer(
    MetalResidentState &resident, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const char *const missing_reason,
    const bool allow_stride = false) {
  const MetalResidentMatch match = FindMetalResidentBuffer(
      resident, ref, handle, missing_reason, allow_stride);
  return match.entry == nullptr
             ? RejectResident<MetalResidentBufferResult>(match.reason)
             : MetalResidentResult(*match.entry, std::move(match.owner));
}

[[nodiscard]] inline bool
SameResidentRef(const rund::kernel::ResidentBufferRef &left,
                const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

// Pipeline-private routes retain the exact MetalResidentOwner authenticated
// during cold Buffer/Pipeline preparation. That owner directly retains the
// MTLBuffer, so the warm route needs neither the adapter gate nor the resident
// registry gate.
[[nodiscard]] inline MetalResidentBufferResult
ResolvePrivateMetalResidentBuffer(MetalAdapter &adapter,
                                  const rund::kernel::ResidentBufferRef &ref,
                                  const std::shared_ptr<void> &handle) {
  if (handle == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_buffer_unavailable");
  }
  const std::shared_ptr<MetalResidentOwner> owner =
      LookupMetalResidentOwner(handle);
  if (owner == nullptr || owner->adapter != &adapter ||
      owner->adapter_owner == nullptr || owner->buffer == nullptr ||
      owner->id != ref.id || !SameResidentRef(owner->ref, ref)) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_buffer_unavailable");
  }
  return MetalResidentBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = owner->ref,
      .handle = owner,
      .device_buffer = owner->buffer,
  };
}
#endif
} // namespace rund::node::accel::detail
