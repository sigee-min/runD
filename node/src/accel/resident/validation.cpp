#include "validation.hpp"

#include "../backend/match.hpp"
#include "ref.hpp"
#include "usage.hpp"

#include <kernel/program/compute/binding/validation.hpp>

namespace rund::node::accel::detail {

const char *ResidentDescReason(const ResidentDesc &desc) noexcept {
  if (!KnownResidentUsage(desc.usage)) {
    return "compute_resident_usage_invalid";
  }
  const rund::kernel::ResidentBufferRef ref = RefFromDesc(1u, desc);
  const rund::kernel::BindingValidation validation =
      rund::kernel::ValidateResidentBuffer(ref, desc.count, desc.element_bytes,
                                           desc.usage);
  return validation.ok ? "ok" : validation.reason;
}

bool ResidentRefFits(const ResidentEntry &entry,
                     const std::shared_ptr<void> &owner,
                     const rund::kernel::ResidentBufferRef &ref,
                     const std::shared_ptr<void> &handle,
                     const char *const id_reason, const char *&reason,
                     const bool allow_stride) noexcept {
  if (ref.id == 0u || ref.id != entry.id) {
    reason = id_reason;
    return false;
  }
  if (owner == nullptr || !SameOwner(entry.owner, owner) ||
      !SameObject(owner, handle)) {
    reason = "accel_buffer_unavailable";
    return false;
  }
  if (!KnownResidentUsage(ref.usage) ||
      (ref.usage == rund::kernel::kResidentUsageRead && !entry.read_capable) ||
      (ref.usage == rund::kernel::kResidentUsageWrite &&
       !entry.write_capable)) {
    reason = "compute_resident_usage_invalid";
    return false;
  }
  if (ref.bytes == 0u || ref.bytes > entry.bytes) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  const rund::kernel::BindingValidation validation =
      rund::kernel::ValidateResidentBuffer(ref, ref.count, ref.element_bytes,
                                           ref.usage, allow_stride);
  if (!validation.ok) {
    reason = validation.reason;
    return false;
  }
  if (ref.count == 0u) {
    reason = "compute_resident_stride_invalid";
    return false;
  }
  reason = "ok";
  return true;
}

} // namespace rund::node::accel::detail
