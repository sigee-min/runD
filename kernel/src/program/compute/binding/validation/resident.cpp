#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::kernel {

using binding_validation_detail::Accept;
using binding_validation_detail::Reject;

BindingValidation ValidateResidentBuffer(const ResidentBufferRef &ref,
                                         const u64 tile_count,
                                         const u64 expected_element_bytes,
                                         const u32 expected_usage,
                                         const bool allow_stride) noexcept {
  if (ref.id == 0u) {
    return Reject("compute_resident_id_invalid");
  }
  if (ref.bytes == 0u) {
    return Reject("compute_resident_bytes_invalid");
  }
  if (ref.usage != expected_usage) {
    return Reject("compute_resident_usage_invalid");
  }
  if (ref.element_bytes != expected_element_bytes) {
    return expected_usage == kResidentUsageWrite
               ? Reject("compute_binding_output_bytes_mismatch")
               : Reject("compute_binding_input_bytes_mismatch");
  }
  if (ref.element_bytes == 0u ||
      (allow_stride ? ref.stride_bytes < ref.element_bytes
                    : ref.stride_bytes != ref.element_bytes) ||
      ref.count < tile_count) {
    return Reject("compute_resident_stride_invalid");
  }

  u64 extent = ref.element_bytes;
  if (tile_count > 1u) {
    const u64 last_tile = tile_count - 1u;
    if (!checked::mul(last_tile, ref.stride_bytes)) {
      return Reject("compute_resident_stride_invalid");
    }
    const u64 offset = last_tile * ref.stride_bytes;
    if (!checked::add(offset, ref.element_bytes)) {
      return Reject("compute_resident_bytes_invalid");
    }
    extent = offset + ref.element_bytes;
  }
  if (!checked::add(ref.offset_bytes, extent) ||
      ref.bytes < ref.offset_bytes + extent) {
    return Reject("compute_resident_bytes_invalid");
  }
  return Accept();
}

} // namespace rund::kernel
