#pragma once

#include "../../../../backend/number.hpp"

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
ResidentRefFitsGeneratedU32(const rund::kernel::ResidentBufferRef &ref,
                            const rund::kernel::u64 tile_count) noexcept {
  constexpr rund::kernel::u64 max_generated_byte_offset =
      static_cast<rund::kernel::u64>(std::numeric_limits<std::uint32_t>::max());
  if (ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
      tile_count == 0u ||
      !rund::kernel::checked::mul(tile_count - 1u, ref.stride_bytes)) {
    return false;
  }
  const rund::kernel::u64 tail = (tile_count - 1u) * ref.stride_bytes;
  return tail <= max_generated_byte_offset &&
         ref.element_bytes <= max_generated_byte_offset - tail;
}

[[nodiscard]] inline bool ResidentFullRangeWindowFitsGeneratedU32(
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings) noexcept {
  constexpr rund::kernel::u64 max_generated_dispatch_tiles =
      static_cast<rund::kernel::u64>(std::numeric_limits<std::uint32_t>::max());
  if (window.tile_count > max_generated_dispatch_tiles ||
      plan.input_buffer_count > bindings.resident_inputs.count ||
      (plan.input_buffer_count != 0u && !bindings.resident_inputs.has_refs()) ||
      !bindings.resident_outputs.has_refs() || plan.output_buffer_count == 0u ||
      bindings.resident_outputs.count != plan.output_buffer_count) {
    return false;
  }
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    const rund::kernel::ResidentBufferRef *const resident =
        bindings.resident_inputs.ref(index);
    if (resident == nullptr ||
        !ResidentRefFitsGeneratedU32(*resident, window.tile_count)) {
      return false;
    }
  }
  for (rund::kernel::u64 index = 0u; index < plan.output_buffer_count;
       ++index) {
    const rund::kernel::ResidentBufferRef *const resident =
        bindings.resident_outputs.ref(index);
    if (resident == nullptr ||
        !ResidentRefFitsGeneratedU32(*resident, window.tile_count)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
