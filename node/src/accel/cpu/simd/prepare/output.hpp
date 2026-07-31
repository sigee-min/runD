#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] const char *
ValidateOutput(const BindingPlan &plan,
               const rund::kernel::BindingSet &bindings,
               const u64 scalar_bytes) {
  if (plan.write_count == 0u) {
    return "cpu_simd_write_missing";
  }
  if (!bindings.has_staged_outputs()) {
    return bindings.has_resident_output()
               ? "cpu_simd_resident_output_unsupported"
               : "cpu_simd_staged_output_missing";
  }
  if (bindings.output_buffer_count != 0u) {
    if (bindings.output_buffers == nullptr ||
        bindings.output_buffer_count != plan.write_count) {
      return "cpu_simd_output_invalid";
    }
    u64 total_bytes = 0u;
    for (u64 index = 0u; index < bindings.output_buffer_count; ++index) {
      const rund::kernel::OutputSpan &span = bindings.output_buffers[index];
      const bool width_ok =
          span.element_bytes == scalar_bytes ||
          (plan.write_count == 1u &&
           ((scalar_bytes == sizeof(u64) &&
             span.element_bytes == sizeof(rund::kernel::u32)) ||
            (scalar_bytes == sizeof(rund::kernel::u32) &&
             span.element_bytes == sizeof(u64))));
      if (!width_ok || span.data == nullptr ||
          span.stride_bytes < span.element_bytes ||
          !FitsSize(span.stride_bytes) || span.count < bindings.tile_count) {
        return "cpu_simd_output_invalid";
      }
      u64 offset = 0u;
      if (!CheckedOffset(bindings.tile_count - 1u, span.stride_bytes,
                         span.element_bytes, offset) ||
          !FitsSize(offset)) {
        return "cpu_simd_output_bounds_overflow";
      }
      total_bytes += span.element_bytes;
    }
    return total_bytes == bindings.output_bytes_per_tile
               ? nullptr
               : "cpu_simd_output_invalid";
  }
  if (plan.write_count != 1u) {
    return "cpu_simd_output_invalid";
  }
  const u64 output_bytes = bindings.output_bytes_per_tile;
  const bool width_ok =
      output_bytes == scalar_bytes ||
      (scalar_bytes == sizeof(u64) &&
       output_bytes == sizeof(rund::kernel::u32)) ||
      (scalar_bytes == sizeof(rund::kernel::u32) &&
       output_bytes == sizeof(u64));
  if (!width_ok || bindings.staged_output_stride < output_bytes ||
      !FitsSize(bindings.staged_output_stride) ||
      bindings.staged_output_count < bindings.tile_count) {
    return "cpu_simd_output_invalid";
  }
  u64 offset = 0u;
  if (!CheckedOffset(bindings.tile_count - 1u, bindings.staged_output_stride,
                     output_bytes, offset) || !FitsSize(offset)) {
    return "cpu_simd_output_bounds_overflow";
  }
  return nullptr;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
