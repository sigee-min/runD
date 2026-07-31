#pragma once

#include <kernel/program/compute/compact/model.hpp>

namespace rund::kernel {
namespace compact_reference_detail {

[[nodiscard]] constexpr CompactResult Reject(
    const u64 element_count,
    const u64 output_count,
    const char* const reason) noexcept {
  return CompactResult{
      .element_count = element_count,
      .output_count = output_count,
      .reason = reason,
  };
}

}  // namespace compact_reference_detail

[[nodiscard]] inline CompactResult ReferenceCompactIdsU32(
    const u32* const flags,
    const u64 count,
    const u64 capacity,
    u32* const output,
    u64* const output_count) noexcept {
  if (count == 0u) {
    if (output_count != nullptr) {
      *output_count = 0u;
    }
    return compact_reference_detail::Reject(
        0u, 0u, "compute_compact_count_zero");
  }
  if (capacity == 0u) {
    if (output_count != nullptr) {
      *output_count = 0u;
    }
    return compact_reference_detail::Reject(
        count, 0u, "compute_compact_capacity_zero");
  }
  if (flags == nullptr || output == nullptr || output_count == nullptr) {
    return compact_reference_detail::Reject(
        count, 0u, "compute_compact_buffer_invalid");
  }

  u64 selected_count = 0u;
  for (u64 index = 0u; index < count; ++index) {
    if (flags[index] == 0u) {
      continue;
    }
    if (index > static_cast<u64>(~u32{0u})) {
      *output_count = selected_count;
      return compact_reference_detail::Reject(
          count, selected_count, "compute_compact_temp_overflow");
    }
    if (selected_count >= capacity) {
      *output_count = selected_count;
      return compact_reference_detail::Reject(
          count, selected_count, "compute_compact_capacity_insufficient");
    }
    output[selected_count] = static_cast<u32>(index);
    ++selected_count;
  }

  *output_count = selected_count;
  return CompactResult{
      .element_count = count,
      .output_count = selected_count,
      .ok = true,
      .reason = "ok",
  };
}

}  // namespace rund::kernel
