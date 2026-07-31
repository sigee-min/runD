#pragma once

#include <kernel/program/compute/gather/plan.hpp>

namespace rund::kernel {
namespace gather_reference_detail {

[[nodiscard]] constexpr GatherResult Reject(
    const u64 element_count,
    const u64 first_invalid_index,
    const char* const reason) noexcept {
  return GatherResult{
      .element_count = element_count,
      .first_invalid_index = first_invalid_index,
      .reason = reason,
  };
}

template <typename Element>
[[nodiscard]] inline GatherResult ReferenceGather(
    const Element* const values,
    const u32* const indices,
    Element* const output,
    const u64 element_count,
    const u64 source_count) noexcept {
  if (element_count == 0u) {
    return Reject(0u, 0u, "compute_gather_count_zero");
  }
  if (source_count == 0u) {
    return Reject(element_count, 0u, "compute_gather_source_zero");
  }
  if (source_count > gather_plan_detail::kMaxU32AddressableCount) {
    return Reject(element_count, 0u, "compute_gather_source_unsupported");
  }
  if (values == nullptr || indices == nullptr || output == nullptr) {
    return Reject(element_count, 0u, "compute_gather_buffer_invalid");
  }

  for (u64 index = 0u; index < element_count; ++index) {
    const u32 source_index = indices[index];
    if (static_cast<u64>(source_index) >= source_count) {
      return Reject(element_count, index, "compute_gather_index_out_of_range");
    }
  }
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 source_index = indices[index];
    output[index] = values[source_index];
  }

  return GatherResult{
      .element_count = element_count,
      .first_invalid_index = element_count,
      .ok = true,
      .reason = "ok",
  };
}

}  // namespace gather_reference_detail

[[nodiscard]] inline GatherResult ReferenceGatherU32(
    const u32* const values,
    const u32* const indices,
    u32* const output,
    const u64 element_count,
    const u64 source_count) noexcept {
  return gather_reference_detail::ReferenceGather(
      values, indices, output, element_count, source_count);
}

[[nodiscard]] inline GatherResult ReferenceGatherU64(
    const u64* const values,
    const u32* const indices,
    u64* const output,
    const u64 element_count,
    const u64 source_count) noexcept {
  return gather_reference_detail::ReferenceGather(
      values, indices, output, element_count, source_count);
}

}  // namespace rund::kernel
