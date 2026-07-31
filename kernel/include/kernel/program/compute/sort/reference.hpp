#pragma once

#include <kernel/program/compute/sort/model.hpp>

namespace rund::kernel {
namespace sort_reference_detail {

[[nodiscard]] constexpr SortResult Reject(
    const u64 element_count,
    const char* const reason) noexcept {
  return SortResult{
      .element_count = element_count,
      .reason = reason,
  };
}

template <typename Key>
inline void MoveRightByOne(Key* const output_keys,
                           u32* const output_values,
                           u64* const output_original_indices,
                           const u64 index) noexcept {
  output_keys[index] = output_keys[index - 1u];
  output_values[index] = output_values[index - 1u];
  output_original_indices[index] = output_original_indices[index - 1u];
}

template <typename Key>
inline void InsertStable(Key* const output_keys,
                         u32* const output_values,
                         u64* const output_original_indices,
                         const u64 sorted_count,
                         const Key key,
                         const u32 value,
                         const u64 input_index) noexcept {
  u64 insert = sorted_count;
  while (insert > 0u && key < output_keys[insert - 1u]) {
    MoveRightByOne(output_keys, output_values, output_original_indices,
                   insert);
    --insert;
  }
  output_keys[insert] = key;
  output_values[insert] = value;
  output_original_indices[insert] = input_index;
}

template <typename Key>
[[nodiscard]] inline SortResult ReferenceStableSort(
    const Key* const keys,
    const u32* const values,
    Key* const output_keys,
    u32* const output_values,
    u64* const output_original_indices,
    const u64 element_count) noexcept {
  if (element_count == 0u) {
    return Reject(0u, "compute_sort_count_zero");
  }
  if (keys == nullptr || values == nullptr || output_keys == nullptr ||
      output_values == nullptr || output_original_indices == nullptr) {
    return Reject(element_count, "compute_sort_buffer_invalid");
  }

  u64 sorted_count = 0u;
  for (u64 input_index = 0u; input_index < element_count; ++input_index) {
    InsertStable(output_keys, output_values, output_original_indices,
                 sorted_count, keys[input_index], values[input_index],
                 input_index);
    ++sorted_count;
  }

  return SortResult{
      .element_count = element_count,
      .ok = true,
      .reason = "ok",
  };
}

}  // namespace sort_reference_detail

[[nodiscard]] inline SortResult ReferenceStableSortU32(
    const u32* const keys,
    const u32* const values,
    u32* const output_keys,
    u32* const output_values,
    u64* const output_original_indices,
    const u64 element_count) noexcept {
  return sort_reference_detail::ReferenceStableSort(
      keys, values, output_keys, output_values, output_original_indices,
      element_count);
}

[[nodiscard]] inline SortResult ReferenceStableSortU64(
    const u64* const keys,
    const u32* const values,
    u64* const output_keys,
    u32* const output_values,
    u64* const output_original_indices,
    const u64 element_count) noexcept {
  return sort_reference_detail::ReferenceStableSort(
      keys, values, output_keys, output_values, output_original_indices,
      element_count);
}

}  // namespace rund::kernel
