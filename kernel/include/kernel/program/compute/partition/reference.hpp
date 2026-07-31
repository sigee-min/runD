#pragma once

#include <kernel/program/compute/partition/model.hpp>

namespace rund::kernel {
namespace partition_reference_detail {

[[nodiscard]] constexpr PartitionResult
Reject(const u64 element_count, const u64 false_count, const u64 true_count,
       const char *const reason) noexcept {
  return PartitionResult{
      .element_count = element_count,
      .false_count = false_count,
      .true_count = true_count,
      .reason = reason,
  };
}

template <class F>
[[nodiscard]] inline u64 CountFalseGroup(const F *const flags,
                                         const u64 count) noexcept {
  u64 false_count = 0u;
  for (u64 index = 0u; index < count; ++index) {
    false_count += flags[index] == 0u ? 1u : 0u;
  }
  return false_count;
}

template <class F, class T>
inline void WriteStableGroups(const F *const flags, const T *const values,
                              const u64 count, const u64 false_count,
                              T *const output) noexcept {
  u64 false_index = 0u;
  for (u64 index = 0u; index < count; ++index) {
    const bool false_group = flags[index] == 0u;
    const u64 true_rank = index - false_index;
    output[false_group ? false_index : false_count + true_rank] = values[index];
    false_index += false_group ? 1u : 0u;
  }
}

template <class F, class T>
[[nodiscard]] inline PartitionResult ReferenceStablePartition(
    const F *const flags, const T *const values, const u64 count,
    T *const output, u64 *const false_count, u64 *const true_count) noexcept {
  if (false_count != nullptr) {
    *false_count = 0u;
  }
  if (true_count != nullptr) {
    *true_count = 0u;
  }
  if (count == 0u) {
    return Reject(0u, 0u, 0u, "compute_partition_count_zero");
  }
  if (flags == nullptr || values == nullptr || output == nullptr ||
      false_count == nullptr || true_count == nullptr) {
    return Reject(count, 0u, 0u, "compute_partition_buffer_invalid");
  }
  const u64 false_index = CountFalseGroup(flags, count);
  WriteStableGroups(flags, values, count, false_index, output);
  *false_count = false_index;
  *true_count = count - false_index;
  return PartitionResult{count, *false_count, *true_count, true, "ok"};
}

} // namespace partition_reference_detail

[[nodiscard]] inline PartitionResult ReferenceStablePartitionU32(
    const u32 *const flags, const u32 *const values, const u64 count,
    u32 *const output, u64 *const false_count, u64 *const true_count) noexcept {
  return partition_reference_detail::ReferenceStablePartition(
      flags, values, count, output, false_count, true_count);
}

[[nodiscard]] inline PartitionResult ReferenceStablePartitionU64(
    const u32 *const flags, const u64 *const values, const u64 count,
    u64 *const output, u64 *const false_count, u64 *const true_count) noexcept {
  return partition_reference_detail::ReferenceStablePartition(
      flags, values, count, output, false_count, true_count);
}

[[nodiscard]] inline PartitionResult ReferenceStablePartitionFlagsU64ValuesU32(
    const u64 *const flags, const u32 *const values, const u64 count,
    u32 *const output, u64 *const false_count, u64 *const true_count) noexcept {
  return partition_reference_detail::ReferenceStablePartition(
      flags, values, count, output, false_count, true_count);
}

[[nodiscard]] inline PartitionResult ReferenceStablePartitionFlagsU64ValuesU64(
    const u64 *const flags, const u64 *const values, const u64 count,
    u64 *const output, u64 *const false_count, u64 *const true_count) noexcept {
  return partition_reference_detail::ReferenceStablePartition(
      flags, values, count, output, false_count, true_count);
}

} // namespace rund::kernel
