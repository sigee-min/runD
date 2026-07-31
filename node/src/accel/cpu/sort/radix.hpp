#pragma once

#include <accel/check.hpp>

#include "scratch.hpp"

#include <kernel/core/model.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

template <typename Key>
[[nodiscard]] inline std::uint32_t
SortBucket(const Key key, const rund::kernel::u32 pass,
           const rund::kernel::u32 pass_count,
           const bool signed_order) noexcept {
  std::uint32_t bucket =
      static_cast<std::uint32_t>((key >> (pass * 8u)) & Key{0xffu});
  if (signed_order && pass + 1u == pass_count) {
    bucket ^= 0x80u;
  }
  return bucket;
}

template <typename Key>
[[nodiscard]] rund::AccelCheck ExecuteCpuRadixSortPrepared(
    const Key *const keys, const rund::kernel::u32 *const values,
    Key *const output_keys, rund::kernel::u32 *const output_values,
    const rund::kernel::u64 element_count, const rund::kernel::u32 pass_count,
    const bool identity_values, const bool signed_order, Key *const temp_keys,
    rund::kernel::u32 *const temp_values,
    std::array<rund::kernel::u64, 256u> &counts,
    std::array<rund::kernel::u64, 256u> &offsets) {
  if (element_count > static_cast<rund::kernel::u64>(
                          std::numeric_limits<std::size_t>::max()) ||
      pass_count == 0u || (pass_count % 2u) != 0u || temp_keys == nullptr ||
      temp_values == nullptr) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  if (identity_values &&
      element_count > static_cast<rund::kernel::u64>(
                          std::numeric_limits<rund::kernel::u32>::max()) +
                          1u) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }

  const std::size_t count = static_cast<std::size_t>(element_count);
  const Key *src_keys = keys;
  const rund::kernel::u32 *src_values = values;
  for (rund::kernel::u32 pass = 0u; pass < pass_count; ++pass) {
    counts.fill(0u);
    for (std::size_t index = 0u; index < count; ++index) {
      ++counts[SortBucket(src_keys[index], pass, pass_count, signed_order)];
    }
    rund::kernel::u64 running = 0u;
    for (std::size_t bucket = 0u; bucket < counts.size(); ++bucket) {
      offsets[bucket] = running;
      running += counts[bucket];
    }

    Key *const dst_keys = (pass % 2u) == 0u ? temp_keys : output_keys;
    rund::kernel::u32 *const dst_values =
        (pass % 2u) == 0u ? temp_values : output_values;
    if (pass == 0u && identity_values) {
      for (std::size_t index = 0u; index < count; ++index) {
        const std::uint32_t bucket =
            SortBucket(src_keys[index], pass, pass_count, signed_order);
        const std::size_t out = static_cast<std::size_t>(offsets[bucket]++);
        dst_keys[out] = src_keys[index];
        dst_values[out] = static_cast<rund::kernel::u32>(index);
      }
    } else {
      for (std::size_t index = 0u; index < count; ++index) {
        const std::uint32_t bucket =
            SortBucket(src_keys[index], pass, pass_count, signed_order);
        const std::size_t out = static_cast<std::size_t>(offsets[bucket]++);
        dst_keys[out] = src_keys[index];
        dst_values[out] = src_values[index];
      }
    }
    src_keys = dst_keys;
    src_values = dst_values;
  }
  return rund::AccelCheck{true, "ok"};
}

template <typename Key>
[[nodiscard]] rund::AccelCheck ExecuteCpuRadixSort(
    const Key *const keys, const rund::kernel::u32 *const values,
    Key *const output_keys, rund::kernel::u32 *const output_values,
    const rund::kernel::u64 element_count, const rund::kernel::u32 pass_count,
    const bool identity_values, const bool signed_order,
    CpuSortScratch &scratch) {
  if (element_count >
      static_cast<rund::kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  const std::size_t count = static_cast<std::size_t>(element_count);
  return ExecuteCpuRadixSortPrepared(
      keys, values, output_keys, output_values, element_count, pass_count,
      identity_values, signed_order, EnsureSortKeys<Key>(scratch, count),
      EnsureSortValues(scratch, count), scratch.counts, scratch.offsets);
}

} // namespace rund::node::accel::detail
