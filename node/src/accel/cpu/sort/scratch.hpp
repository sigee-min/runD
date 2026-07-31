#pragma once

#include <kernel/core/model.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace rund::node::accel::detail {

struct CpuSortScratch {
  std::vector<rund::kernel::u32> keys32{};
  std::vector<rund::kernel::u64> keys64{};
  std::vector<rund::kernel::u32> values{};
  std::array<rund::kernel::u64, 256u> counts{};
  std::array<rund::kernel::u64, 256u> offsets{};
};

template <typename Key>
[[nodiscard]] Key* EnsureSortKeys(CpuSortScratch& scratch,
                                  const std::size_t count) {
  if constexpr (std::is_same_v<Key, rund::kernel::u32>) {
    if (scratch.keys32.size() < count) { scratch.keys32.resize(count); }
    return scratch.keys32.data();
  } else {
    static_assert(std::is_same_v<Key, rund::kernel::u64>);
    if (scratch.keys64.size() < count) { scratch.keys64.resize(count); }
    return scratch.keys64.data();
  }
}

[[nodiscard]] inline rund::kernel::u32* EnsureSortValues(
    CpuSortScratch& scratch,
    const std::size_t count) {
  if (scratch.values.size() < count) { scratch.values.resize(count); }
  return scratch.values.data();
}

}  // namespace rund::node::accel::detail
