#pragma once

#include "../../local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace node_accel_contract::collective::sort_run {

template <typename Key, std::size_t Count> struct Reference {
  std::array<rund::kernel::u32, Count> input_values{};
  std::array<Key, Count> keys{};
  std::array<rund::kernel::u32, Count> values{};
  std::array<rund::kernel::u64, Count> indices{};
  bool ok = false;
};

template <typename Key, std::size_t Count>
[[nodiscard]] Reference<Key, Count>
BuildReference(const std::array<Key, Count> &input_keys,
               const rund::kernel::u32 key_bits = 0u,
               const bool signed_order = false) {
  Reference<Key, Count> ref{};
  for (std::size_t index = 0u; index < ref.indices.size(); ++index) {
    ref.indices[index] = static_cast<rund::kernel::u64>(index);
    ref.input_values[index] = static_cast<rund::kernel::u32>(index);
  }
  std::stable_sort(
      ref.indices.begin(), ref.indices.end(),
      [&](const rund::kernel::u64 lhs, const rund::kernel::u64 rhs) {
        return SortDomainKey(input_keys[static_cast<std::size_t>(lhs)],
                             key_bits, signed_order) <
               SortDomainKey(input_keys[static_cast<std::size_t>(rhs)],
                             key_bits, signed_order);
      });
  for (std::size_t out = 0u; out < ref.indices.size(); ++out) {
    const std::size_t input = static_cast<std::size_t>(ref.indices[out]);
    ref.keys[out] = input_keys[input];
    ref.values[out] = ref.input_values[input];
  }
  ref.ok = StableEqualKeyOrderOk(ref.keys, ref.values, key_bits, signed_order);
  return ref;
}

} // namespace node_accel_contract::collective::sort_run
