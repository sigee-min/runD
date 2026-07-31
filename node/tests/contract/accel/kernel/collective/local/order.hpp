#pragma once

#include "hash.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::collective {

template <typename Key>
[[nodiscard]] constexpr Key
SortDomainKey(const Key key, const rund::kernel::u32 key_bits,
              const bool signed_order = false) noexcept {
  constexpr rund::kernel::u32 full_bits =
      static_cast<rund::kernel::u32>(sizeof(Key) * 8u);
  const rund::kernel::u32 bits =
      key_bits == 0u || key_bits >= full_bits ? full_bits : key_bits;
  const Key mask = bits == full_bits ? static_cast<Key>(~Key{0u})
                                     : (Key{1u} << bits) - Key{1u};
  const Key masked = key & mask;
  return signed_order ? masked ^ (Key{1u} << (bits - 1u)) : masked;
}

template <typename Key, std::size_t Count>
[[nodiscard]] bool
StableEqualKeyOrderOk(const std::array<Key, Count> &keys,
                      const std::array<rund::kernel::u32, Count> &values,
                      const rund::kernel::u32 key_bits = 0u,
                      const bool signed_order = false) noexcept {
  for (std::size_t index = 1u; index < keys.size(); ++index) {
    if (SortDomainKey(keys[index - 1u], key_bits, signed_order) ==
            SortDomainKey(keys[index], key_bits, signed_order) &&
        values[index - 1u] > values[index]) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract::collective
