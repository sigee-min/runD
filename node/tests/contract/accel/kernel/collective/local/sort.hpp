#pragma once

#include "hash.hpp"

#include <array>
#include <cstdint>

namespace node_accel_contract::collective {

[[nodiscard]] constexpr std::array<rund::kernel::u32, 8u>
SortU32Fixture() noexcept {
  return std::array<rund::kernel::u32, 8u>{7u, 3u, 7u, 0u,
                                           3u, 9u, 3u, 0u};
}

[[nodiscard]] constexpr std::array<rund::kernel::u32, 8u>
SortU32Declared16Fixture() noexcept {
  return std::array<rund::kernel::u32, 8u>{
      0x00020002u, 0x00010001u, 0x00030002u, 0x00000001u,
      0x00040003u, 0x00050000u, 0x00060003u, 0x00070000u};
}

[[nodiscard]] constexpr std::array<rund::kernel::u64, 8u>
SortU64Fixture() noexcept {
  return std::array<rund::kernel::u64, 8u>{
      0x100000000ull, 4u, 0x100000000ull, 2u,
      4u,            9u, 2u,              4u};
}

template <typename Key>
struct SortRunHashes {
  std::uint64_t key_hash = 0u;
  std::uint64_t value_hash = 0u;
  std::uint64_t dispatch_count = 0u;
  std::uint64_t command_submit_count = 0u;
  bool ok = false;
};

}  // namespace node_accel_contract::collective
