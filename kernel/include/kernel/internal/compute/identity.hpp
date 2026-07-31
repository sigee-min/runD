#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel::internal {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

template <typename Hash>
[[nodiscard]] constexpr Hash MixIdentity(const Hash hash, const u64 value,
                                         const u64 salt) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi +
                  (hash.lo << 6u) + (hash.lo >> 2u));
  return Hash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + salt),
  };
}

} // namespace rund::kernel::internal
