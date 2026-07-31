#pragma once

#include <kernel/program/compute/sort/model.hpp>

namespace rund::kernel {
namespace sort_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr SortHash Mix(
    const SortHash hash,
    const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return SortHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x517cc1b727220a95ull),
  };
}

}  // namespace sort_identity_detail

[[nodiscard]] constexpr SortHash HashSort(
    const SortDesc& desc) noexcept {
  SortHash hash{
      .hi = 0xa4093822299f31d0ull,
      .lo = 0x082efa98ec4e6c89ull,
  };
  hash = sort_identity_detail::Mix(hash, static_cast<u64>(desc.key));
  hash = sort_identity_detail::Mix(hash, static_cast<u64>(desc.value));
  hash = sort_identity_detail::Mix(hash, desc.element_count);
  hash = sort_identity_detail::Mix(hash, static_cast<u64>(desc.radix_bits));
  hash = sort_identity_detail::Mix(hash, static_cast<u64>(desc.key_bits));
  hash = sort_identity_detail::Mix(hash, desc.stable ? 1u : 0u);
  hash = sort_identity_detail::Mix(hash,
                                   static_cast<u64>(desc.count_source));
  return hash;
}

}  // namespace rund::kernel
