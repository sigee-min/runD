#pragma once

#include <kernel/program/compute/scan/model.hpp>

namespace rund::kernel {
namespace scan_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr ScanHash Mix(
    const ScanHash hash,
    const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return ScanHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x517cc1b727220a95ull),
  };
}

}  // namespace scan_identity_detail

[[nodiscard]] constexpr ScanHash HashScan(
    const ScanDesc& desc) noexcept {
  ScanHash hash{
      .hi = 0x243f6a8885a308d3ull,
      .lo = 0x13198a2e03707344ull,
  };
  hash = scan_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = scan_identity_detail::Mix(hash, static_cast<u64>(desc.element));
  hash = scan_identity_detail::Mix(hash, desc.element_count);
  hash = scan_identity_detail::Mix(hash, desc.block_size);
  hash = scan_identity_detail::Mix(
      hash, static_cast<u64>(desc.count_source));
  return hash;
}

}  // namespace rund::kernel
