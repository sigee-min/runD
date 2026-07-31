#pragma once

#include <kernel/program/compute/segmented/scan/model.hpp>

namespace rund::kernel {
namespace segmented_scan_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr SegmentedScanHash
Mix(const SegmentedScanHash hash, const u64 value) noexcept {
  const u64 mixed = Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi +
                                (hash.lo << 6u) + (hash.lo >> 2u));
  return SegmentedScanHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x517cc1b727220a95ull),
  };
}

} // namespace segmented_scan_identity_detail

[[nodiscard]] constexpr SegmentedScanHash
HashSegmentedScan(const SegmentedScanDesc &desc) noexcept {
  SegmentedScanHash hash{
      .hi = 0x452821e638d01377ull,
      .lo = 0xbe5466cf34e90c6cull,
  };
  hash = segmented_scan_identity_detail::Mix(hash,
                                                     static_cast<u64>(desc.op));
  hash = segmented_scan_identity_detail::Mix(
      hash, static_cast<u64>(desc.element));
  hash = segmented_scan_identity_detail::Mix(hash, desc.element_count);
  hash = segmented_scan_identity_detail::Mix(hash, desc.block_size);
  return hash;
}

} // namespace rund::kernel
