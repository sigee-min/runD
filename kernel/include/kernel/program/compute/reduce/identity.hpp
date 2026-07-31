#pragma once

#include <kernel/program/compute/reduce/model.hpp>

namespace rund::kernel {
namespace reduce_identity_detail {

[[nodiscard]] constexpr u64 Avalanche64(u64 value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

[[nodiscard]] constexpr ReduceHash Mix(
    const ReduceHash hash,
    const u64 value) noexcept {
  const u64 mixed =
      Avalanche64(value + 0x9e3779b97f4a7c15ull + hash.hi + (hash.lo << 6u) +
                  (hash.lo >> 2u));
  return ReduceHash{
      .hi = Avalanche64(hash.hi ^ mixed),
      .lo = Avalanche64(hash.lo + mixed + 0x517cc1b727220a95ull),
  };
}

}  // namespace reduce_identity_detail

[[nodiscard]] constexpr ReduceHash HashReduce(
    const ReduceDesc& desc) noexcept {
  ReduceHash hash{
      .hi = 0x452821e638d01377ull,
      .lo = 0xbe5466cf34e90c6cull,
  };
  hash = reduce_identity_detail::Mix(hash, static_cast<u64>(desc.op));
  hash = reduce_identity_detail::Mix(hash, static_cast<u64>(desc.element));
  hash = reduce_identity_detail::Mix(hash, desc.element_count);
  hash = reduce_identity_detail::Mix(hash, desc.block_size);
  hash = reduce_identity_detail::Mix(
      hash, static_cast<u64>(desc.count_source));
  hash = reduce_identity_detail::Mix(hash, kReduceItemsPerThread);
  hash = reduce_identity_detail::Mix(hash, kReduceFirstPassMaxGroups);
  hash = reduce_identity_detail::Mix(hash, kReduceWideElementBytes);
  hash = reduce_identity_detail::Mix(hash, kReduceNarrowChunkItems);
  return hash;
}

}  // namespace rund::kernel
