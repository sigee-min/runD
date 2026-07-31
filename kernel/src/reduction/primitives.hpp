#pragma once

#include <kernel/core/model.hpp>

#include <limits>

namespace rund::kernel::reduction_primitives {

constexpr u64 kFixedBinaryTreeOddPadding = 0xA11CE5u;

inline u64 MixHash(u64 seed, const u64 value) {
  seed ^= value + 0x9E3779B97F4A7C15ull + (seed << 6u) + (seed >> 2u);
  return seed;
}

inline u64 CombineHashes(const u64 left, const u64 right) {
  u64 seed = 0xC6A4A7935BD1E995ull;
  seed = MixHash(seed, left);
  seed = MixHash(seed, right);
  return seed;
}

inline u64 SaturatingAdd(const u64 left, const u64 right) {
  const u64 max_value = std::numeric_limits<u64>::max();
  return left > max_value - right ? max_value : left + right;
}

} // namespace rund::kernel::reduction_primitives
