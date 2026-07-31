#pragma once

#include <math64/simd/select.hpp>
#include <math64/simd/model.hpp>

namespace rund::math64::simd {

inline Mask64x MaskTrue() noexcept { return SplatU64(MaskTrueLane); }
inline Mask64x MaskFalse() noexcept { return SplatU64(MaskFalseLane); }
inline u64 MaskLaneFromBool(const bool value) noexcept {
  return u64{0} - static_cast<u64>(value);
}

inline Mask64x MaskFromBools(const bool a, const bool b) noexcept {
  return Mask64x{MaskLaneFromBool(a), MaskLaneFromBool(b)};
}

[[nodiscard]] inline Mask64x Not(const Mask64x mask) noexcept { return ~mask; }
[[nodiscard]] inline Mask64x And(const Mask64x lhs, const Mask64x rhs) noexcept { return lhs & rhs; }
[[nodiscard]] inline Mask64x Or(const Mask64x lhs, const Mask64x rhs) noexcept { return lhs | rhs; }
[[nodiscard]] inline Mask64x Xor(const Mask64x lhs, const Mask64x rhs) noexcept { return lhs ^ rhs; }

inline bool All(const Mask64x mask) noexcept {
  const Mask64x folded = mask & __builtin_shufflevector(mask, mask, 1, 0);
  return folded[0] == MaskTrueLane;
}

inline bool Any(const Mask64x mask) noexcept {
  const Mask64x folded = mask | __builtin_shufflevector(mask, mask, 1, 0);
  return folded[0] != MaskFalseLane;
}

}  // namespace rund::math64::simd
