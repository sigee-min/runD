#pragma once

#include <math32/simd/select.hpp>
#include <math32/simd/model.hpp>

namespace rund::math32::simd {

inline Mask32x MaskTrue() noexcept { return SplatU32(MaskTrueLane); }
inline Mask32x MaskFalse() noexcept { return SplatU32(MaskFalseLane); }
inline u32 MaskLaneFromBool(const bool value) noexcept {
  return u32{0} - static_cast<u32>(value);
}

inline Mask32x MaskFromBools(const bool a,
                             const bool b,
                             const bool c,
                             const bool d) noexcept {
  return Mask32x{MaskLaneFromBool(a), MaskLaneFromBool(b), MaskLaneFromBool(c), MaskLaneFromBool(d)};
}

[[nodiscard]] inline Mask32x Not(const Mask32x mask) noexcept { return ~mask; }
[[nodiscard]] inline Mask32x And(const Mask32x lhs, const Mask32x rhs) noexcept { return lhs & rhs; }
[[nodiscard]] inline Mask32x Or(const Mask32x lhs, const Mask32x rhs) noexcept { return lhs | rhs; }
[[nodiscard]] inline Mask32x Xor(const Mask32x lhs, const Mask32x rhs) noexcept { return lhs ^ rhs; }

inline bool All(const Mask32x mask) noexcept {
  const Mask32x folded2 = mask & __builtin_shufflevector(mask, mask, 2, 3, 0, 1);
  const Mask32x folded4 = folded2 & __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2);
  return folded4[0] == MaskTrueLane;
}

inline bool Any(const Mask32x mask) noexcept {
  const Mask32x folded2 = mask | __builtin_shufflevector(mask, mask, 2, 3, 0, 1);
  const Mask32x folded4 = folded2 | __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2);
  return folded4[0] != MaskFalseLane;
}

}  // namespace rund::math32::simd
