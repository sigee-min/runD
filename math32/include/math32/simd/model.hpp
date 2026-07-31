#pragma once

#include <math32/core/model.hpp>

#include <cstddef>

#if !defined(__clang__) && !defined(__GNUC__)
#error "runD math32 vector-first math requires compiler vector extensions"
#endif

namespace rund::math32::simd {

inline constexpr std::size_t LaneCount = 4u;

using I32x = i32 __attribute__((vector_size(16)));
using U32x = u32 __attribute__((vector_size(16)));
using Mask32x = U32x;

inline constexpr u32 MaskTrueLane = 0xffffffffu;
inline constexpr u32 MaskFalseLane = 0u;

static_assert(sizeof(I32x) == 16u);
static_assert(sizeof(U32x) == 16u);
static_assert(sizeof(Mask32x) == 16u);

inline I32x SplatI32(const i32 value) noexcept {
  return I32x{value, value, value, value};
}

inline U32x SplatU32(const u32 value) noexcept {
  return U32x{value, value, value, value};
}

} // namespace rund::math32::simd
