#pragma once

#include <math64/core/model.hpp>

#include <cstddef>

#if !defined(__clang__) && !defined(__GNUC__)
#error "runD math64 vector-first math requires compiler vector extensions"
#endif

namespace rund::math64::simd {

inline constexpr std::size_t LaneCount = 2u;

using I64x = i64 __attribute__((vector_size(16)));
using U64x = u64 __attribute__((vector_size(16)));
using Mask64x = U64x;

inline constexpr u64 MaskTrueLane = 0xffffffffffffffffull;
inline constexpr u64 MaskFalseLane = 0ull;

static_assert(sizeof(I64x) == 16u);
static_assert(sizeof(U64x) == 16u);
static_assert(sizeof(Mask64x) == 16u);

inline I64x SplatI64(const i64 value) noexcept { return I64x{value, value}; }

inline U64x SplatU64(const u64 value) noexcept { return U64x{value, value}; }

} // namespace rund::math64::simd
