#pragma once

#include <math32/core/model.hpp>
#include <math32/fixed/constants.hpp>
#include <math32/simd/model.hpp>

#include <bit>

namespace rund::math32::detail {

using I64x2 = i64 __attribute__((vector_size(16)));
using U64x2 = u64 __attribute__((vector_size(16)));

struct I64x4 {
  I64x2 low{};
  I64x2 high{};
};

struct U64x4 {
  U64x2 low{};
  U64x2 high{};
};

using Mask64x4 = U64x4;

static_assert(sizeof(I64x4) == 32u);
static_assert(sizeof(U64x4) == 32u);
static_assert(alignof(I64x4) == 16u);
static_assert(alignof(U64x4) == 16u);

[[nodiscard]] inline I64x4 operator+(const I64x4 lhs,
                                     const I64x4 rhs) noexcept {
  return I64x4{lhs.low + rhs.low, lhs.high + rhs.high};
}

[[nodiscard]] inline I64x4 operator-(const I64x4 lhs,
                                     const I64x4 rhs) noexcept {
  return I64x4{lhs.low - rhs.low, lhs.high - rhs.high};
}

[[nodiscard]] inline I64x4 operator-(const I64x4 value) noexcept {
  return I64x4{-value.low, -value.high};
}

[[nodiscard]] inline I64x4 operator*(const I64x4 lhs,
                                     const I64x4 rhs) noexcept {
  return I64x4{lhs.low * rhs.low, lhs.high * rhs.high};
}

[[nodiscard]] inline I64x4 operator>>(const I64x4 value,
                                     const u32 shift) noexcept {
  return I64x4{value.low >> shift, value.high >> shift};
}

[[nodiscard]] inline U64x4 operator+(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low + rhs.low, lhs.high + rhs.high};
}

[[nodiscard]] inline U64x4 operator-(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low - rhs.low, lhs.high - rhs.high};
}

[[nodiscard]] inline U64x4 operator*(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low * rhs.low, lhs.high * rhs.high};
}

[[nodiscard]] inline U64x4 operator&(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low & rhs.low, lhs.high & rhs.high};
}

[[nodiscard]] inline U64x4 operator|(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low | rhs.low, lhs.high | rhs.high};
}

[[nodiscard]] inline U64x4 operator^(const U64x4 lhs,
                                     const U64x4 rhs) noexcept {
  return U64x4{lhs.low ^ rhs.low, lhs.high ^ rhs.high};
}

[[nodiscard]] inline U64x4 operator~(const U64x4 value) noexcept {
  return U64x4{~value.low, ~value.high};
}

[[nodiscard]] inline U64x4 operator<<(const U64x4 value,
                                     const u32 shift) noexcept {
  return U64x4{value.low << shift, value.high << shift};
}

[[nodiscard]] inline U64x4 operator<<(const U64x4 value,
                                     const U64x4 shift) noexcept {
  return U64x4{value.low << shift.low, value.high << shift.high};
}

[[nodiscard]] inline U64x4 operator>>(const U64x4 value,
                                     const u32 shift) noexcept {
  return U64x4{value.low >> shift, value.high >> shift};
}

[[nodiscard]] inline U64x4 operator>>(const U64x4 value,
                                     const U64x4 shift) noexcept {
  return U64x4{value.low >> shift.low, value.high >> shift.high};
}

inline U64x4 &operator>>=(U64x4 &value, const u32 shift) noexcept {
  value = value >> shift;
  return value;
}

[[nodiscard]] inline I64x4 SplatI64x4(const i64 value) noexcept {
  const I64x2 half{value, value};
  return I64x4{half, half};
}

[[nodiscard]] inline U64x4 SplatU64x4(const u64 value) noexcept {
  const U64x2 half{value, value};
  return U64x4{half, half};
}

[[nodiscard]] inline I64x4 Widen64(const simd::I32x value) noexcept {
  return I64x4{
      I64x2{static_cast<i64>(value[0]), static_cast<i64>(value[1])},
      I64x2{static_cast<i64>(value[2]), static_cast<i64>(value[3])},
  };
}

[[nodiscard]] inline U64x4 Widen64(const simd::U32x value) noexcept {
  return U64x4{
      U64x2{static_cast<u64>(value[0]), static_cast<u64>(value[1])},
      U64x2{static_cast<u64>(value[2]), static_cast<u64>(value[3])},
  };
}

[[nodiscard]] inline I64x4 Signed64(const U64x4 value) noexcept {
  return std::bit_cast<I64x4>(value);
}

[[nodiscard]] inline U64x4 Unsigned64(const I64x4 value) noexcept {
  return std::bit_cast<U64x4>(value);
}

[[nodiscard]] inline simd::I32x Narrow32(const I64x4 value) noexcept {
  return simd::I32x{
      static_cast<i32>(value.low[0]),
      static_cast<i32>(value.low[1]),
      static_cast<i32>(value.high[0]),
      static_cast<i32>(value.high[1]),
  };
}

[[nodiscard]] inline simd::U32x Narrow32(const U64x4 value) noexcept {
  return simd::U32x{
      static_cast<u32>(value.low[0]),
      static_cast<u32>(value.low[1]),
      static_cast<u32>(value.high[0]),
      static_cast<u32>(value.high[1]),
  };
}

[[nodiscard]] inline simd::Mask32x NarrowMask(const Mask64x4 value) noexcept {
  return Narrow32(value);
}

[[nodiscard]] inline Mask64x4 Mask64(const I64x4 value) noexcept {
  return Unsigned64(value);
}

[[nodiscard]] inline U64x4 Select64(const Mask64x4 mask, const U64x4 a,
                                    const U64x4 b) noexcept {
  return (mask & a) | (~mask & b);
}

[[nodiscard]] inline I64x4 Select64(const Mask64x4 mask, const I64x4 a,
                                    const I64x4 b) noexcept {
  return Signed64(Select64(mask, Unsigned64(a), Unsigned64(b)));
}

[[nodiscard]] inline Mask64x4 Eq64(const U64x4 lhs, const U64x4 rhs) noexcept {
  return Mask64(I64x4{
      static_cast<I64x2>(lhs.low == rhs.low),
      static_cast<I64x2>(lhs.high == rhs.high),
  });
}

[[nodiscard]] inline Mask64x4 Ge64(const U64x4 lhs, const U64x4 rhs) noexcept {
  return Mask64(I64x4{
      static_cast<I64x2>(lhs.low >= rhs.low),
      static_cast<I64x2>(lhs.high >= rhs.high),
  });
}

[[nodiscard]] inline Mask64x4 Gt64(const U64x4 lhs, const U64x4 rhs) noexcept {
  return Mask64(I64x4{
      static_cast<I64x2>(lhs.low > rhs.low),
      static_cast<I64x2>(lhs.high > rhs.high),
  });
}

[[nodiscard]] inline Mask64x4 Lt64(const I64x4 lhs, const I64x4 rhs) noexcept {
  return Mask64(I64x4{
      static_cast<I64x2>(lhs.low < rhs.low),
      static_cast<I64x2>(lhs.high < rhs.high),
  });
}

[[nodiscard]] inline Mask64x4 Gt64(const I64x4 lhs, const I64x4 rhs) noexcept {
  return Mask64(I64x4{
      static_cast<I64x2>(lhs.low > rhs.low),
      static_cast<I64x2>(lhs.high > rhs.high),
  });
}

[[nodiscard]] inline U64x4 Min64(const U64x4 lhs, const U64x4 rhs) noexcept {
  return Select64(Ge64(lhs, rhs), rhs, lhs);
}

[[nodiscard]] inline simd::I32x ClampWideToI32x(const I64x4 value) noexcept {
  const I64x4 low = Select64(Lt64(value, SplatI64x4(FixedMin)),
                             SplatI64x4(FixedMin), value);
  const I64x4 clamped = Select64(Gt64(low, SplatI64x4(FixedMax)),
                                 SplatI64x4(FixedMax), low);
  return Narrow32(clamped);
}

[[nodiscard]] inline simd::I32x ClampU64x4ToI32x(const U64x4 value) noexcept {
  const U64x4 clamped = Min64(
      value, SplatU64x4(static_cast<u64>(static_cast<u32>(FixedMax))));
  return std::bit_cast<simd::I32x>(Narrow32(clamped));
}

[[nodiscard]] inline U64x4 AbsMagnitudeWide(const I64x4 value) noexcept {
  return Unsigned64(
      Select64(Lt64(value, SplatI64x4(0)), -value, value));
}

[[nodiscard]] inline U64x4 AbsMagnitudeWide(const simd::I32x value) noexcept {
  return AbsMagnitudeWide(Widen64(value));
}

[[nodiscard]] inline U64x4
UnsignedDiv64(U64x4 numerator, const U64x4 denominator) noexcept {
  const U64x4 safe_denominator =
      Select64(Eq64(denominator, SplatU64x4(0)), SplatU64x4(1),
               denominator);
  U64x4 quotient = SplatU64x4(0);
  U64x4 remainder = SplatU64x4(0);
  for (int bit = 62; bit >= 0; --bit) {
    remainder =
        (remainder << 1u) | ((numerator >> bit) & SplatU64x4(1));
    const Mask64x4 ge = Ge64(remainder, safe_denominator);
    remainder = Select64(ge, remainder - safe_denominator, remainder);
    quotient = quotient |
               Select64(ge, SplatU64x4(u64{1} << bit), SplatU64x4(0));
  }
  return quotient;
}

[[nodiscard]] inline simd::I32x
ClampSignedMagnitude64(const Mask64x4 negative,
                       const U64x4 magnitude) noexcept {
  const U64x4 pos_mag = Min64(
      magnitude, SplatU64x4(static_cast<u64>(static_cast<u32>(FixedMax))));
  const U64x4 neg_mag = Min64(magnitude, SplatU64x4(u64{1} << 31u));
  const I64x4 pos = Signed64(pos_mag);
  const I64x4 neg = -Signed64(neg_mag);
  return Narrow32(Select64(negative, neg, pos));
}

}  // namespace rund::math32::detail
