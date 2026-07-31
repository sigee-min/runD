#include "local.hpp"

namespace rund::kernel::strict_float {
namespace {

constexpr u32 kFloat32CanonicalQuietNan = 0x7FC00000u;
constexpr u64 kFloat64CanonicalQuietNan = 0x7FF8000000000000ull;

} // namespace

Format Float32Format() {
  return Format{
      .exponent_bits = 8u,
      .fraction_bits = 23u,
      .sign_mask = 0x80000000ull,
      .exponent_mask = 0x7F800000ull,
      .fraction_mask = 0x007FFFFFull,
      .exponent_max = 0xFFu,
      .exponent_bias = 127,
      .canonical_nan = kFloat32CanonicalQuietNan,
  };
}

Format Float64Format() {
  return Format{
      .exponent_bits = 11u,
      .fraction_bits = 52u,
      .sign_mask = 0x8000000000000000ull,
      .exponent_mask = 0x7FF0000000000000ull,
      .fraction_mask = 0x000FFFFFFFFFFFFFull,
      .exponent_max = 0x7FFu,
      .exponent_bias = 1023,
      .canonical_nan = kFloat64CanonicalQuietNan,
  };
}

u64 Compose(const bool sign, const u32 raw_exponent, const u64 fraction, const Format& format) {
  const u64 sign_bits = sign ? format.sign_mask : 0u;
  return sign_bits |
         (static_cast<u64>(raw_exponent) << format.fraction_bits) |
         (fraction & format.fraction_mask);
}

u64 Infinity(const bool sign, const Format& format) {
  return Compose(sign, format.exponent_max, 0u, format);
}

u64 Zero(const bool sign, const Format& format) {
  return Compose(sign, 0u, 0u, format);
}

} // namespace rund::kernel::strict_float
