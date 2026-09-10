#pragma once

#include <string_view>

namespace rund::node::accel::detail {

inline constexpr std::string_view MetalWideSimdPrefixSource = R"MSL(// Exact modulo-64 prefix using only 32-bit SIMD arithmetic.
inline ulong rund_simd_prefix_u64(ulong value) {
  const uint low = uint(value);
  const uint prefix_low = simd_prefix_inclusive_sum(low);
  const uint carry = uint(prefix_low < low);
  const uint prefix_high = simd_prefix_inclusive_sum(uint(value >> 32u))
      + simd_prefix_inclusive_sum(carry);
  return (ulong(prefix_high) << 32u) | ulong(prefix_low);
}
inline ulong rund_simd_last_u64(ulong value, uint last) {
  return (ulong(simd_shuffle(uint(value >> 32u), last)) << 32u)
      | ulong(simd_shuffle(uint(value), last));
}

)MSL";

} // namespace rund::node::accel::detail
