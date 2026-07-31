#include "local.hpp"

namespace rund::kernel::strict_float {
namespace {

u64 MaskedFiniteBits(const u64 bits, const Format& format) {
  return bits & (format.sign_mask | format.exponent_mask | format.fraction_mask);
}

} // namespace

bool ResolveSpecialAddResult(const Decoded& left,
                             const Decoded& right,
                             const u64 left_bits,
                             const u64 right_bits,
                             const Format& format,
                             const StrictFloatReductionPolicy policy,
                             u64& out_result) {
  if (left.nan || right.nan) {
    out_result = format.canonical_nan;
    return true;
  }
  if (left.inf || right.inf) {
    if (left.inf && right.inf && left.sign != right.sign) {
      out_result = format.canonical_nan;
      return true;
    }
    out_result = Infinity(left.inf ? left.sign : right.sign, format);
    return true;
  }
  if (left.zero && right.zero) {
    const bool negative_zero =
        policy.signed_zero_policy == StrictFloatSignedZeroPolicy::PreserveOnlyWhenBothNegative &&
        left.sign && right.sign;
    out_result = Zero(negative_zero, format);
    return true;
  }
  if (left.zero) {
    out_result = MaskedFiniteBits(right_bits, format);
    return true;
  }
  if (right.zero) {
    out_result = MaskedFiniteBits(left_bits, format);
    return true;
  }
  return false;
}

} // namespace rund::kernel::strict_float
