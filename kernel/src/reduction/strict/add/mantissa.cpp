#include "local.hpp"

#include <algorithm>

namespace rund::kernel::strict_float {

FiniteSum AddFiniteMantissas(const Decoded& left, const Decoded& right) {
  i32 common_exponent = std::max(left.exponent, right.exponent);
  u128 left_mantissa = left.mantissa << kAddGuardBits;
  u128 right_mantissa = right.mantissa << kAddGuardBits;
  left_mantissa = ShiftRightJam128(left_mantissa, static_cast<u32>(common_exponent - left.exponent));
  right_mantissa = ShiftRightJam128(right_mantissa, static_cast<u32>(common_exponent - right.exponent));

  const i128 signed_left = left.sign ? -static_cast<i128>(left_mantissa)
                                         : static_cast<i128>(left_mantissa);
  const i128 signed_right = right.sign ? -static_cast<i128>(right_mantissa)
                                           : static_cast<i128>(right_mantissa);
  const i128 signed_sum = signed_left + signed_right;
  if (signed_sum == 0) {
    return FiniteSum{.zero = true};
  }
  const bool result_sign = signed_sum < 0;
  const u128 mantissa = result_sign
                                         ? static_cast<u128>(-signed_sum)
                                         : static_cast<u128>(signed_sum);
  return FiniteSum{
      .zero = false,
      .sign = result_sign,
      .exponent = common_exponent,
      .mantissa = mantissa,
  };
}

} // namespace rund::kernel::strict_float
