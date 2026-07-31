#include "add/local.hpp"

namespace rund::kernel::strict_float {

u64 AddBits(const u64 left_bits,
            const u64 right_bits,
            const Format& format,
            const StrictFloatReductionPolicy policy) {
  const Decoded left = Decode(left_bits, format);
  const Decoded right = Decode(right_bits, format);
  u64 special_result = 0u;
  if (ResolveSpecialAddResult(left, right, left_bits, right_bits, format, policy, special_result)) {
    return special_result;
  }
  return ComposeRoundedFiniteSum(AddFiniteMantissas(left, right), format);
}

u64 AddFloat32Bits(const u64 left_bits,
                   const u64 right_bits,
                   const StrictFloatReductionPolicy policy) {
  return AddBits(left_bits & 0xFFFFFFFFull, right_bits & 0xFFFFFFFFull, Float32Format(), policy);
}

u64 AddFloat64Bits(const u64 left_bits,
                   const u64 right_bits,
                   const StrictFloatReductionPolicy policy) {
  return AddBits(left_bits, right_bits, Float64Format(), policy);
}

} // namespace rund::kernel::strict_float
