#pragma once

#include <kernel/program/compute/dsl/functions/bit/logical/binary.hpp>
#include <kernel/program/compute/dsl/functions/bit/shift.hpp>

namespace rund::compute_dsl::detail {

template <rund::kernel::u32 FirstShift, rund::kernel::u32 SecondShift,
          rund::kernel::u32 FinalShift>
[[nodiscard]] inline ComputeValue HashFinalizer(
    const ComputeValue value, const rund::kernel::u64 first_multiplier,
    const rund::kernel::u64 second_multiplier) noexcept {
  ComputeValue x = bit_xor(value, shr_logical_const<FirstShift>(value));
  x = Binary(rund::kernel::IrOp::MulWrap, x,
             Constant(value, first_multiplier));
  x = bit_xor(x, shr_logical_const<SecondShift>(x));
  x = Binary(rund::kernel::IrOp::MulWrap, x,
             Constant(value, second_multiplier));
  return bit_xor(x, shr_logical_const<FinalShift>(x));
}

} // namespace rund::compute_dsl::detail

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue hash(const ComputeValue value) noexcept {
  if (detail::ScalarModeOf(value) == detail::ScalarMode::FixedLane64) {
    return detail::HashFinalizer<33, 33, 33>(
        value, 0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull);
  }
  return detail::HashFinalizer<16, 15, 16>(value, 0x7feb352dull,
                                           0x846ca68bull);
}

[[nodiscard]] inline ComputeValue hash(const ComputeValue value,
                                       const ComputeValue seed) noexcept {
  return hash(bit_xor(value, seed));
}

} // namespace rund::compute_dsl
