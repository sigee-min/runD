#pragma once

#include "../fixed.hpp"
#include "work.hpp"

namespace node_accel_contract::cpu::pick {

[[nodiscard]] inline rund::kernel::i32 ExpectedValue(
    const rund::kernel::i32 lhs,
    const rund::kernel::i32 rhs) {
  const rund::kernel::i32 selected =
      lhs > rhs ? rund::math32::detail::ScalarAddSat(lhs, rhs)
                : rund::math32::detail::ScalarSubSat(lhs, rhs);
  return fixed::QuantizeSum<rund::kernel::i32>(
      fixed::DivNearestEven(lhs, rhs),
      fixed::SqrtFloor(rund::math32::detail::ScalarAbs(lhs)),
      selected);
}

}  // namespace node_accel_contract::cpu::pick
