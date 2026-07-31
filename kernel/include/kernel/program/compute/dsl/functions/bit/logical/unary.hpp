#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue bit_not(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::BitNot, value);
}

} // namespace rund::compute_dsl
