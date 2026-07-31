#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue log(const ComputeValue value) noexcept {
  return detail::Unary(rund::kernel::IrOp::Log, value);
}

} // namespace rund::compute_dsl
