#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
window(const WindowOpTriangular, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue delta =
      detail::StorageQuantize(
          centered(CenteredOp::Abs, t, fixed(FixedOp::Half, t)));
  return sub_sat(fixed_one(t), add_sat(delta, delta));
}

} // namespace rund::compute_dsl
