#pragma once

namespace rund::compute_dsl {

namespace detail {

[[nodiscard]] inline ComputeValue WindowCosTurn(
    const ComputeValue amount) noexcept {
  return detail::Unary(rund::kernel::IrOp::Cos, saturate(amount));
}

} // namespace detail

[[nodiscard]] inline ComputeValue
window(const WindowOpHann, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue half = fixed(FixedOp::Half, t);
  return sub_sat(half, mul_fixed(half, detail::WindowCosTurn(t)));
}

[[nodiscard]] inline ComputeValue
window(const WindowOpHamming, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue a = detail::FixedQ31Constant(t, 0x451eb852u);
  const ComputeValue b = detail::FixedQ31Constant(t, 0x3ae147aeu);
  return sub_sat(a, mul_fixed(b, detail::WindowCosTurn(t)));
}

[[nodiscard]] inline ComputeValue
window(const WindowOpBlackman, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue a0 = detail::FixedQ31Constant(t, 0x35c28f5cu);
  const ComputeValue a1 = fixed(FixedOp::Half, t);
  const ComputeValue a2 = detail::FixedQ31Constant(t, 0x0a3d70a3u);
  const ComputeValue first = sub_sat(a0, mul_fixed(a1, detail::WindowCosTurn(t)));
  return add_sat(first, mul_fixed(a2, detail::WindowCosTurn(add_sat(t, t))));
}

[[nodiscard]] inline ComputeValue
window(const WindowOpLanczos, const ComputeValue amount) noexcept {
  const ComputeValue t = saturate(amount);
  const ComputeValue half = fixed(FixedOp::Half, t);
  const ComputeValue centered = sub_sat(t, half);
  const ComputeValue numerator =
      detail::Unary(rund::kernel::IrOp::Sin, centered);
  const ComputeValue denom =
      max(detail::StorageQuantize(abs(centered)),
          fixed(FixedOp::Quarter, t));
  return div_fixed(numerator, denom);
}

} // namespace rund::compute_dsl
