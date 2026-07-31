#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
mix(const ComputeValue a, const ComputeValue b, const ComputeValue wa,
    const ComputeValue wb) noexcept {
  return add_sat(mul_fixed(a, wa), mul_fixed(b, wb));
}

[[nodiscard]] inline ComputeValue
mix(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue wa, const ComputeValue wb,
    const ComputeValue wc) noexcept {
  return add_sat(mix(a, b, wa, wb), mul_fixed(c, wc));
}

[[nodiscard]] inline ComputeValue
mix(const ComputeValue a, const ComputeValue b, const ComputeValue c,
    const ComputeValue d, const ComputeValue wa, const ComputeValue wb,
    const ComputeValue wc, const ComputeValue wd) noexcept {
  return add_sat(mix(a, b, wa, wb), mix(c, d, wc, wd));
}

} // namespace rund::compute_dsl
