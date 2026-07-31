#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
FixedStorageIncrementWrap(const ComputeValue value) noexcept {
  const ComputeValue maximum = fixed_max(value);
  const ComputeValue minimum = FixedLiteral(value, FixedLaneMinimumBits(value));
  const ComputeValue incremented =
      add_sat(value, FixedLiteral(value, 1u));
  return SelectValue(CompareValue(rund::kernel::IrOp::Eq, value, maximum),
                     minimum, incremented);
}

} // namespace rund::compute_dsl::detail

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue noise(const ComputeValue cell,
                                        const ComputeValue amount) noexcept {
  const ComputeValue next = detail::FixedStorageIncrementWrap(cell);
  return lerp(hash(HashOp::Unit, cell), hash(HashOp::Unit, next), fade(amount));
}

[[nodiscard]] inline ComputeValue noise(const ComputeValue cell,
                                        const ComputeValue amount,
                                        const ComputeValue seed) noexcept {
  const ComputeValue next = detail::FixedStorageIncrementWrap(cell);
  return lerp(hash(HashOp::Unit, cell, seed), hash(HashOp::Unit, next, seed),
              fade(amount));
}

} // namespace rund::compute_dsl
