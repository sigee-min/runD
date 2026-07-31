#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
noise(const ComputeValue x, const ComputeValue y, const ComputeValue tx,
      const ComputeValue ty, const ComputeValue seed) noexcept {
  const ComputeValue x_next = detail::FixedStorageIncrementWrap(x);
  const ComputeValue y_next = detail::FixedStorageIncrementWrap(y);
  const ComputeValue y_seed = bit_xor(y, seed);
  const ComputeValue y_next_seed = bit_xor(y_next, seed);
  const ComputeValue row0 =
      lerp(hash(HashOp::Unit, x, y_seed), hash(HashOp::Unit, x_next, y_seed),
           fade(tx));
  const ComputeValue row1 =
      lerp(hash(HashOp::Unit, x, y_next_seed),
           hash(HashOp::Unit, x_next, y_next_seed), fade(tx));
  return lerp(row0, row1, fade(ty));
}

[[nodiscard]] inline ComputeValue
noise(const ComputeValue x, const ComputeValue y, const ComputeValue tx,
      const ComputeValue ty) noexcept {
  return noise(x, y, tx, ty, fixed_zero(x));
}

} // namespace rund::compute_dsl
