#pragma once

#include <math64/prob/entropy/cross.hpp>

namespace rund::math64::rl {
[[nodiscard]] inline simd::I64x Bellman(const simd::I64x reward,
                                        const simd::I64x gamma,
                                        const simd::I64x next) noexcept {
  return ::rund::math64::AddSat(reward, ::rund::math64::MulFixed(gamma, next));
}
[[nodiscard]] inline simd::I64x TdError(const simd::I64x reward,
                                        const simd::I64x gamma,
                                        const simd::I64x next,
                                        const simd::I64x value) noexcept {
  return ::rund::math64::SubSat(Bellman(reward, gamma, next), value);
}
[[nodiscard]] inline simd::I64x QUpdate(const simd::I64x q,
                                        const simd::I64x alpha,
                                        const simd::I64x td) noexcept {
  return ::rund::math64::AddSat(q, ::rund::math64::MulFixed(alpha, td));
}
[[nodiscard]] inline simd::I64x ReturnStep(const simd::I64x reward,
                                           const simd::I64x gamma,
                                           const simd::I64x next_return) noexcept {
  return Bellman(reward, gamma, next_return);
}
[[nodiscard]] inline simd::I64x GaeStep(const simd::I64x reward,
                                        const simd::I64x gamma,
                                        const simd::I64x next_value,
                                        const simd::I64x value,
                                        const simd::I64x lambda,
                                        const simd::I64x next_advantage) noexcept {
  const simd::I64x delta = TdError(reward, gamma, next_value, value);
  return ::rund::math64::AddSat(delta, ::rund::math64::MulFixed(::rund::math64::MulFixed(gamma, lambda), next_advantage));
}
}  // namespace rund::math64::rl
