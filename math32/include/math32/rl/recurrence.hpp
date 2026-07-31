#pragma once

#include <math32/prob/entropy/cross.hpp>

namespace rund::math32::rl {
[[nodiscard]] inline simd::I32x Bellman(const simd::I32x reward,
                                        const simd::I32x gamma,
                                        const simd::I32x next) noexcept {
  return ::rund::math32::AddSat(reward, ::rund::math32::MulFixed(gamma, next));
}
[[nodiscard]] inline simd::I32x TdError(const simd::I32x reward,
                                        const simd::I32x gamma,
                                        const simd::I32x next,
                                        const simd::I32x value) noexcept {
  return ::rund::math32::SubSat(Bellman(reward, gamma, next), value);
}
[[nodiscard]] inline simd::I32x QUpdate(const simd::I32x q,
                                        const simd::I32x alpha,
                                        const simd::I32x td) noexcept {
  return ::rund::math32::AddSat(q, ::rund::math32::MulFixed(alpha, td));
}
[[nodiscard]] inline simd::I32x ReturnStep(const simd::I32x reward,
                                           const simd::I32x gamma,
                                           const simd::I32x next_return) noexcept {
  return Bellman(reward, gamma, next_return);
}
[[nodiscard]] inline simd::I32x GaeStep(const simd::I32x reward,
                                        const simd::I32x gamma,
                                        const simd::I32x next_value,
                                        const simd::I32x value,
                                        const simd::I32x lambda,
                                        const simd::I32x next_advantage) noexcept {
  const simd::I32x delta = TdError(reward, gamma, next_value, value);
  return ::rund::math32::AddSat(delta, ::rund::math32::MulFixed(::rund::math32::MulFixed(gamma, lambda), next_advantage));
}
}  // namespace rund::math32::rl
