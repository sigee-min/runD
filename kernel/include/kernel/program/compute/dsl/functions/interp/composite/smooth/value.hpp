#pragma once

#include <kernel/program/compute/dsl/functions/interp/base/blend.hpp>
#include <kernel/program/compute/dsl/functions/interp/base/smoothing.hpp>

namespace rund::compute_dsl {

struct LerpSmooth final {};

struct LerpOp final {
  inline static constexpr LerpSmooth Smooth{};
};

[[nodiscard]] inline ComputeValue lerp(const LerpSmooth, const ComputeValue lhs,
                                       const ComputeValue rhs,
                                       const ComputeValue t) noexcept {
  return lerp(lhs, rhs, fade(t));
}

[[nodiscard]] inline ComputeValue
lerp(const LerpSmooth, const ComputeValue x00, const ComputeValue x10,
     const ComputeValue x01, const ComputeValue x11, const ComputeValue tx,
     const ComputeValue ty) noexcept {
  return lerp(x00, x10, x01, x11, fade(tx), fade(ty));
}

[[nodiscard]] inline ComputeValue
lerp(const LerpSmooth, const ComputeValue x000, const ComputeValue x100,
     const ComputeValue x010, const ComputeValue x110,
     const ComputeValue x001, const ComputeValue x101,
     const ComputeValue x011, const ComputeValue x111, const ComputeValue tx,
     const ComputeValue ty, const ComputeValue tz) noexcept {
  return lerp(x000, x100, x010, x110, x001, x101, x011, x111, fade(tx),
              fade(ty), fade(tz));
}

} // namespace rund::compute_dsl
