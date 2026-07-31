#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/fixed/arithmetic.hpp>
#include <kernel/program/compute/transform/model.hpp>
#include <math32/turn/trig.hpp>
#include <math64/turn/trig.hpp>

#include <bit>
#include <cstdint>

namespace rund::kernel::transform_twiddle {

struct Layout final {
  u64 count = 0u;
  u64 bytes = 0u;
  bool ok = false;
};

[[nodiscard]] constexpr Layout Plan(const u64 elements,
                                    const u32 element_bytes) noexcept {
  if (elements == 0u || !std::has_single_bit(elements) ||
      (element_bytes != sizeof(i32) && element_bytes != sizeof(i64)) ||
      !checked::mul(elements, element_bytes)) {
    return {};
  }
  return Layout{
      .count = elements >> 1u,
      .bytes = elements == 1u ? 0u : elements * element_bytes,
      .ok = true,
  };
}

[[nodiscard]] constexpr u64 Index(const u64 phase, const u64 span,
                                  const u64 elements) noexcept {
  return phase * (elements / span);
}

template <typename S> [[nodiscard]] constexpr S Negate(const S value) noexcept {
  constexpr i128 minimum =
      -(static_cast<i128>(1) << (sizeof(S) * 8u - 1u));
  constexpr i128 maximum =
      (static_cast<i128>(1) << (sizeof(S) * 8u - 1u)) - 1;
  const i128 negated = -static_cast<i128>(value);
  return static_cast<S>(negated < minimum   ? minimum
                        : negated > maximum ? maximum
                                            : negated);
}

template <typename S>
[[nodiscard]] S Cos(const u64 phase, const u64 span,
                    const ComputeFixedFormat format) noexcept {
  const u32 span_bits = static_cast<u32>(std::countr_zero(span));
  if constexpr (sizeof(S) == sizeof(i64)) {
    const u64 turn = phase << (64u - span_bits);
    return compute_fixed_detail::Rescale<S>(
        ::rund::math64::TurnSinCos(turn).cos, 63u, format);
  } else {
    const u32 turn = static_cast<u32>(phase) << (32u - span_bits);
    return compute_fixed_detail::Rescale<S>(
        ::rund::math32::TurnSinCos(turn).cos, 31u, format);
  }
}

template <typename S>
[[nodiscard]] S Sin(const u64 phase, const u64 span,
                    const TransformDir direction,
                    const ComputeFixedFormat format) noexcept {
  const u32 span_bits = static_cast<u32>(std::countr_zero(span));
  S value{};
  if constexpr (sizeof(S) == sizeof(i64)) {
    const u64 turn = phase << (64u - span_bits);
    value = compute_fixed_detail::Rescale<S>(
        ::rund::math64::TurnSinCos(turn).sin, 63u, format);
  } else {
    const u32 turn = static_cast<u32>(phase) << (32u - span_bits);
    value = compute_fixed_detail::Rescale<S>(
        ::rund::math32::TurnSinCos(turn).sin, 31u, format);
  }
  return direction == TransformDir::Forward ? Negate(value) : value;
}

template <typename S>
[[nodiscard]] bool Fill(S *const values, const u64 elements,
                        const TransformDir direction,
                        const ComputeFixedFormat format) noexcept {
  const Layout layout = Plan(elements, sizeof(S));
  if (!layout.ok || (layout.count != 0u && values == nullptr)) {
    return false;
  }
  for (u64 phase = 0u; phase < layout.count; ++phase) {
    values[phase] = Cos<S>(phase, elements, format);
    values[layout.count + phase] = Sin<S>(phase, elements, direction, format);
  }
  return true;
}

} // namespace rund::kernel::transform_twiddle
