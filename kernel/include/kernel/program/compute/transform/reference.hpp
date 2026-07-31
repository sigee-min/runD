#pragma once

#include <kernel/program/compute/fixed/arithmetic.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/transform/twiddle.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

namespace rund::kernel {
namespace transform_reference_detail {

template <typename S>
[[nodiscard]] constexpr S ClampSigned(const i128 value) noexcept {
  const i128 max = (static_cast<i128>(1) << (sizeof(S) * 8u - 1u)) - 1;
  const i128 min = -(static_cast<i128>(1) << (sizeof(S) * 8u - 1u));
  if (value > max) {
    return static_cast<S>(max);
  }
  if (value < min) {
    return static_cast<S>(min);
  }
  return static_cast<S>(value);
}

template <typename S>
[[nodiscard]] constexpr S AddSat(const S lhs, const S rhs) noexcept {
  return ClampSigned<S>(static_cast<i128>(lhs) + rhs);
}

template <typename S>
[[nodiscard]] constexpr S SubSat(const S lhs, const S rhs) noexcept {
  return ClampSigned<S>(static_cast<i128>(lhs) - rhs);
}

template <typename S>
[[nodiscard]] constexpr S MulFixed(const S lhs, const S rhs,
                                   const ComputeFixedFormat format) noexcept {
  return compute_fixed_detail::Mul(lhs, rhs, format);
}

template <typename S>
void Normalize(S *const real, S *const imag, const u64 count,
               const u64 divisor) noexcept {
  if (divisor <= 1u) {
    return;
  }
  for (u64 index = 0u; index < count; ++index) {
    real[index] = static_cast<S>(real[index] / static_cast<S>(divisor));
    imag[index] = static_cast<S>(imag[index] / static_cast<S>(divisor));
  }
}

template <typename S>
[[nodiscard]] TransformResult
ReferenceFourierSplit(const S *const input_real, const S *const input_imag,
                      S *const output_real, S *const output_imag,
                      const TransformPlan &plan,
                      const S *const twiddle) noexcept {
  if (!plan.ok) {
    return TransformResult{.reason = plan.reason};
  }
  if (input_real == nullptr || input_imag == nullptr ||
      output_real == nullptr || output_imag == nullptr) {
    return TransformResult{.element_count = plan.element_count,
                           .reason = "compute_transform_buffer_invalid"};
  }
  const u64 count = plan.element_count;
  const u32 bits = static_cast<u32>(std::countr_zero(count));
  for (u64 index = 0u; index < count; ++index) {
    const u64 target = transform_stage::Reverse(index, bits);
    output_real[target] = input_real[index];
    output_imag[target] = input_imag[index];
  }
  for (u64 span = 2u; span <= count; span <<= 1u) {
    const u64 half = span >> 1u;
    for (u64 base = 0u; base < count; base += span) {
      for (u64 phase = 0u; phase < half; ++phase) {
        const u64 twiddle_index = transform_twiddle::Index(phase, span, count);
        const S wr =
            twiddle == nullptr
                ? transform_twiddle::Cos<S>(phase, span, plan.fixed_format)
                : twiddle[twiddle_index];
        const S wi = twiddle == nullptr
                         ? transform_twiddle::Sin<S>(
                               phase, span, plan.direction, plan.fixed_format)
                         : twiddle[plan.twiddle_count + twiddle_index];
        const u64 lhs = base + phase;
        const u64 rhs = lhs + half;
        const S tr = SubSat(MulFixed(wr, output_real[rhs], plan.fixed_format),
                            MulFixed(wi, output_imag[rhs], plan.fixed_format));
        const S ti = AddSat(MulFixed(wr, output_imag[rhs], plan.fixed_format),
                            MulFixed(wi, output_real[rhs], plan.fixed_format));
        const S ur = output_real[lhs];
        const S ui = output_imag[lhs];
        output_real[lhs] = AddSat(ur, tr);
        output_imag[lhs] = AddSat(ui, ti);
        output_real[rhs] = SubSat(ur, tr);
        output_imag[rhs] = SubSat(ui, ti);
      }
    }
  }
  Normalize(output_real, output_imag, count, plan.normalization_divisor);
  return TransformResult{
      .element_count = count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace transform_reference_detail

[[nodiscard]] inline TransformResult
ReferenceFourierSplitI32(const i32 *const input_real,
                         const i32 *const input_imag, i32 *const output_real,
                         i32 *const output_imag, const TransformPlan &plan,
                         const i32 *const twiddle = nullptr) noexcept {
  return transform_reference_detail::ReferenceFourierSplit(
      input_real, input_imag, output_real, output_imag, plan, twiddle);
}

[[nodiscard]] inline TransformResult
ReferenceFourierSplitI64(const i64 *const input_real,
                         const i64 *const input_imag, i64 *const output_real,
                         i64 *const output_imag, const TransformPlan &plan,
                         const i64 *const twiddle = nullptr) noexcept {
  return transform_reference_detail::ReferenceFourierSplit(
      input_real, input_imag, output_real, output_imag, plan, twiddle);
}

} // namespace rund::kernel
