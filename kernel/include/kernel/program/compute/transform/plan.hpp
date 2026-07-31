#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/transform/model.hpp>
#include <kernel/program/compute/transform/stage.hpp>
#include <kernel/program/compute/transform/twiddle.hpp>

#include <bit>

namespace rund::kernel {
namespace transform_plan_detail {

[[nodiscard]] constexpr bool PowerOfTwo(const u64 value) noexcept {
  return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] constexpr u64 PassCount(const u64 value) noexcept {
  return transform_stage::Dispatches(value);
}

[[nodiscard]] constexpr u64 IntegerSqrt(u64 value) noexcept {
  u64 root = 0u;
  u64 bit = u64{1u} << 62u;
  while (bit > value) {
    bit >>= 2u;
  }
  while (bit != 0u) {
    if (value >= root + bit) {
      value -= root + bit;
      root = (root >> 1u) + bit;
    } else {
      root >>= 1u;
    }
    bit >>= 2u;
  }
  return root;
}

[[nodiscard]] constexpr u64 Divisor(const TransformNorm normalization,
                                    const u64 elements) noexcept {
  return normalization == TransformNorm::None ? 1u
         : normalization == TransformNorm::InverseLength
             ? elements
             : IntegerSqrt(elements);
}

[[nodiscard]] constexpr TransformPlan
Reject(const TransformDesc &desc, const char *const reason) noexcept {
  return TransformPlan{
      .op = desc.op,
      .direction = desc.direction,
      .layout = desc.layout,
      .normalization = desc.normalization,
      .element_count = desc.element_count,
      .fixed_format = desc.fixed_format,
      .reason = reason,
  };
}

} // namespace transform_plan_detail

[[nodiscard]] constexpr TransformPlan
PlanTransform(const TransformDesc &desc) noexcept {
  if (desc.op != TransformOp::Fourier) {
    return transform_plan_detail::Reject(desc,
                                         "compute_transform_op_unsupported");
  }
  if (desc.direction != TransformDir::Forward &&
      desc.direction != TransformDir::Inverse) {
    return transform_plan_detail::Reject(
        desc, "compute_transform_direction_unsupported");
  }
  if (desc.layout != TransformLayout::Split &&
      desc.layout != TransformLayout::Interleaved) {
    return transform_plan_detail::Reject(
        desc, "compute_transform_layout_unsupported");
  }
  if (desc.normalization != TransformNorm::None &&
      desc.normalization != TransformNorm::InverseLength &&
      desc.normalization != TransformNorm::Unitary) {
    return transform_plan_detail::Reject(desc,
                                         "compute_transform_norm_unsupported");
  }
  const u32 element_bytes =
      static_cast<u32>(desc.fixed_format.integer_bits) +
                  static_cast<u32>(desc.fixed_format.fraction_bits) ==
              64u
          ? 8u
          : 4u;
  if (!ComputePrimitiveFixedFormatValid(element_bytes, desc.fixed_format,
                                        ComputeApproximation::Deterministic)) {
    return transform_plan_detail::Reject(
        desc, "compute_transform_numeric_policy_unsupported");
  }
  if (!transform_plan_detail::PowerOfTwo(desc.element_count)) {
    return transform_plan_detail::Reject(
        desc, "compute_transform_count_not_power_of_two");
  }
  if (desc.layout == TransformLayout::Interleaved &&
      !checked::mul(desc.element_count, 2u)) {
    return transform_plan_detail::Reject(desc,
                                         "compute_transform_shape_overflow");
  }
  const transform_twiddle::Layout twiddle =
      transform_twiddle::Plan(desc.element_count, element_bytes);
  if (!twiddle.ok) {
    return transform_plan_detail::Reject(
        desc, "compute_transform_workspace_overflow");
  }
  return TransformPlan{
      .op = desc.op,
      .direction = desc.direction,
      .layout = desc.layout,
      .normalization = desc.normalization,
      .element_count = desc.element_count,
      .twiddle_count = twiddle.count,
      .workspace_bytes = twiddle.bytes,
      .pass_count = transform_plan_detail::PassCount(desc.element_count),
      .normalization_divisor = transform_plan_detail::Divisor(
          desc.normalization, desc.element_count),
      .element_bytes = element_bytes,
      .fixed_format = desc.fixed_format,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
TransformPlanMatchesDesc(const TransformDesc &desc,
                         const TransformPlan &plan) noexcept {
  const u32 element_bytes =
      static_cast<u32>(desc.fixed_format.integer_bits) +
                  static_cast<u32>(desc.fixed_format.fraction_bits) ==
              64u
          ? 8u
          : 4u;
  const transform_twiddle::Layout twiddle =
      transform_twiddle::Plan(desc.element_count, element_bytes);
  return twiddle.ok && plan.ok && desc.op == plan.op &&
         desc.direction == plan.direction && desc.layout == plan.layout &&
         desc.normalization == plan.normalization &&
         desc.element_count == plan.element_count &&
         desc.fixed_format == plan.fixed_format &&
         plan.element_bytes == element_bytes &&
         plan.twiddle_count == twiddle.count &&
         plan.workspace_bytes == twiddle.bytes &&
         plan.normalization_divisor ==
             transform_plan_detail::Divisor(desc.normalization,
                                            desc.element_count) &&
         plan.pass_count ==
             transform_plan_detail::PassCount(desc.element_count);
}

static_assert(transform_plan_detail::IntegerSqrt(0u) == 0u);
static_assert(transform_plan_detail::IntegerSqrt(2u) == 1u);
static_assert(transform_plan_detail::IntegerSqrt(1u << 20u) == 1u << 10u);
static_assert(transform_plan_detail::IntegerSqrt(~u64{0u}) == 0xffffffffu);

} // namespace rund::kernel
