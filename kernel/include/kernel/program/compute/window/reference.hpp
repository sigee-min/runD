#pragma once

#include <kernel/program/compute/fixed/arithmetic.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include <bit>
#include <limits>
#include <type_traits>

namespace rund::kernel {
namespace window_reference_detail {

[[nodiscard]] constexpr WindowResult Reject(const WindowPlan &plan,
                                            const char *const reason) noexcept {
  return WindowResult{
      .input_count = plan.input_count,
      .output_count = plan.output_count,
      .reason = reason,
  };
}

template <typename T>
[[nodiscard]] constexpr T AddWrap(const T lhs, const T rhs) noexcept {
  using U = std::make_unsigned_t<T>;
  const U value = static_cast<U>(lhs) + static_cast<U>(rhs);
  if constexpr (std::is_signed_v<T>) {
    return std::bit_cast<T>(value);
  } else {
    return value;
  }
}

template <typename T>
[[nodiscard]] constexpr T Identity(const WindowOp op) noexcept {
  return op == WindowOp::Sum
             ? T{0}
             : (op == WindowOp::Min ? std::numeric_limits<T>::max()
                                    : std::numeric_limits<T>::lowest());
}

struct Sample final {
  u64 index{};
  bool active{};
};

[[nodiscard]] constexpr Sample
SampleFor(const WindowPlan &plan, const u64 anchor, const u64 slot) noexcept {
  if (slot < plan.pad_left) {
    const u64 distance = plan.pad_left - slot;
    if (anchor < distance) {
      return Sample{.index = 0u,
                    .active = plan.boundary == WindowBoundary::Clamp};
    }
    const u64 index = anchor - distance;
    return index < plan.input_count
               ? Sample{.index = index, .active = true}
               : Sample{.index = plan.input_count - 1u,
                        .active = plan.boundary == WindowBoundary::Clamp};
  }
  const u64 distance = slot - plan.pad_left;
  if (anchor >= plan.input_count || distance >= plan.input_count - anchor) {
    return Sample{.index = plan.input_count - 1u,
                  .active = plan.boundary == WindowBoundary::Clamp};
  }
  return Sample{.index = anchor + distance, .active = true};
}

template <typename T>
[[nodiscard]] inline WindowResult
ReferenceWindow(const T *const input, T *const output, const WindowPlan &plan,
                const bool fixed) noexcept {
  if (!plan.ok) {
    return Reject(plan, plan.reason);
  }
  if (plan.input_count == 0u && plan.output_count == 0u) {
    return WindowResult{.ok = true, .reason = "ok"};
  }
  if (input == nullptr || output == nullptr) {
    return Reject(plan, "compute_window_buffer_invalid");
  }

  u64 anchor = 0u;
  for (u64 output_index = 0u; output_index < plan.output_count;
       ++output_index) {
    T value = Identity<T>(plan.op);
    for (u64 slot = 0u; slot < plan.window_size; ++slot) {
      const Sample sample = SampleFor(plan, anchor, slot);
      if (!sample.active) {
        continue;
      }
      const T input_value = input[sample.index];
      if (plan.op == WindowOp::Sum) {
        value = fixed ? compute_fixed_detail::Narrow<T>(
                            static_cast<i128>(value) + input_value,
                            plan.fixed_format.overflow)
                      : AddWrap(value, input_value);
      } else if (plan.op == WindowOp::Min) {
        value = input_value < value ? input_value : value;
      } else {
        value = input_value > value ? input_value : value;
      }
    }
    output[output_index] = value;
    if (output_index + 1u != plan.output_count) {
      // PlanWindow proved this product representable.
      anchor += plan.stride;
    }
  }
  return WindowResult{
      .input_count = plan.input_count,
      .output_count = plan.output_count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace window_reference_detail

[[nodiscard]] inline WindowResult
ReferenceWindowI32(const i32 *input, i32 *output,
                   const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, false);
}

[[nodiscard]] inline WindowResult
ReferenceWindowU32(const u32 *input, u32 *output,
                   const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, false);
}

[[nodiscard]] inline WindowResult
ReferenceWindowI64(const i64 *input, i64 *output,
                   const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, false);
}

[[nodiscard]] inline WindowResult
ReferenceWindowU64(const u64 *input, u64 *output,
                   const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, false);
}

[[nodiscard]] inline WindowResult
ReferenceWindowFixedI32(const i32 *input, i32 *output,
                        const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, true);
}

[[nodiscard]] inline WindowResult
ReferenceWindowFixedI64(const i64 *input, i64 *output,
                        const WindowPlan &plan) noexcept {
  return window_reference_detail::ReferenceWindow(input, output, plan, true);
}

} // namespace rund::kernel
