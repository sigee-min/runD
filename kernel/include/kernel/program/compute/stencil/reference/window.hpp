#pragma once

#include <kernel/program/compute/stencil/plan.hpp>

namespace rund::kernel::stencil_reference {

[[nodiscard]] constexpr StencilResult Reject(
    const u64 element_count,
    const char* const reason) noexcept {
  return StencilResult{
      .element_count = element_count,
      .reason = reason,
  };
}

template <typename Element>
[[nodiscard]] constexpr Element Combine(const StencilOp op,
                                        const Element current,
                                        const Element value) noexcept {
  if (op == StencilOp::Min) {
    return value < current ? value : current;
  }
  if (op == StencilOp::Max) {
    return value > current ? value : current;
  }
  return static_cast<Element>(current + value);
}

template <typename Element>
[[nodiscard]] inline StencilResult ReferenceWindow(
    const Element* const input,
    Element* const output,
    const u64 element_count,
    const u64 radius,
    const StencilOp op) noexcept {
  if (element_count == 0u) {
    return Reject(0u, "compute_stencil_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return Reject(element_count, "compute_stencil_buffer_invalid");
  }
  if (radius == 0u || radius > element_count) {
    return Reject(element_count, "compute_stencil_radius_invalid");
  }

  for (u64 index = 0u; index < element_count; ++index) {
    Element value = input[index];
    for (u64 step = 1u; step <= radius; ++step) {
      const u64 left = index < step ? 0u : index - step;
      const u64 right = index + step >= element_count ? element_count - 1u
                                                      : index + step;
      value = Combine(op, value, input[left]);
      value = Combine(op, value, input[right]);
    }
    output[index] = value;
  }

  return StencilResult{
      .element_count = element_count,
      .ok = true,
      .reason = "ok",
  };
}

}  // namespace rund::kernel::stencil_reference
