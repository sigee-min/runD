#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/window/model.hpp>

namespace rund::kernel {
namespace window_plan_detail {

[[nodiscard]] constexpr bool KnownOp(const WindowOp op) noexcept {
  return op == WindowOp::Sum || op == WindowOp::Min || op == WindowOp::Max;
}

[[nodiscard]] constexpr bool
KnownBoundary(const WindowBoundary boundary) noexcept {
  return boundary == WindowBoundary::Clamp || boundary == WindowBoundary::Clip;
}

[[nodiscard]] constexpr u64 ElementBytes(const WindowElement element) noexcept {
  return element == WindowElement::U32
             ? 4u
             : (element == WindowElement::U64 ? 8u : 0u);
}

[[nodiscard]] constexpr bool DomainMatches(const ComputeDomain domain,
                                           const u64 element_bytes) noexcept {
  switch (domain) {
  case ComputeDomain::I32:
  case ComputeDomain::U32:
    return element_bytes == 4u;
  case ComputeDomain::I64:
  case ComputeDomain::U64:
    return element_bytes == 8u;
  case ComputeDomain::Fixed:
    return element_bytes == 4u || element_bytes == 8u;
  }
  return false;
}

[[nodiscard]] constexpr WindowPlan Reject(const WindowDesc &desc,
                                          const u64 element_bytes,
                                          const char *const reason) noexcept {
  return WindowPlan{
      .op = desc.op,
      .element = desc.element,
      .boundary = desc.boundary,
      .domain = desc.domain,
      .fixed_format = desc.fixed_format,
      .input_count = desc.input_count,
      .output_count = desc.output_count,
      .window_size = desc.window_size,
      .stride = desc.stride,
      .pad_left = desc.pad_left,
      .element_bytes = element_bytes,
      .reason = reason,
  };
}

[[nodiscard]] constexpr bool
LastWindowIntersects(const WindowDesc &desc) noexcept {
  // With P < K the first window always intersects index zero. Starts are
  // monotone in j, so it is sufficient to prove that the last start is < N.
  u64 last_anchor = 0u;
  if (!checked::mul(desc.output_count - 1u, desc.stride, last_anchor)) {
    return false;
  }
  return last_anchor < desc.pad_left ||
         last_anchor - desc.pad_left < desc.input_count;
}

} // namespace window_plan_detail

[[nodiscard]] constexpr WindowPlan PlanWindow(const WindowDesc &desc) noexcept {
  if (!window_plan_detail::KnownOp(desc.op)) {
    return window_plan_detail::Reject(desc, 0u,
                                      "compute_window_op_unsupported");
  }
  if (!window_plan_detail::KnownBoundary(desc.boundary)) {
    return window_plan_detail::Reject(desc, 0u,
                                      "compute_window_boundary_unsupported");
  }
  const u64 element_bytes = window_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return window_plan_detail::Reject(desc, 0u,
                                      "compute_window_element_unsupported");
  }
  if (!window_plan_detail::DomainMatches(desc.domain, element_bytes)) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_domain_unsupported");
  }
  if (desc.domain == ComputeDomain::Fixed) {
    if (!ComputePrimitiveFixedFormatValid(static_cast<u32>(element_bytes),
                                          desc.fixed_format,
                                          desc.fixed_format.approximation)) {
      return window_plan_detail::Reject(desc, element_bytes,
                                        "compute_window_fixed_invalid");
    }
  } else if (!ComputeFixedFormatAbsent(desc.fixed_format)) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_fixed_unexpected");
  }
  if (desc.window_size == 0u) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_size_invalid");
  }
  if (desc.stride == 0u) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_stride_invalid");
  }
  if (desc.pad_left >= desc.window_size) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_padding_invalid");
  }

  if (desc.input_count == 0u || desc.output_count == 0u) {
    if (desc.input_count != 0u || desc.output_count != 0u) {
      return window_plan_detail::Reject(desc, element_bytes,
                                        "compute_window_count_invalid");
    }
    if (desc.op != WindowOp::Sum) {
      return window_plan_detail::Reject(desc, element_bytes,
                                        "compute_window_count_zero");
    }
    return WindowPlan{
        .op = desc.op,
        .element = desc.element,
        .boundary = desc.boundary,
        .domain = desc.domain,
        .fixed_format = desc.fixed_format,
        .input_count = 0u,
        .output_count = 0u,
        .window_size = desc.window_size,
        .stride = desc.stride,
        .pad_left = desc.pad_left,
        .element_bytes = element_bytes,
        .ok = true,
        .reason = "ok",
    };
  }

  if (!window_plan_detail::LastWindowIntersects(desc)) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_shape_invalid");
  }
  u64 input_bytes = 0u;
  u64 output_bytes = 0u;
  if (!checked::mul(desc.input_count, element_bytes, input_bytes) ||
      !checked::mul(desc.output_count, element_bytes, output_bytes)) {
    return window_plan_detail::Reject(desc, element_bytes,
                                      "compute_window_bytes_overflow");
  }
  return WindowPlan{
      .op = desc.op,
      .element = desc.element,
      .boundary = desc.boundary,
      .domain = desc.domain,
      .fixed_format = desc.fixed_format,
      .input_count = desc.input_count,
      .output_count = desc.output_count,
      .window_size = desc.window_size,
      .stride = desc.stride,
      .pad_left = desc.pad_left,
      .element_bytes = element_bytes,
      .input_bytes = input_bytes,
      .output_bytes = output_bytes,
      .temp_bytes = 0u,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
WindowPlanMatchesDesc(const WindowDesc &desc, const WindowPlan &plan) noexcept {
  const WindowPlan expected = PlanWindow(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.boundary == expected.boundary && plan.domain == expected.domain &&
         plan.fixed_format == expected.fixed_format &&
         plan.input_count == expected.input_count &&
         plan.output_count == expected.output_count &&
         plan.window_size == expected.window_size &&
         plan.stride == expected.stride && plan.pad_left == expected.pad_left &&
         plan.element_bytes == expected.element_bytes &&
         plan.input_bytes == expected.input_bytes &&
         plan.output_bytes == expected.output_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
