#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/stencil/model.hpp>

namespace rund::kernel {
namespace stencil_plan_detail {

[[nodiscard]] constexpr u64
ElementBytes(const StencilElement element) noexcept {
  switch (element) {
  case StencilElement::U32:
    return 4u;
  case StencilElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr bool KnownOp(const StencilOp op) noexcept {
  return op == StencilOp::Sum || op == StencilOp::Min || op == StencilOp::Max;
}

[[nodiscard]] constexpr StencilPlan
Reject(const StencilDesc &desc, const u64 element_bytes, const u64 input_bytes,
       const u64 output_bytes, const char *const reason) noexcept {
  return StencilPlan{
      .op = desc.op,
      .element = desc.element,
      .boundary = desc.boundary,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .radius = desc.radius,
      .input_bytes = input_bytes,
      .output_bytes = output_bytes,
      .reason = reason,
  };
}

} // namespace stencil_plan_detail

[[nodiscard]] constexpr StencilPlan
PlanStencil(const StencilDesc &desc) noexcept {
  if (!stencil_plan_detail::KnownOp(desc.op)) {
    return stencil_plan_detail::Reject(desc, 0u, 0u, 0u,
                                       "compute_stencil_op_unsupported");
  }
  if (desc.boundary != StencilBoundary::Clamp) {
    return stencil_plan_detail::Reject(desc, 0u, 0u, 0u,
                                       "compute_stencil_boundary_unsupported");
  }
  const u64 element_bytes = stencil_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return stencil_plan_detail::Reject(desc, 0u, 0u, 0u,
                                       "compute_stencil_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return stencil_plan_detail::Reject(desc, element_bytes, 0u, 0u,
                                       "compute_stencil_count_zero");
  }
  if (desc.radius == 0u || desc.radius > desc.element_count) {
    return stencil_plan_detail::Reject(desc, element_bytes, 0u, 0u,
                                       "compute_stencil_radius_invalid");
  }
  if (!checked::mul(desc.element_count, element_bytes)) {
    return stencil_plan_detail::Reject(desc, element_bytes, 0u, 0u,
                                       "compute_stencil_bytes_overflow");
  }
  const u64 bytes = desc.element_count * element_bytes;
  return StencilPlan{
      .op = desc.op,
      .element = desc.element,
      .boundary = desc.boundary,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .radius = desc.radius,
      .input_bytes = bytes,
      .output_bytes = bytes,
      .temp_bytes = 0u,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
StencilPlanMatchesDesc(const StencilDesc &desc,
                       const StencilPlan &plan) noexcept {
  const StencilPlan expected = PlanStencil(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.boundary == expected.boundary &&
         plan.element_count == expected.element_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.radius == expected.radius &&
         plan.input_bytes == expected.input_bytes &&
         plan.output_bytes == expected.output_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
