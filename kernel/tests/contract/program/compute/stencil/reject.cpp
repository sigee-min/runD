#include "local.hpp"

namespace program_compute_contract {

int StencilReject() {
  rund::kernel::StencilDesc zero_count = U32Stencil();
  zero_count.element_count = 0u;
  const rund::kernel::StencilPlan zero_count_plan =
      rund::kernel::PlanStencil(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_stencil_count_zero");

  rund::kernel::StencilDesc unknown_op = U32Stencil();
  unknown_op.op = static_cast<rund::kernel::StencilOp>(0u);
  const rund::kernel::StencilPlan unknown_op_plan =
      rund::kernel::PlanStencil(unknown_op);
  TEST_ASSERT(!unknown_op_plan.ok);
  TEST_ASSERT(std::string_view{unknown_op_plan.reason} ==
              "compute_stencil_op_unsupported");

  rund::kernel::StencilDesc unknown_boundary = U32Stencil();
  unknown_boundary.boundary =
      static_cast<rund::kernel::StencilBoundary>(0u);
  const rund::kernel::StencilPlan unknown_boundary_plan =
      rund::kernel::PlanStencil(unknown_boundary);
  TEST_ASSERT(!unknown_boundary_plan.ok);
  TEST_ASSERT(std::string_view{unknown_boundary_plan.reason} ==
              "compute_stencil_boundary_unsupported");

  rund::kernel::StencilDesc unknown_element = U32Stencil();
  unknown_element.element =
      static_cast<rund::kernel::StencilElement>(0u);
  const rund::kernel::StencilPlan unknown_element_plan =
      rund::kernel::PlanStencil(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_stencil_element_unsupported");

  rund::kernel::StencilDesc zero_radius = U32Stencil();
  zero_radius.radius = 0u;
  const rund::kernel::StencilPlan zero_radius_plan =
      rund::kernel::PlanStencil(zero_radius);
  TEST_ASSERT(!zero_radius_plan.ok);
  TEST_ASSERT(std::string_view{zero_radius_plan.reason} ==
              "compute_stencil_radius_invalid");

  rund::kernel::StencilDesc overflow = U32Stencil();
  overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 8u) + 1u;
  overflow.element = rund::kernel::StencilElement::U64;
  const rund::kernel::StencilPlan overflow_plan =
      rund::kernel::PlanStencil(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_stencil_bytes_overflow");
  return 0;
}

}  // namespace program_compute_contract
