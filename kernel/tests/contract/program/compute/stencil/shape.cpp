#include "local.hpp"

namespace program_compute_contract {

int StencilShape() {
  const rund::kernel::StencilPlan u32_plan =
      rund::kernel::PlanStencil(U32Stencil());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.op == rund::kernel::StencilOp::Sum);
  TEST_ASSERT(u32_plan.element == rund::kernel::StencilElement::U32);
  TEST_ASSERT(u32_plan.boundary == rund::kernel::StencilBoundary::Clamp);
  TEST_ASSERT(u32_plan.element_count == 8u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.radius == 1u);
  TEST_ASSERT(u32_plan.input_bytes == 32u);
  TEST_ASSERT(u32_plan.output_bytes == 32u);
  TEST_ASSERT(u32_plan.temp_bytes == 0u);
  TEST_ASSERT(u32_plan.pass_count == 1u);
  TEST_ASSERT(rund::kernel::StencilPlanMatchesDesc(U32Stencil(), u32_plan));
  rund::kernel::StencilPlan forged = u32_plan;
  ++forged.input_bytes;
  TEST_ASSERT(!rund::kernel::StencilPlanMatchesDesc(U32Stencil(), forged));

  rund::kernel::StencilDesc wide_desc = U32Stencil();
  wide_desc.radius = 2u;
  const rund::kernel::StencilPlan wide_plan =
      rund::kernel::PlanStencil(wide_desc);
  TEST_ASSERT(wide_plan.ok);
  TEST_ASSERT(wide_plan.radius == 2u);
  TEST_ASSERT(wide_plan.input_bytes == 32u);
  TEST_ASSERT(wide_plan.output_bytes == 32u);

  rund::kernel::StencilDesc min_desc = U32Stencil();
  min_desc.op = rund::kernel::StencilOp::Min;
  const rund::kernel::StencilPlan min_plan =
      rund::kernel::PlanStencil(min_desc);
  TEST_ASSERT(min_plan.ok);
  TEST_ASSERT(min_plan.op == rund::kernel::StencilOp::Min);

  rund::kernel::StencilDesc max_desc = U32Stencil();
  max_desc.op = rund::kernel::StencilOp::Max;
  const rund::kernel::StencilPlan max_plan =
      rund::kernel::PlanStencil(max_desc);
  TEST_ASSERT(max_plan.ok);
  TEST_ASSERT(max_plan.op == rund::kernel::StencilOp::Max);

  rund::kernel::StencilDesc u64_desc = U32Stencil();
  u64_desc.element = rund::kernel::StencilElement::U64;
  const rund::kernel::StencilPlan u64_plan =
      rund::kernel::PlanStencil(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element == rund::kernel::StencilElement::U64);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.input_bytes == 64u);
  TEST_ASSERT(u64_plan.output_bytes == 64u);
  return 0;
}

} // namespace program_compute_contract
