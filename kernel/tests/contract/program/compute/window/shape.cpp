#include "local.hpp"

namespace program_compute_contract {

int WindowShape() {
  const rund::kernel::WindowDesc desc = U32Window();
  const rund::kernel::WindowPlan plan = rund::kernel::PlanWindow(desc);
  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.op == rund::kernel::WindowOp::Sum);
  TEST_ASSERT(plan.element == rund::kernel::WindowElement::U32);
  TEST_ASSERT(plan.boundary == rund::kernel::WindowBoundary::Clamp);
  TEST_ASSERT(plan.domain == rund::kernel::ComputeDomain::U32);
  TEST_ASSERT(rund::kernel::ComputeFixedFormatAbsent(plan.fixed_format));
  TEST_ASSERT(plan.input_count == 6u);
  TEST_ASSERT(plan.output_count == 4u);
  TEST_ASSERT(plan.window_size == 3u);
  TEST_ASSERT(plan.stride == 2u);
  TEST_ASSERT(plan.pad_left == 1u);
  TEST_ASSERT(plan.element_bytes == 4u);
  TEST_ASSERT(plan.input_bytes == 24u);
  TEST_ASSERT(plan.output_bytes == 16u);
  TEST_ASSERT(plan.temp_bytes == 0u);
  TEST_ASSERT(plan.pass_count == 1u);
  TEST_ASSERT(rund::kernel::WindowPlanMatchesDesc(desc, plan));
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == rund::kernel::NodeKind::Window);
  TEST_ASSERT(signature.value_count == 2u);
  TEST_ASSERT(signature.output_count == 1u);
  TEST_ASSERT(signature.values[0u].role == rund::kernel::BufferRole::Read);
  TEST_ASSERT(signature.values[0u].count == desc.input_count);
  TEST_ASSERT(signature.values[1u].role == rund::kernel::BufferRole::Write);
  TEST_ASSERT(signature.values[1u].count == desc.output_count);

  rund::kernel::WindowPlan forged = plan;
  ++forged.output_bytes;
  TEST_ASSERT(!rund::kernel::WindowPlanMatchesDesc(desc, forged));

  rund::kernel::WindowDesc wide = desc;
  wide.input_count = 4u;
  wide.output_count = 4u;
  wide.window_size = 9u;
  wide.stride = 1u;
  wide.pad_left = 4u;
  const rund::kernel::WindowPlan wide_plan = rund::kernel::PlanWindow(wide);
  TEST_ASSERT(wide_plan.ok);
  TEST_ASSERT(wide_plan.window_size == 9u);

  rund::kernel::WindowDesc zero = desc;
  zero.input_count = 0u;
  zero.output_count = 0u;
  const rund::kernel::WindowPlan zero_plan = rund::kernel::PlanWindow(zero);
  TEST_ASSERT(zero_plan.ok);
  TEST_ASSERT(zero_plan.input_bytes == 0u);
  TEST_ASSERT(zero_plan.output_bytes == 0u);
  TEST_ASSERT(zero_plan.pass_count == 0u);

  rund::kernel::WindowDesc fixed = desc;
  fixed.domain = rund::kernel::ComputeDomain::Fixed;
  fixed.fixed_format =
      Fixed32(rund::kernel::ComputeOverflow::Saturate,
              rund::kernel::ComputeApproximation::Deterministic);
  const rund::kernel::WindowPlan fixed_plan = rund::kernel::PlanWindow(fixed);
  TEST_ASSERT(fixed_plan.ok);
  TEST_ASSERT(fixed_plan.fixed_format == fixed.fixed_format);
  return 0;
}

} // namespace program_compute_contract
