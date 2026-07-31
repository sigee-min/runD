#include "local.hpp"

namespace program_compute_contract {

int ScatterShape() {
  const rund::kernel::ScatterPlan u32_plan =
      rund::kernel::PlanScatter(U32Scatter());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.element == rund::kernel::ScatterElement::U32);
  TEST_ASSERT(u32_plan.element_count == 4u);
  TEST_ASSERT(u32_plan.output_count == 8u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.index_bytes == 4u);
  TEST_ASSERT(u32_plan.status_bytes == 36u);
  TEST_ASSERT(u32_plan.temp_bytes == 36u);
  TEST_ASSERT(u32_plan.scratch_slots == 8u);
  TEST_ASSERT(u32_plan.pass_count == 1u);
  TEST_ASSERT(rund::kernel::ScatterPlanMatchesDesc(U32Scatter(), u32_plan));
  rund::kernel::ScatterPlan forged = u32_plan;
  ++forged.scratch_slots;
  TEST_ASSERT(!rund::kernel::ScatterPlanMatchesDesc(U32Scatter(), forged));

  rund::kernel::ScatterDesc u64_desc = U32Scatter();
  u64_desc.element = rund::kernel::ScatterElement::U64;
  const rund::kernel::ScatterPlan u64_plan =
      rund::kernel::PlanScatter(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element == rund::kernel::ScatterElement::U64);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.index_bytes == 4u);
  TEST_ASSERT(u64_plan.status_bytes == 36u);
  TEST_ASSERT(u64_plan.scratch_slots == 8u);
  static_assert(rund::kernel::scatter_plan_detail::ScratchSlots(1u) == 2u);
  static_assert(rund::kernel::scatter_plan_detail::ScratchSlots(8u) == 16u);
  static_assert(rund::kernel::scatter_plan_detail::ScratchSlots(9u) == 32u);
  static_assert(rund::kernel::scatter_plan_detail::ScratchSlots(
                    rund::kernel::u64{1u} << 62u) ==
                (rund::kernel::u64{1u} << 63u));
  static_assert(rund::kernel::scatter_plan_detail::ScratchSlots(
                    (rund::kernel::u64{1u} << 62u) + 1u) == 0u);
  return 0;
}

} // namespace program_compute_contract
