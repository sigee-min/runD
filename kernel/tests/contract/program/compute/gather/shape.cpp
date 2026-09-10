#include "local.hpp"

namespace program_compute_contract {

int GatherShape() {
  const rund::kernel::GatherPlan u32_plan =
      rund::kernel::PlanGather(U32Gather());
  TEST_ASSERT(u32_plan.ok);
  TEST_ASSERT(std::string_view{u32_plan.reason} == "ok");
  TEST_ASSERT(u32_plan.element == rund::kernel::GatherElement::U32);
  TEST_ASSERT(u32_plan.element_count == 8u);
  TEST_ASSERT(u32_plan.source_count == 16u);
  TEST_ASSERT(u32_plan.element_bytes == 4u);
  TEST_ASSERT(u32_plan.index_bytes == 4u);
  TEST_ASSERT(u32_plan.status_bytes == 8u);
  TEST_ASSERT(u32_plan.pass_count == 2u);
  TEST_ASSERT(u32_plan.temp_bytes == 24u);
  TEST_ASSERT(u32_plan.count_source ==
              rund::kernel::ComputeCountSource::Descriptor);
  TEST_ASSERT(rund::kernel::GatherPlanMatchesDesc(U32Gather(), u32_plan));
  rund::kernel::GatherPlan forged = u32_plan;
  ++forged.temp_bytes;
  TEST_ASSERT(!rund::kernel::GatherPlanMatchesDesc(U32Gather(), forged));

  rund::kernel::GatherDesc bounded_desc = U32Gather();
  bounded_desc.count_source = rund::kernel::ComputeCountSource::BufferU32;
  const rund::kernel::GatherPlan bounded_plan =
      rund::kernel::PlanGather(bounded_desc);
  TEST_ASSERT(bounded_plan.ok);
  TEST_ASSERT(bounded_plan.count_source ==
              rund::kernel::ComputeCountSource::BufferU32);

  rund::kernel::GatherDesc u64_desc = U32Gather();
  u64_desc.element = rund::kernel::GatherElement::U64;
  const rund::kernel::GatherPlan u64_plan = rund::kernel::PlanGather(u64_desc);
  TEST_ASSERT(u64_plan.ok);
  TEST_ASSERT(u64_plan.element == rund::kernel::GatherElement::U64);
  TEST_ASSERT(u64_plan.element_bytes == 8u);
  TEST_ASSERT(u64_plan.index_bytes == 4u);
  TEST_ASSERT(u64_plan.status_bytes == 8u);
  TEST_ASSERT(u64_plan.pass_count == 2u);
  TEST_ASSERT(u64_plan.temp_bytes == 24u);
  for (const rund::kernel::u64 capacity :
       {65536ull, 65537ull, 1048576ull, 4194304ull, 4294967295ull}) {
    auto desc = U32Gather();
    desc.element_count = capacity;
    const auto plan = rund::kernel::PlanGather(desc);
    TEST_ASSERT(plan.ok);
    TEST_ASSERT(plan.preflight.group_count >= 1u);
    TEST_ASSERT(plan.preflight.group_count <= 1024u);
    TEST_ASSERT(plan.preflight.partial_bytes <= 4096u);
    TEST_ASSERT(plan.pass_count == (capacity <= 65536u ? 2u : 3u));
    TEST_ASSERT(plan.status_bytes == 8u + plan.preflight.partial_bytes);
    TEST_ASSERT(plan.temp_bytes == plan.status_bytes + 16u);
    TEST_ASSERT(rund::kernel::GatherPlanMatchesDesc(desc, plan));
    auto forged_shape = plan;
    ++forged_shape.preflight.group_count;
    TEST_ASSERT(!rund::kernel::GatherPlanMatchesDesc(desc, forged_shape));
  }
  return 0;
}

} // namespace program_compute_contract
