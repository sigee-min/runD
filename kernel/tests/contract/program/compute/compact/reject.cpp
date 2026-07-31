#include "local.hpp"

namespace program_compute_contract {

int CompactReject() {
  rund::kernel::CompactDesc zero_count = U32Compact();
  zero_count.element_count = 0u;
  const rund::kernel::CompactPlan zero_count_plan =
      rund::kernel::PlanCompact(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_compact_count_zero");

  rund::kernel::CompactDesc zero_capacity = U32Compact();
  zero_capacity.output_capacity = 0u;
  const rund::kernel::CompactPlan zero_capacity_plan =
      rund::kernel::PlanCompact(zero_capacity);
  TEST_ASSERT(!zero_capacity_plan.ok);
  TEST_ASSERT(std::string_view{zero_capacity_plan.reason} ==
              "compute_compact_capacity_zero");

  rund::kernel::CompactDesc bad_flag = U32Compact();
  bad_flag.flag_bytes = 8u;
  rund::kernel::CompactDesc bad_output = U32Compact();
  bad_output.output_bytes = 8u;
  const rund::kernel::CompactPlan flag_plan =
      rund::kernel::PlanCompact(bad_flag);
  const rund::kernel::CompactPlan output_plan =
      rund::kernel::PlanCompact(bad_output);
  TEST_ASSERT(!flag_plan.ok);
  TEST_ASSERT(std::string_view{flag_plan.reason} == "compute_compact_invalid");
  TEST_ASSERT(!output_plan.ok);
  TEST_ASSERT(std::string_view{output_plan.reason} ==
              "compute_compact_invalid");

  rund::kernel::CompactDesc overflow = U32Compact();
  overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 4u) + 1u;
  const rund::kernel::CompactPlan overflow_plan =
      rund::kernel::PlanCompact(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_compact_temp_overflow");
  return 0;
}

} // namespace program_compute_contract
