#include "local.hpp"

namespace program_compute_contract {

int GatherReject() {
  rund::kernel::GatherDesc unknown_count_source = U32Gather();
  unknown_count_source.count_source =
      static_cast<rund::kernel::ComputeCountSource>(0xffu);
  const rund::kernel::GatherPlan unknown_count_source_plan =
      rund::kernel::PlanGather(unknown_count_source);
  TEST_ASSERT(!unknown_count_source_plan.ok);
  TEST_ASSERT(std::string_view{unknown_count_source_plan.reason} ==
              "compute_gather_count_source_unsupported");

  rund::kernel::GatherDesc zero_count = U32Gather();
  zero_count.element_count = 0u;
  const rund::kernel::GatherPlan zero_count_plan =
      rund::kernel::PlanGather(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_gather_count_zero");

  rund::kernel::GatherDesc unsupported_count = U32Gather();
  unsupported_count.element_count =
      static_cast<rund::kernel::u64>(
          std::numeric_limits<rund::kernel::u32>::max()) +
      1u;
  const rund::kernel::GatherPlan unsupported_count_plan =
      rund::kernel::PlanGather(unsupported_count);
  TEST_ASSERT(!unsupported_count_plan.ok);
  TEST_ASSERT(std::string_view{unsupported_count_plan.reason} ==
              "compute_gather_count_unsupported");

  rund::kernel::GatherDesc zero_source = U32Gather();
  zero_source.source_count = 0u;
  const rund::kernel::GatherPlan zero_source_plan =
      rund::kernel::PlanGather(zero_source);
  TEST_ASSERT(!zero_source_plan.ok);
  TEST_ASSERT(std::string_view{zero_source_plan.reason} ==
              "compute_gather_source_zero");

  rund::kernel::GatherDesc unaddressable = U32Gather();
  unaddressable.source_count =
      static_cast<rund::kernel::u64>(
          std::numeric_limits<rund::kernel::u32>::max()) +
      2u;
  const rund::kernel::GatherPlan unaddressable_plan =
      rund::kernel::PlanGather(unaddressable);
  TEST_ASSERT(!unaddressable_plan.ok);
  TEST_ASSERT(std::string_view{unaddressable_plan.reason} ==
              "compute_gather_source_unsupported");

  rund::kernel::GatherDesc unknown_element = U32Gather();
  unknown_element.element = static_cast<rund::kernel::GatherElement>(0u);
  const rund::kernel::GatherPlan unknown_element_plan =
      rund::kernel::PlanGather(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_gather_element_unsupported");

  rund::kernel::GatherDesc overflow = U32Gather();
  overflow.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 8u) + 1u;
  overflow.element = rund::kernel::GatherElement::U64;
  const rund::kernel::GatherPlan overflow_plan =
      rund::kernel::PlanGather(overflow);
  TEST_ASSERT(!overflow_plan.ok);
  TEST_ASSERT(std::string_view{overflow_plan.reason} ==
              "compute_gather_count_unsupported");
  return 0;
}

} // namespace program_compute_contract
