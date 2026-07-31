#include "local.hpp"

namespace program_compute_contract {

int test_compute_sort_plan_rejects_unstable_policy() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.stable = false;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_sort_stability_required");
  return 0;
}

int test_compute_sort_plan_rejects_zero_count() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.element_count = 0u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_count_zero");
  return 0;
}

int test_compute_sort_plan_rejects_unknown_key_width() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.key = static_cast<rund::kernel::SortKey>(0u);

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_key_unsupported");
  return 0;
}

int test_compute_sort_plan_rejects_unknown_value_width() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.value = static_cast<rund::kernel::SortValue>(0u);

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_value_unsupported");
  return 0;
}

int test_compute_sort_plan_rejects_unsupported_radix_width() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.radix_bits = 4u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_radix_invalid");
  return 0;
}

int test_compute_sort_plan_rejects_unsupported_key_bits() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.key_bits = 24u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_key_bits_invalid");
  return 0;
}

int test_compute_sort_plan_rejects_temp_byte_overflow() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.element_count =
      (std::numeric_limits<rund::kernel::u64>::max() / 4u) + 1u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_sort_temp_overflow");
  return 0;
}

}  // namespace program_compute_contract
