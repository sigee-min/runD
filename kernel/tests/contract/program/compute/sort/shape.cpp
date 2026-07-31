#include "local.hpp"

namespace program_compute_contract {

int test_compute_sort_plan_computes_u32_key_shape() {
  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(U32Sort());

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.key == rund::kernel::SortKey::U32);
  TEST_ASSERT(plan.value == rund::kernel::SortValue::U32);
  TEST_ASSERT(plan.element_count == 16u);
  TEST_ASSERT(plan.key_bytes == 4u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.radix_bits == 8u);
  TEST_ASSERT(plan.key_bits == 32u);
  TEST_ASSERT(plan.radix_pass_count == 4u);
  TEST_ASSERT(plan.bucket_count == 256u);
  TEST_ASSERT(plan.temp_key_bytes == 64u);
  TEST_ASSERT(plan.temp_value_bytes == 64u);
  TEST_ASSERT(plan.temp_count_bytes == 4096u);
  TEST_ASSERT(plan.temp_rank_bytes == 64u);
  TEST_ASSERT(plan.temp_bytes == 4288u);
  TEST_ASSERT(plan.count_source ==
              rund::kernel::ComputeCountSource::Descriptor);
  TEST_ASSERT(plan.stable);
  TEST_ASSERT(rund::kernel::SortPlanMatchesDesc(U32Sort(), plan));
  rund::kernel::SortPlan forged = plan;
  ++forged.temp_count_bytes;
  TEST_ASSERT(!rund::kernel::SortPlanMatchesDesc(U32Sort(), forged));
  return 0;
}

int test_compute_sort_plan_computes_declared_u16_domain_shape() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.key_bits = 16u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.key == rund::kernel::SortKey::U32);
  TEST_ASSERT(plan.key_bytes == 4u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.radix_bits == 8u);
  TEST_ASSERT(plan.key_bits == 16u);
  TEST_ASSERT(plan.radix_pass_count == 2u);
  TEST_ASSERT(plan.bucket_count == 256u);
  TEST_ASSERT(plan.temp_key_bytes == 64u);
  TEST_ASSERT(plan.temp_value_bytes == 64u);
  TEST_ASSERT(plan.temp_count_bytes == 2048u);
  TEST_ASSERT(plan.temp_rank_bytes == 64u);
  TEST_ASSERT(plan.temp_bytes == 2240u);
  return 0;
}

int test_compute_sort_plan_computes_identity_u32_value_shape() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.value = rund::kernel::SortValue::IdentityU32;
  desc.key_bits = 16u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);
  const rund::kernel::SortHash identity_hash = rund::kernel::HashSort(desc);
  rund::kernel::SortDesc generic = desc;
  generic.value = rund::kernel::SortValue::U32;
  const rund::kernel::SortHash generic_hash = rund::kernel::HashSort(generic);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.value == rund::kernel::SortValue::IdentityU32);
  TEST_ASSERT(plan.key_bytes == 4u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.radix_pass_count == 2u);
  TEST_ASSERT(plan.temp_key_bytes == 64u);
  TEST_ASSERT(plan.temp_value_bytes == 64u);
  TEST_ASSERT(plan.temp_count_bytes == 2048u);
  TEST_ASSERT(plan.temp_rank_bytes == 64u);
  TEST_ASSERT(plan.temp_bytes == 2240u);
  TEST_ASSERT(identity_hash.hi != generic_hash.hi ||
              identity_hash.lo != generic_hash.lo);
  return 0;
}

int test_compute_sort_plan_computes_u64_key_shape() {
  rund::kernel::SortDesc desc = U32Sort();
  desc.key = rund::kernel::SortKey::U64;
  desc.element_count = 9u;

  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.key == rund::kernel::SortKey::U64);
  TEST_ASSERT(plan.key_bytes == 8u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.key_bits == 64u);
  TEST_ASSERT(plan.radix_pass_count == 8u);
  TEST_ASSERT(plan.bucket_count == 256u);
  TEST_ASSERT(plan.temp_key_bytes == 72u);
  TEST_ASSERT(plan.temp_value_bytes == 36u);
  TEST_ASSERT(plan.temp_count_bytes == 8192u);
  TEST_ASSERT(plan.temp_rank_bytes == 36u);
  TEST_ASSERT(plan.temp_bytes == 8336u);
  return 0;
}

} // namespace program_compute_contract
