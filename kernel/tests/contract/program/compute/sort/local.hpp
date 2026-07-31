#pragma once

#include "test/assert.hpp"

#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/sort/identity.hpp>
#include <kernel/program/compute/sort/plan.hpp>
#include <kernel/program/compute/sort/reference.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

[[nodiscard]] constexpr rund::kernel::SortDesc U32Sort() noexcept {
  return rund::kernel::SortDesc{
      .key = rund::kernel::SortKey::U32,
      .value = rund::kernel::SortValue::U32,
      .element_count = 16u,
      .radix_bits = 8u,
      .stable = true,
  };
}

[[nodiscard]] inline bool SamePlan(const rund::kernel::SortPlan& lhs,
                                   const rund::kernel::SortPlan& rhs) noexcept {
  return lhs.key == rhs.key && lhs.value == rhs.value &&
         lhs.element_count == rhs.element_count &&
         lhs.key_bytes == rhs.key_bytes &&
         lhs.value_bytes == rhs.value_bytes &&
         lhs.radix_bits == rhs.radix_bits &&
         lhs.key_bits == rhs.key_bits &&
         lhs.radix_pass_count == rhs.radix_pass_count &&
         lhs.bucket_count == rhs.bucket_count &&
         lhs.temp_key_bytes == rhs.temp_key_bytes &&
         lhs.temp_value_bytes == rhs.temp_value_bytes &&
         lhs.temp_count_bytes == rhs.temp_count_bytes &&
         lhs.temp_rank_bytes == rhs.temp_rank_bytes &&
         lhs.temp_bytes == rhs.temp_bytes &&
         lhs.count_source == rhs.count_source &&
         lhs.stable == rhs.stable && lhs.ok == rhs.ok &&
         std::string_view{lhs.reason} == std::string_view{rhs.reason};
}

int test_compute_sort_plan_rejects_unstable_policy();
int test_compute_sort_plan_rejects_zero_count();
int test_compute_sort_plan_rejects_unknown_key_width();
int test_compute_sort_plan_rejects_unknown_value_width();
int test_compute_sort_plan_rejects_unsupported_radix_width();
int test_compute_sort_plan_rejects_unsupported_key_bits();
int test_compute_sort_plan_rejects_temp_byte_overflow();
int test_compute_sort_descriptor_hash_is_deterministic_and_field_sensitive();
int test_compute_sort_plan_is_deterministic();
int test_compute_sort_plan_computes_u32_key_shape();
int test_compute_sort_plan_computes_declared_u16_domain_shape();
int test_compute_sort_plan_computes_identity_u32_value_shape();
int test_compute_sort_plan_computes_u64_key_shape();
int test_compute_sort_cpu_reference_is_stable_for_equal_u32_keys();
int test_compute_sort_cpu_reference_sorts_u64_keys();
int test_compute_sort_cpu_reference_rejects_zero_count();
int test_compute_sort_cpu_reference_rejects_missing_buffers();

}  // namespace program_compute_contract
