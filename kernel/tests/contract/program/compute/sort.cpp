#include "sort/local.hpp"

namespace program_compute_contract {

int RunSortContract() {
  if (test_compute_sort_plan_rejects_unstable_policy() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_zero_count() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_unknown_key_width() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_unknown_value_width() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_unsupported_radix_width() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_unsupported_key_bits() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_rejects_temp_byte_overflow() != 0) {
    return 1;
  }
  if (test_compute_sort_descriptor_hash_is_deterministic_and_field_sensitive() !=
      0) {
    return 1;
  }
  if (test_compute_sort_plan_is_deterministic() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_computes_u32_key_shape() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_computes_declared_u16_domain_shape() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_computes_identity_u32_value_shape() != 0) {
    return 1;
  }
  if (test_compute_sort_plan_computes_u64_key_shape() != 0) {
    return 1;
  }
  if (test_compute_sort_cpu_reference_is_stable_for_equal_u32_keys() != 0) {
    return 1;
  }
  if (test_compute_sort_cpu_reference_sorts_u64_keys() != 0) {
    return 1;
  }
  if (test_compute_sort_cpu_reference_rejects_zero_count() != 0) {
    return 1;
  }
  if (test_compute_sort_cpu_reference_rejects_missing_buffers() != 0) {
    return 1;
  }
  return 0;
}

}  // namespace program_compute_contract
