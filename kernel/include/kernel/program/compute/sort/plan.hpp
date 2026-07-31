#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/sort/model.hpp>

namespace rund::kernel {
namespace sort_plan_detail {

[[nodiscard]] constexpr u64 KeyBytes(const SortKey key) noexcept {
  switch (key) {
  case SortKey::U32:
    return 4u;
  case SortKey::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr u64 ValueBytes(const SortValue value) noexcept {
  switch (value) {
  case SortValue::U32:
  case SortValue::IdentityU32:
    return 4u;
  }
  return 0u;
}

[[nodiscard]] constexpr SortPlan
Reject(const SortDesc &desc, const u64 key_bytes, const u64 value_bytes,
       const u32 radix_pass_count, const u64 bucket_count,
       const u64 temp_key_bytes, const u64 temp_value_bytes,
       const u64 temp_count_bytes, const u64 temp_rank_bytes,
       const u64 temp_bytes, const char *const reason) noexcept {
  return SortPlan{
      .key = desc.key,
      .value = desc.value,
      .element_count = desc.element_count,
      .key_bytes = key_bytes,
      .value_bytes = value_bytes,
      .radix_bits = desc.radix_bits,
      .key_bits = desc.key_bits,
      .radix_pass_count = radix_pass_count,
      .bucket_count = bucket_count,
      .temp_key_bytes = temp_key_bytes,
      .temp_value_bytes = temp_value_bytes,
      .temp_count_bytes = temp_count_bytes,
      .temp_rank_bytes = temp_rank_bytes,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .stable = desc.stable,
      .reason = reason,
  };
}

[[nodiscard]] constexpr u32 EffectiveKeyBits(const u32 declared_key_bits,
                                             const u64 key_bytes) noexcept {
  return declared_key_bits == 0u ? static_cast<u32>(key_bytes * 8u)
                                 : declared_key_bits;
}

[[nodiscard]] constexpr bool KeyBitsAdmitted(const u32 key_bits,
                                             const u64 key_bytes) noexcept {
  return key_bits != 0u && key_bits <= key_bytes * 8u && key_bits % 16u == 0u;
}

} // namespace sort_plan_detail

[[nodiscard]] constexpr SortPlan PlanSort(const SortDesc &desc) noexcept {
  if (!desc.stable) {
    return sort_plan_detail::Reject(desc, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                    "compute_sort_stability_required");
  }
  const u64 key_bytes = sort_plan_detail::KeyBytes(desc.key);
  if (key_bytes == 0u) {
    return sort_plan_detail::Reject(desc, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                    "compute_sort_key_unsupported");
  }
  const u64 value_bytes = sort_plan_detail::ValueBytes(desc.value);
  if (value_bytes == 0u) {
    return sort_plan_detail::Reject(desc, key_bytes, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                    0u, "compute_sort_value_unsupported");
  }
  if (desc.element_count == 0u) {
    return sort_plan_detail::Reject(desc, key_bytes, value_bytes, 0u, 0u, 0u,
                                    0u, 0u, 0u, 0u, "compute_sort_count_zero");
  }
  if (desc.radix_bits != 8u) {
    return sort_plan_detail::Reject(desc, key_bytes, value_bytes, 0u, 0u, 0u,
                                    0u, 0u, 0u, 0u,
                                    "compute_sort_radix_invalid");
  }
  if (desc.count_source != ComputeCountSource::Descriptor &&
      ComputeCountBytes(desc.count_source) == 0u) {
    return sort_plan_detail::Reject(desc, key_bytes, value_bytes, 0u, 0u, 0u,
                                    0u, 0u, 0u, 0u,
                                    "compute_sort_count_source_unsupported");
  }
  const u32 key_bits =
      sort_plan_detail::EffectiveKeyBits(desc.key_bits, key_bytes);
  if (!sort_plan_detail::KeyBitsAdmitted(key_bits, key_bytes)) {
    return sort_plan_detail::Reject(desc, key_bytes, value_bytes, 0u, 0u, 0u,
                                    0u, 0u, 0u, 0u,
                                    "compute_sort_key_bits_invalid");
  }

  constexpr u64 kCountBytes = 4u;
  constexpr u64 kRankBytes = 4u;
  constexpr u64 kBucketCount = 256u;
  const u32 radix_pass_count = key_bits / desc.radix_bits;

  if (!checked::mul(desc.element_count, key_bytes) ||
      !checked::mul(desc.element_count, value_bytes) ||
      !checked::mul(desc.element_count, kRankBytes) ||
      !checked::mul(kBucketCount, radix_pass_count)) {
    return sort_plan_detail::Reject(desc, key_bytes, value_bytes,
                                    radix_pass_count, kBucketCount, 0u, 0u, 0u,
                                    0u, 0u, "compute_sort_temp_overflow");
  }
  const u64 temp_key_bytes = desc.element_count * key_bytes;
  const u64 temp_value_bytes = desc.element_count * value_bytes;
  const u64 temp_rank_bytes = desc.element_count * kRankBytes;
  const u64 count_sets = kBucketCount * static_cast<u64>(radix_pass_count);
  if (!checked::mul(count_sets, kCountBytes)) {
    return sort_plan_detail::Reject(
        desc, key_bytes, value_bytes, radix_pass_count, kBucketCount,
        temp_key_bytes, temp_value_bytes, 0u, temp_rank_bytes, 0u,
        "compute_sort_temp_overflow");
  }
  const u64 temp_count_bytes = count_sets * kCountBytes;

  u64 temp_bytes = temp_key_bytes;
  if (!checked::add(temp_bytes, temp_value_bytes)) {
    return sort_plan_detail::Reject(
        desc, key_bytes, value_bytes, radix_pass_count, kBucketCount,
        temp_key_bytes, temp_value_bytes, temp_count_bytes, temp_rank_bytes, 0u,
        "compute_sort_temp_overflow");
  }
  temp_bytes += temp_value_bytes;
  if (!checked::add(temp_bytes, temp_count_bytes)) {
    return sort_plan_detail::Reject(
        desc, key_bytes, value_bytes, radix_pass_count, kBucketCount,
        temp_key_bytes, temp_value_bytes, temp_count_bytes, temp_rank_bytes, 0u,
        "compute_sort_temp_overflow");
  }
  temp_bytes += temp_count_bytes;
  if (!checked::add(temp_bytes, temp_rank_bytes)) {
    return sort_plan_detail::Reject(
        desc, key_bytes, value_bytes, radix_pass_count, kBucketCount,
        temp_key_bytes, temp_value_bytes, temp_count_bytes, temp_rank_bytes, 0u,
        "compute_sort_temp_overflow");
  }
  temp_bytes += temp_rank_bytes;

  return SortPlan{
      .key = desc.key,
      .value = desc.value,
      .element_count = desc.element_count,
      .key_bytes = key_bytes,
      .value_bytes = value_bytes,
      .radix_bits = desc.radix_bits,
      .key_bits = key_bits,
      .radix_pass_count = radix_pass_count,
      .bucket_count = kBucketCount,
      .temp_key_bytes = temp_key_bytes,
      .temp_value_bytes = temp_value_bytes,
      .temp_count_bytes = temp_count_bytes,
      .temp_rank_bytes = temp_rank_bytes,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .stable = desc.stable,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
SortPlanMatchesDesc(const SortDesc &desc, const SortPlan &plan) noexcept {
  const SortPlan expected = PlanSort(desc);
  return expected.ok && plan.ok && plan.key == expected.key &&
         plan.value == expected.value &&
         plan.element_count == expected.element_count &&
         plan.key_bytes == expected.key_bytes &&
         plan.value_bytes == expected.value_bytes &&
         plan.radix_bits == expected.radix_bits &&
         plan.key_bits == expected.key_bits &&
         plan.radix_pass_count == expected.radix_pass_count &&
         plan.bucket_count == expected.bucket_count &&
         plan.temp_key_bytes == expected.temp_key_bytes &&
         plan.temp_value_bytes == expected.temp_value_bytes &&
         plan.temp_count_bytes == expected.temp_count_bytes &&
         plan.temp_rank_bytes == expected.temp_rank_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.count_source == expected.count_source &&
         plan.stable == expected.stable;
}

} // namespace rund::kernel
