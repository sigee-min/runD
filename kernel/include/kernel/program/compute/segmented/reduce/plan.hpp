#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/operation.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>

namespace rund::kernel {
namespace segmented_reduce_plan_detail {

inline constexpr u64 kHeadBytes = 4u;
inline constexpr u64 kStatusBytes = 4u;

[[nodiscard]] constexpr u64 ElementBytes(const ReduceElement element) noexcept {
  switch (element) {
  case ReduceElement::U32:
    return 4u;
  case ReduceElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr SegmentedReducePlan
Reject(const SegmentedReduceDesc &desc, const u64 element_bytes,
       const u64 block_count, const u64 temp_value_bytes,
       const u64 temp_head_bytes, const u64 status_bytes, const u64 temp_bytes,
       const char *const reason) noexcept {
  return SegmentedReducePlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .head_bytes = kHeadBytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .temp_value_bytes = temp_value_bytes,
      .temp_head_bytes = temp_head_bytes,
      .status_bytes = status_bytes,
      .temp_bytes = temp_bytes,
      .reason = reason,
  };
}

} // namespace segmented_reduce_plan_detail

[[nodiscard]] constexpr SegmentedReducePlan
PlanSegmentedReduce(const SegmentedReduceDesc &desc) noexcept {
  if (!reduce::valid(desc.op)) {
    return segmented_reduce_plan_detail::Reject(
        desc, 0u, 0u, 0u, 0u, 0u, 0u,
        "compute_segmented_reduce_op_unsupported");
  }
  const u64 element_bytes =
      segmented_reduce_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return segmented_reduce_plan_detail::Reject(
        desc, 0u, 0u, 0u, 0u, 0u, 0u,
        "compute_segmented_reduce_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return segmented_reduce_plan_detail::Reject(
        desc, element_bytes, 0u, 0u, 0u, 0u, 0u,
        "compute_segmented_reduce_count_zero");
  }
  if (desc.block_size == 0u) {
    return segmented_reduce_plan_detail::Reject(
        desc, element_bytes, 0u, 0u, 0u, 0u, 0u,
        "compute_segmented_reduce_block_invalid");
  }

  const u64 block_count = checked::ceil(desc.element_count, desc.block_size);
  if (!checked::mul(desc.element_count, element_bytes) ||
      !checked::mul(desc.element_count,
                    segmented_reduce_plan_detail::kHeadBytes)) {
    return segmented_reduce_plan_detail::Reject(
        desc, element_bytes, block_count, 0u, 0u, 0u, 0u,
        "compute_segmented_reduce_temp_overflow");
  }
  const u64 temp_value_bytes = desc.element_count * element_bytes;
  const u64 temp_head_bytes =
      desc.element_count * segmented_reduce_plan_detail::kHeadBytes;
  if (!checked::add(temp_value_bytes, temp_head_bytes)) {
    return segmented_reduce_plan_detail::Reject(
        desc, element_bytes, block_count, temp_value_bytes, temp_head_bytes, 0u,
        0u, "compute_segmented_reduce_temp_overflow");
  }
  const u64 data_temp_bytes = temp_value_bytes + temp_head_bytes;
  if (!checked::add(data_temp_bytes,
                    segmented_reduce_plan_detail::kStatusBytes)) {
    return segmented_reduce_plan_detail::Reject(
        desc, element_bytes, block_count, temp_value_bytes, temp_head_bytes,
        segmented_reduce_plan_detail::kStatusBytes, 0u,
        "compute_segmented_reduce_temp_overflow");
  }
  return SegmentedReducePlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .head_bytes = segmented_reduce_plan_detail::kHeadBytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .pass_count = block_count > 1u ? 2u : 1u,
      .temp_value_bytes = temp_value_bytes,
      .temp_head_bytes = temp_head_bytes,
      .status_bytes = segmented_reduce_plan_detail::kStatusBytes,
      .temp_bytes =
          data_temp_bytes + segmented_reduce_plan_detail::kStatusBytes,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
SegmentedReducePlanMatchesDesc(const SegmentedReduceDesc &desc,
                               const SegmentedReducePlan &plan) noexcept {
  const SegmentedReducePlan expected = PlanSegmentedReduce(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.head_bytes == expected.head_bytes &&
         plan.block_size == expected.block_size &&
         plan.block_count == expected.block_count &&
         plan.pass_count == expected.pass_count &&
         plan.temp_value_bytes == expected.temp_value_bytes &&
         plan.temp_head_bytes == expected.temp_head_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes;
}

} // namespace rund::kernel
