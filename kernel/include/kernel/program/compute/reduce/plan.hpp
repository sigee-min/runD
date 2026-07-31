#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/operation.hpp>

namespace rund::kernel {
namespace reduce_plan_detail {

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

[[nodiscard]] constexpr ReducePlan
Reject(const ReduceDesc &desc, const u64 element_bytes,
       const u64 items_per_thread, const u64 first_pass_group_count,
       const u64 pass_count, const u64 partial_element_count,
       const u64 partial_element_bytes, const u64 partial_bytes,
       const u64 temp_bytes, const char *const reason) noexcept {
  return ReducePlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .block_size = desc.block_size,
      .items_per_thread = items_per_thread,
      .first_pass_group_count = first_pass_group_count,
      .pass_count = pass_count,
      .partial_element_count = partial_element_count,
      .partial_element_bytes = partial_element_bytes,
      .partial_bytes = partial_bytes,
      .status_bytes = kStatusBytes,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .reason = reason,
  };
}

[[nodiscard]] constexpr bool BlockSizeAdmitted(const u64 block_size,
                                               const u64 count) noexcept {
  return block_size != 0u && !(block_size == 1u && count > 1u);
}

} // namespace reduce_plan_detail

[[nodiscard]] constexpr u64 ReduceGroupCount(const u64 element_count,
                                             const u64 group_width) noexcept {
  return checked::ceil(element_count, group_width);
}

[[nodiscard]] constexpr ReducePlan PlanReduce(const ReduceDesc &desc) noexcept {
  if (!reduce::valid(desc.op)) {
    return reduce_plan_detail::Reject(desc, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                      "compute_reduce_op_unsupported");
  }
  if (desc.count_source != ComputeCountSource::Descriptor &&
      ComputeCountBytes(desc.count_source) == 0u) {
    return reduce_plan_detail::Reject(
        desc, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        "compute_reduce_count_source_unsupported");
  }
  const u64 element_bytes = reduce_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return reduce_plan_detail::Reject(desc, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                      "compute_reduce_element_unsupported");
  }
  const bool wide = ReduceUsesWidePartials(desc.op);
  const u64 items_per_thread = wide ? kReduceItemsPerThread : 1u;
  const u64 partial_element_bytes =
      wide ? kReduceWideElementBytes : element_bytes;
  if (desc.element_count == 0u) {
    return reduce_plan_detail::Reject(desc, element_bytes, items_per_thread, 0u,
                                      0u, 0u, partial_element_bytes, 0u, 0u,
                                      "compute_reduce_count_zero");
  }
  if (!reduce_plan_detail::BlockSizeAdmitted(desc.block_size,
                                             desc.element_count)) {
    return reduce_plan_detail::Reject(desc, element_bytes, items_per_thread, 0u,
                                      0u, 0u, partial_element_bytes, 0u, 0u,
                                      "compute_reduce_block_invalid");
  }

  if (!checked::mul(desc.block_size, items_per_thread)) {
    return reduce_plan_detail::Reject(desc, element_bytes, items_per_thread, 0u,
                                      0u, 0u, partial_element_bytes, 0u, 0u,
                                      "compute_reduce_temp_overflow");
  }
  const u64 first_pass_width = desc.block_size * items_per_thread;
  u64 first_pass_group_count =
      ReduceGroupCount(desc.element_count, first_pass_width);
  if (wide && first_pass_group_count > kReduceFirstPassMaxGroups) {
    first_pass_group_count = kReduceFirstPassMaxGroups;
  }

  u64 current_count = desc.element_count;
  u64 pass_count = 0u;
  u64 partial_element_count = 0u;
  if (wide) {
    pass_count = first_pass_group_count > 1u ? 2u : 1u;
    partial_element_count =
        first_pass_group_count > 1u ? first_pass_group_count : 0u;
  } else {
    while (current_count > 1u) {
      const u64 next_count = ReduceGroupCount(current_count, desc.block_size);
      ++pass_count;
      if (next_count > 1u) {
        if (!checked::add(partial_element_count, next_count)) {
          return reduce_plan_detail::Reject(
              desc, element_bytes, items_per_thread, first_pass_group_count,
              pass_count, partial_element_count, partial_element_bytes, 0u, 0u,
              "compute_reduce_temp_overflow");
        }
        partial_element_count += next_count;
      }
      current_count = next_count;
    }
    if (pass_count == 0u) {
      pass_count = 1u;
    }
  }

  if (!checked::mul(partial_element_count, partial_element_bytes)) {
    return reduce_plan_detail::Reject(
        desc, element_bytes, items_per_thread, first_pass_group_count,
        pass_count, partial_element_count, partial_element_bytes, 0u, 0u,
        "compute_reduce_temp_overflow");
  }
  const u64 partial_bytes = partial_element_count * partial_element_bytes;
  if (!checked::add(partial_bytes, reduce_plan_detail::kStatusBytes)) {
    return reduce_plan_detail::Reject(
        desc, element_bytes, items_per_thread, first_pass_group_count,
        pass_count, partial_element_count, partial_element_bytes, partial_bytes,
        0u, "compute_reduce_temp_overflow");
  }
  const u64 temp_bytes = partial_bytes + reduce_plan_detail::kStatusBytes;

  return ReducePlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .block_size = desc.block_size,
      .items_per_thread = items_per_thread,
      .first_pass_group_count = first_pass_group_count,
      .pass_count = pass_count,
      .partial_element_count = partial_element_count,
      .partial_element_bytes = partial_element_bytes,
      .partial_bytes = partial_bytes,
      .status_bytes = reduce_plan_detail::kStatusBytes,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
ReducePlanMatchesDesc(const ReduceDesc &desc, const ReducePlan &plan) noexcept {
  const ReducePlan expected = PlanReduce(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.block_size == expected.block_size &&
         plan.items_per_thread == expected.items_per_thread &&
         plan.first_pass_group_count == expected.first_pass_group_count &&
         plan.pass_count == expected.pass_count &&
         plan.partial_element_count == expected.partial_element_count &&
         plan.partial_element_bytes == expected.partial_element_bytes &&
         plan.partial_bytes == expected.partial_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.count_source == expected.count_source;
}

} // namespace rund::kernel
