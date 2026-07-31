#pragma once

#include <kernel/program/compute/scatter/plan.hpp>

namespace rund::kernel {
namespace scatter_reference_detail {

[[nodiscard]] constexpr ScatterResult
Reject(const u64 element_count, const u64 output_count,
       const u64 first_rejected_index, const char *const reason) noexcept {
  return ScatterResult{
      .element_count = element_count,
      .output_count = output_count,
      .first_rejected_index = first_rejected_index,
      .reason = reason,
  };
}

template <typename Element>
[[nodiscard]] inline ScatterResult
ReferenceScatter(const Element *const values, const u32 *const indices,
                 Element *const output, const u64 element_count,
                 const u64 output_count) noexcept {
  if (element_count == 0u) {
    return Reject(0u, output_count, 0u, "compute_scatter_count_zero");
  }
  if (element_count > scatter_plan_detail::kMaxEncodedCount) {
    return Reject(element_count, output_count, 0u,
                  "compute_scatter_count_unsupported");
  }
  if (output_count == 0u) {
    return Reject(element_count, 0u, 0u, "compute_scatter_output_zero");
  }
  if (output_count > scatter_plan_detail::kMaxOutputCount) {
    return Reject(element_count, output_count, 0u,
                  "compute_scatter_output_unsupported");
  }
  if (values == nullptr || indices == nullptr || output == nullptr) {
    return Reject(element_count, output_count, 0u,
                  "compute_scatter_buffer_invalid");
  }

  for (u64 index = 0u; index < element_count; ++index) {
    const u32 target = indices[index];
    if (static_cast<u64>(target) >= output_count) {
      return Reject(element_count, output_count, index,
                    "compute_scatter_index_out_of_range");
    }
    for (u64 prior = 0u; prior < index; ++prior) {
      if (indices[prior] == target) {
        return Reject(element_count, output_count, index,
                      "compute_scatter_duplicate_index");
      }
    }
  }

  for (u64 index = 0u; index < element_count; ++index) {
    output[indices[index]] = values[index];
  }
  return ScatterResult{
      .element_count = element_count,
      .output_count = output_count,
      .first_rejected_index = element_count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace scatter_reference_detail

[[nodiscard]] inline ScatterResult
ReferenceScatterU32(const u32 *const values, const u32 *const indices,
                    u32 *const output, const u64 element_count,
                    const u64 output_count) noexcept {
  return scatter_reference_detail::ReferenceScatter(
      values, indices, output, element_count, output_count);
}

[[nodiscard]] inline ScatterResult
ReferenceScatterU64(const u64 *const values, const u32 *const indices,
                    u64 *const output, const u64 element_count,
                    const u64 output_count) noexcept {
  return scatter_reference_detail::ReferenceScatter(
      values, indices, output, element_count, output_count);
}

} // namespace rund::kernel
