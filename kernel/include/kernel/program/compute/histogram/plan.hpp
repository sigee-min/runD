#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/histogram/model.hpp>

namespace rund::kernel {
namespace histogram_plan_detail {

[[nodiscard]] constexpr u64 IndexBytes(const HistogramIndex index) noexcept {
  switch (index) {
  case HistogramIndex::U32:
    return 4u;
  }
  return 0u;
}

[[nodiscard]] constexpr u64 CountBytes(const HistogramCount count) noexcept {
  switch (count) {
  case HistogramCount::U32:
    return 4u;
  }
  return 0u;
}

[[nodiscard]] constexpr HistogramPlan
Reject(const HistogramDesc &desc, const u64 index_bytes, const u64 count_bytes,
       const char *const reason) noexcept {
  return HistogramPlan{
      .index = desc.index,
      .count = desc.count,
      .element_count = desc.element_count,
      .bin_count = desc.bin_count,
      .index_bytes = index_bytes,
      .count_bytes = count_bytes,
      .reason = reason,
  };
}

} // namespace histogram_plan_detail

[[nodiscard]] constexpr HistogramPlan
PlanHistogram(const HistogramDesc &desc) noexcept {
  const u64 index_bytes = histogram_plan_detail::IndexBytes(desc.index);
  if (index_bytes == 0u) {
    return histogram_plan_detail::Reject(desc, 0u, 0u,
                                         "compute_histogram_index_unsupported");
  }
  const u64 count_bytes = histogram_plan_detail::CountBytes(desc.count);
  if (count_bytes == 0u) {
    return histogram_plan_detail::Reject(desc, index_bytes, 0u,
                                         "compute_histogram_count_unsupported");
  }
  if (desc.element_count == 0u) {
    return histogram_plan_detail::Reject(desc, index_bytes, count_bytes,
                                         "compute_histogram_count_zero");
  }
  if (desc.bin_count == 0u) {
    return histogram_plan_detail::Reject(desc, index_bytes, count_bytes,
                                         "compute_histogram_bin_count_zero");
  }
  if (desc.count == HistogramCount::U32 &&
      desc.element_count > static_cast<u64>(~u32{0u})) {
    return histogram_plan_detail::Reject(desc, index_bytes, count_bytes,
                                         "compute_histogram_count_overflow");
  }
  if (!checked::mul(desc.element_count, index_bytes) ||
      !checked::mul(desc.bin_count, count_bytes)) {
    return histogram_plan_detail::Reject(desc, index_bytes, count_bytes,
                                         "compute_histogram_bytes_overflow");
  }

  return HistogramPlan{
      .index = desc.index,
      .count = desc.count,
      .element_count = desc.element_count,
      .bin_count = desc.bin_count,
      .index_bytes = index_bytes,
      .count_bytes = count_bytes,
      .input_bytes = desc.element_count * index_bytes,
      .output_bytes = desc.bin_count * count_bytes,
      .status_bytes = 4u,
      .temp_bytes = 4u,
      .pass_count = 2u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
HistogramPlanMatchesDesc(const HistogramDesc &desc,
                         const HistogramPlan &plan) noexcept {
  const HistogramPlan expected = PlanHistogram(desc);
  return expected.ok && plan.ok && plan.index == expected.index &&
         plan.count == expected.count &&
         plan.element_count == expected.element_count &&
         plan.bin_count == expected.bin_count &&
         plan.index_bytes == expected.index_bytes &&
         plan.count_bytes == expected.count_bytes &&
         plan.input_bytes == expected.input_bytes &&
         plan.output_bytes == expected.output_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
