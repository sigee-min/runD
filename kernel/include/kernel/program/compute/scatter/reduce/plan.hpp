#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>

namespace rund::kernel {
namespace scatter_reduce_plan_detail {

inline constexpr u64 kStatusBytes = 4u * sizeof(u32);
inline constexpr u64 kIndirectBytes = 6u * sizeof(u32);

[[nodiscard]] constexpr u64 ElementBytes(const ComputeDomain domain) noexcept {
  switch (domain) {
  case ComputeDomain::I32:
  case ComputeDomain::U32:
    return 4u;
  case ComputeDomain::I64:
  case ComputeDomain::U64:
    return 8u;
  case ComputeDomain::Fixed:
    return 0u;
  }
  return 0u;
}

[[nodiscard]] constexpr u64
ElementBytes(const ComputeDomain domain,
             const ComputeFixedFormat format) noexcept {
  if (domain != ComputeDomain::Fixed) {
    return ElementBytes(domain);
  }
  const u32 width = static_cast<u32>(format.integer_bits) +
                    static_cast<u32>(format.fraction_bits);
  return width == 32u ? 4u : (width == 64u ? 8u : 0u);
}

[[nodiscard]] constexpr ScatterReducePlan
Reject(const ScatterReduceDesc &desc, const u64 element_bytes,
       const char *const reason) noexcept {
  return ScatterReducePlan{
      .op = desc.op,
      .domain = desc.domain,
      .fixed_format = desc.fixed_format,
      .element_count = desc.element_count,
      .output_count = desc.output_count,
      .element_bytes = element_bytes,
      .count_source = desc.count_source,
      .reason = reason,
  };
}

} // namespace scatter_reduce_plan_detail

[[nodiscard]] constexpr ScatterReducePlan
PlanScatterReduce(const ScatterReduceDesc &desc) noexcept {
  if (desc.op != ScatterReduceOp::Sum && desc.op != ScatterReduceOp::Min &&
      desc.op != ScatterReduceOp::Max) {
    return scatter_reduce_plan_detail::Reject(
        desc, 0u, "compute_scatter_reduce_op_unsupported");
  }
  const u64 element_bytes =
      scatter_reduce_plan_detail::ElementBytes(desc.domain, desc.fixed_format);
  if (element_bytes == 0u) {
    return scatter_reduce_plan_detail::Reject(
        desc, 0u, "compute_scatter_reduce_domain_unsupported");
  }
  if (desc.domain == ComputeDomain::Fixed) {
    if (!ComputePrimitiveFixedFormatValid(static_cast<u32>(element_bytes),
                                          desc.fixed_format,
                                          desc.fixed_format.approximation)) {
      return scatter_reduce_plan_detail::Reject(
          desc, element_bytes, "compute_scatter_reduce_fixed_invalid");
    }
  } else if (!ComputeFixedFormatAbsent(desc.fixed_format)) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_fixed_unexpected");
  }
  if (desc.element_count == 0u) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_count_zero");
  }
  if (desc.element_count > ~u32{0u}) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_count_unsupported");
  }
  if (desc.output_count == 0u || desc.output_count > ~u32{0u}) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes,
        desc.output_count == 0u ? "compute_scatter_reduce_output_zero"
                                : "compute_scatter_reduce_output_unsupported");
  }
  if (desc.count_source != ComputeCountSource::Descriptor &&
      ComputeCountBytes(desc.count_source) == 0u) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_count_source_unsupported");
  }
  constexpr u64 index_bytes = 4u;
  if (!checked::mul(desc.output_count, sizeof(u32))) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_temp_overflow");
  }
  // GPU execution uses one O(output_count) contributor table. A parallel
  // identity pass initializes outputs/counts, then one source-ordinal fold
  // preserves non-associative fixed saturation exactly. No key/value radix
  // copies or source-sized device scratch are required.
  constexpr u64 sorted_index_bytes = 0u;
  constexpr u64 sorted_value_bytes = 0u;
  const u64 segment_bytes = desc.output_count * sizeof(u32);
  if (!checked::add(segment_bytes, scatter_reduce_plan_detail::kStatusBytes) ||
      !checked::add(segment_bytes + scatter_reduce_plan_detail::kStatusBytes,
                    scatter_reduce_plan_detail::kIndirectBytes)) {
    return scatter_reduce_plan_detail::Reject(
        desc, element_bytes, "compute_scatter_reduce_temp_overflow");
  }
  const u64 temp_bytes = segment_bytes +
                         scatter_reduce_plan_detail::kStatusBytes +
                         scatter_reduce_plan_detail::kIndirectBytes;
  const u32 radix_pass_count = 0u;
  // Control/preflight, parallel initialization, source-ordinal fold.
  const u32 fold_pass_count = 1u;
  return ScatterReducePlan{
      .op = desc.op,
      .domain = desc.domain,
      .fixed_format = desc.fixed_format,
      .element_count = desc.element_count,
      .output_count = desc.output_count,
      .element_bytes = element_bytes,
      .index_bytes = index_bytes,
      .sorted_index_bytes = sorted_index_bytes,
      .sorted_value_bytes = sorted_value_bytes,
      .segment_bytes = segment_bytes,
      .status_bytes = scatter_reduce_plan_detail::kStatusBytes,
      .indirect_bytes = scatter_reduce_plan_detail::kIndirectBytes,
      .temp_bytes = temp_bytes,
      .radix_pass_count = radix_pass_count,
      .fold_pass_count = fold_pass_count,
      .pass_count = 3u,
      .count_source = desc.count_source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
ScatterReducePlanMatchesDesc(const ScatterReduceDesc &desc,
                             const ScatterReducePlan &plan) noexcept {
  const ScatterReducePlan expected = PlanScatterReduce(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.domain == expected.domain &&
         plan.fixed_format == expected.fixed_format &&
         plan.element_count == expected.element_count &&
         plan.output_count == expected.output_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.index_bytes == expected.index_bytes &&
         plan.sorted_index_bytes == expected.sorted_index_bytes &&
         plan.sorted_value_bytes == expected.sorted_value_bytes &&
         plan.segment_bytes == expected.segment_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.indirect_bytes == expected.indirect_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.radix_pass_count == expected.radix_pass_count &&
         plan.fold_pass_count == expected.fold_pass_count &&
         plan.pass_count == expected.pass_count &&
         plan.count_source == expected.count_source;
}

[[nodiscard]] constexpr bool
ScatterReduceFoldParallel(const ScatterReducePlan &plan) noexcept {
  const bool saturating_fixed_sum =
      plan.op == ScatterReduceOp::Sum && plan.domain == ComputeDomain::Fixed &&
      plan.fixed_format.overflow == ComputeOverflow::Saturate;
  return plan.element_bytes == 4u && !saturating_fixed_sum;
}

} // namespace rund::kernel
