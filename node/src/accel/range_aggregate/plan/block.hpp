#pragma once

#include "../model/block.hpp"
#include "../model/candidate.hpp"
#include "../model/shape.hpp"
#include "evaluation.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <optional>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildBlockPrefixSuffix(const RangeShape &shape, const RangeCaps &capabilities,
                       const RangeCandidate candidate,
                       bool &overflow) noexcept {
  if (!shape.traits().associative() || !shape.traits().has_identity() ||
      !shape.traits().idempotent() || !shape.traits().ordered()) {
    return std::nullopt;
  }
  rund::kernel::u64 padded = 0u;
  const std::optional<rund::kernel::u64> span = shape.affine_span();
  if (!span.has_value()) {
    overflow = true;
    return std::nullopt;
  }
  padded = *span;
  if (padded > capabilities.maximum_storage_element_count()) {
    return std::nullopt;
  }
  const rund::kernel::u64 window = shape.window_size();
  const rund::kernel::u64 blocks =
      padded / window + static_cast<rund::kernel::u64>(padded % window != 0u);
  const bool parallel = capabilities.source_variant() == RangeSource::Metal;
  const rund::kernel::u64 prepare_groups =
      capabilities.cpu_only() ? 1u
                              : RangeBlockGroups(capabilities.source_variant(),
                                                 blocks, candidate.width());
  const rund::kernel::u64 output_groups =
      RangeBlockQueryGroups(capabilities.source_variant(), shape.output_count(),
                            shape.stride(), window, candidate.width());
  if (!FitsGroups(capabilities, prepare_groups) ||
      !FitsGroups(capabilities, output_groups)) {
    return std::nullopt;
  }
  rund::kernel::u64 value_bytes = 0u;
  if (!rund::kernel::checked::mul(padded, shape.element_bytes(), value_bytes)) {
    overflow = true;
    return std::nullopt;
  }

  CandidateEvaluation evaluation{candidate};
  if (!AppendTemporary(evaluation, RangeTempRole::ForwardValues, 0u,
                       value_bytes, shape.element_bytes(), 0u, 1u, capabilities,
                       overflow) ||
      (!parallel && !AppendTemporary(evaluation, RangeTempRole::BackwardValues,
                                     0u, value_bytes, shape.element_bytes(), 0u,
                                     1u, capabilities, overflow))) {
    return std::nullopt;
  }

  const bool cpu = capabilities.source_variant() == RangeSource::Cpu;
  const rund::kernel::u64 execution_prepare_groups = cpu ? 1u : prepare_groups;
  const rund::kernel::u64 execution_output_groups = cpu ? 1u : output_groups;
  const rund::kernel::u32 execution_width = cpu ? 0u : candidate.width();
  const rund::kernel::u64 prepared_input_elements =
      shape.boundary() == RangeBoundary::Clamp
          ? padded
          : std::min(shape.input_count(), padded - shape.padding());
  const rund::kernel::u64 query_span = parallel ? output_groups * window : 0u;
  const rund::kernel::u64 query_input_elements =
      shape.boundary() == RangeBoundary::Clamp ? query_span
      : query_span > shape.padding()
          ? std::min(shape.input_count(), query_span - shape.padding())
          : 0u;
  if (!AppendStage(evaluation, capabilities, RangeStageKind::BlockPrefixSuffix,
                   0u, padded, execution_prepare_groups, execution_width,
                   overflow) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::BlockWindow, 0u,
                   shape.output_count(), execution_output_groups,
                   execution_width, overflow) ||
      !AccumulateBytes(
          evaluation.cost.global_read_bytes,
          static_cast<rund::kernel::u128>(prepared_input_elements) *
                  (parallel ? 1u : 2u) +
              static_cast<rund::kernel::u128>(query_input_elements),
          shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes,
                       static_cast<rund::kernel::u128>(shape.output_count()) *
                           (parallel ? 1u : 2u),
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes,
                       static_cast<rund::kernel::u128>(padded) *
                           (parallel ? 1u : 2u),
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes, shape.output_count(),
                       shape.element_bytes()) ||
      !AccumulateProduct(evaluation.cost.combine_ops,
                         parallel ? padded : padded - blocks, 2u) ||
      !Accumulate(evaluation.cost.combine_ops, shape.output_count())) {
    overflow = true;
    return std::nullopt;
  }
  if (parallel) {
    rund::kernel::u64 levels = 0u;
    for (auto width = candidate.width(); width > 1u; width /= 2u) {
      ++levels;
    }
    if (!AccumulateProduct(evaluation.cost.combine_ops, padded, 2u * levels)) {
      overflow = true;
      return std::nullopt;
    }
  }
  if (cpu &&
      (!AccumulateProduct(evaluation.cost.launched_lanes, padded, 2u) ||
       !Accumulate(evaluation.cost.launched_lanes, shape.output_count()))) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
