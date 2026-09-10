#pragma once

#include "../model/candidate.hpp"
#include "../model/prefix.hpp"
#include "../model/shape.hpp"
#include "budget.hpp"
#include "evaluation.hpp"
#include "traffic.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildPrefixDifference(const RangeShape &shape, const RangeCaps &capabilities,
                      const RangeCandidate candidate, bool &overflow) noexcept {
  if (!shape.traits().associative() || !shape.traits().has_identity() ||
      !shape.traits().invertible()) {
    return std::nullopt;
  }
  CandidateEvaluation evaluation{candidate};
  if (capabilities.cpu_only()) {
    const std::optional<BoundaryTraffic> traffic = AffineBoundaryTraffic(shape);
    if (!traffic.has_value() ||
        !AppendTemporary(evaluation, RangeTempRole::PrefixValues, 0u,
                         shape.payload_bytes(), shape.element_bytes(), 0u, 1u,
                         capabilities, overflow) ||
        !AppendStage(evaluation, capabilities, RangeStageKind::PrefixSequential,
                     0u, shape.input_count(), 1u, 0u, overflow) ||
        !AppendStage(evaluation, capabilities, RangeStageKind::PrefixWindow, 0u,
                     shape.output_count(), 1u, 0u, overflow)) {
      return std::nullopt;
    }
    const rund::kernel::u128 endpoint_reads =
        shape.boundary() == RangeBoundary::Clamp
            ? static_cast<rund::kernel::u128>(traffic->left_affected) +
                  traffic->right_affected
            : 0u;
    rund::kernel::u128 output_reads = 0u;
    if (!Add(shape.output_count(), traffic->left_prefix_reads, output_reads) ||
        !Add(output_reads, endpoint_reads, output_reads) ||
        !AccumulateBytes(evaluation.cost.global_read_bytes, shape.input_count(),
                         shape.element_bytes()) ||
        !AccumulateBytes(evaluation.cost.global_read_bytes, output_reads,
                         shape.element_bytes()) ||
        !AccumulateBytes(evaluation.cost.global_write_bytes,
                         shape.input_count(), shape.element_bytes()) ||
        !AccumulateBytes(evaluation.cost.global_write_bytes,
                         shape.output_count(), shape.element_bytes()) ||
        !Accumulate(evaluation.cost.combine_ops, shape.input_count() - 1u) ||
        !Accumulate(evaluation.cost.combine_ops, endpoint_reads) ||
        !Accumulate(evaluation.cost.inverse_ops, traffic->left_prefix_reads) ||
        !Accumulate(evaluation.cost.scale_ops, endpoint_reads) ||
        !Add(shape.input_count(), shape.output_count(),
             evaluation.cost.launched_lanes)) {
      overflow = true;
      return std::nullopt;
    }
    return evaluation;
  }
  const rund::kernel::u64 local_shared =
      static_cast<rund::kernel::u64>(candidate.width()) *
      shape.element_bytes();
  if (!SharedBudgetFits(capabilities, local_shared)) {
    return std::nullopt;
  }
  evaluation.cost.shared_bytes = local_shared;
  const RangePrefixExec hierarchy = PlanRangePrefixTree(
      shape.input_count(), candidate.width(), shape.element_bytes(),
      capabilities.maximum_group_count());
  if (!hierarchy.ok() || hierarchy.stage_count() == 0u ||
      hierarchy.stage_count() >= std::numeric_limits<std::uint8_t>::max()) {
    return std::nullopt;
  }
  // The output query applies the level-zero block offset on demand.
  const std::size_t query_stage =
      hierarchy.stage_count() -
      static_cast<std::size_t>(shape.input_count() > candidate.width());
  if (!AppendTemporary(evaluation, RangeTempRole::PrefixValues, 0u,
                       shape.payload_bytes(), shape.element_bytes(), 0u,
                       static_cast<std::uint8_t>(query_stage), capabilities,
                       overflow)) {
    return std::nullopt;
  }

  for (std::size_t index = 0u; index < hierarchy.stage_count(); ++index) {
    const RangeStagePlan stage = hierarchy.stage(index);
    if (stage.disposition == RangeStageKind::PrefixFixup && stage.level == 0u) {
      continue;
    }
    if (!AppendStage(evaluation, capabilities, stage.disposition, stage.level,
                     stage.element_count, stage.groups, stage.width,
                     overflow)) {
      return std::nullopt;
    }
    if (stage.disposition == RangeStageKind::PrefixBlock ||
        stage.disposition == RangeStageKind::PrefixSummary) {
      if (!AccumulateBytes(evaluation.cost.global_read_bytes,
                           stage.element_count, shape.element_bytes()) ||
          !AccumulateBytes(evaluation.cost.global_write_bytes,
                           stage.element_count, shape.element_bytes()) ||
          !AccumulateProduct(evaluation.cost.combine_ops, stage.groups,
                             2u * (candidate.width() - 1u))) {
        overflow = true;
        return std::nullopt;
      }
      if (stage.groups > 1u) {
        bool found = false;
        for (std::size_t temporary_index = 0u;
             temporary_index < hierarchy.temporary_count(); ++temporary_index) {
          const RangeTempReq temporary = hierarchy.temporary(temporary_index);
          if (temporary.role != RangeTempRole::BlockSummaries ||
              temporary.ordinal != stage.level) {
            continue;
          }
          found = AppendTemporary(evaluation, temporary.role, temporary.ordinal,
                                  temporary.bytes, temporary.alignment,
                                  temporary.first_stage, temporary.last_stage,
                                  capabilities, overflow) &&
                  AccumulateBytes(evaluation.cost.global_write_bytes,
                                  stage.groups, shape.element_bytes());
          break;
        }
        if (!found) {
          overflow = true;
          return std::nullopt;
        }
      }
      continue;
    }
    if (stage.disposition != RangeStageKind::PrefixFixup) {
      overflow = true;
      return std::nullopt;
    }
    const rund::kernel::u64 adjusted =
        stage.element_count -
        std::min<rund::kernel::u64>(stage.element_count, candidate.width());
    if (!AccumulateBytes(evaluation.cost.global_read_bytes,
                         static_cast<rund::kernel::u128>(adjusted) * 2u,
                         shape.element_bytes()) ||
        !AccumulateBytes(evaluation.cost.global_write_bytes, adjusted,
                         shape.element_bytes()) ||
        !Accumulate(evaluation.cost.combine_ops, adjusted)) {
      overflow = true;
      return std::nullopt;
    }
  }

  const rund::kernel::u64 output_groups =
      Groups(shape.output_count(), candidate.width());
  const std::optional<BoundaryTraffic> traffic = AffineBoundaryTraffic(shape);
  if (!traffic.has_value()) {
    overflow = true;
    return std::nullopt;
  }
  const rund::kernel::u64 left_prefix_reads = traffic->left_prefix_reads;
  // Number of queried endpoints outside block zero. Compute anchor bounds
  // in u128 so padding + width cannot wrap for admitted u64 affine shapes.
  rund::kernel::u64 right_offsets = 0u;
  rund::kernel::u64 left_offsets = 0u;
  if (shape.input_count() > candidate.width()) {
    const rund::kernel::u128 width = candidate.width();
    const rund::kernel::u128 extent = shape.window_size() - shape.padding();
    const rund::kernel::u128 right_first =
        extent > width
            ? 0u
            : (width + 1u - extent + shape.stride() - 1u) / shape.stride();
    const rund::kernel::u128 left_first =
        (static_cast<rund::kernel::u128>(shape.padding()) + width) /
            shape.stride() +
        1u;
    right_offsets = shape.output_count() -
                    static_cast<rund::kernel::u64>(std::min<rund::kernel::u128>(
                        shape.output_count(), right_first));
    left_offsets = shape.output_count() -
                   static_cast<rund::kernel::u64>(std::min<rund::kernel::u128>(
                       shape.output_count(), left_first));
  }
  rund::kernel::u128 output_reads = 0u;
  const rund::kernel::u128 endpoint_reads =
      shape.boundary() == RangeBoundary::Clamp
          ? static_cast<rund::kernel::u128>(traffic->left_affected) +
                traffic->right_affected
          : 0u;
  if (!Add(shape.output_count(), left_prefix_reads, output_reads) ||
      !Accumulate(output_reads, right_offsets) ||
      !Accumulate(output_reads, left_offsets) ||
      !Accumulate(evaluation.cost.combine_ops, right_offsets) ||
      !Accumulate(evaluation.cost.inverse_ops, left_offsets) ||
      !Add(output_reads, endpoint_reads, output_reads) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::PrefixWindow, 0u,
                   shape.output_count(), output_groups, candidate.width(),
                   overflow) ||
      !AccumulateBytes(evaluation.cost.global_read_bytes, output_reads,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes, shape.output_count(),
                       shape.element_bytes()) ||
      !Accumulate(evaluation.cost.inverse_ops, left_prefix_reads) ||
      !Accumulate(evaluation.cost.scale_ops, endpoint_reads) ||
      !Accumulate(evaluation.cost.combine_ops, endpoint_reads)) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
