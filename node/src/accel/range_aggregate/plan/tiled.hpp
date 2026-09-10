#pragma once
#include "../model/candidate.hpp"
#include "budget.hpp"
#include "evaluation.hpp"
#include "traffic.hpp"

namespace rund::node::accel::detail::range_plan_detail {

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildTiledDifference(const RangeShape &shape, const RangeCaps &capabilities,
                     const RangeCandidate candidate, bool &overflow) noexcept {
  if (capabilities.cpu_only() || !shape.centered_clamp() ||
      !shape.traits().invertible()) {
    return std::nullopt;
  }
  CandidateEvaluation result{candidate};
  const rund::kernel::u64 tile = candidate.width() * kRangeTileOutputsPerLane;
  const rund::kernel::u64 groups = Groups(shape.output_count(), tile);
  const rund::kernel::u64 shared =
      (2u * candidate.width()) * shape.element_bytes();
  if (!FitsGroups(capabilities, groups) ||
      !SharedBudgetFits(capabilities, shared)) {
    return std::nullopt;
  }
  const auto anchors = RangeShape::affine(
      shape.traits(), RangeBoundary::Clamp, shape.input_count(), groups,
      shape.window_size(), tile, shape.padding(), shape.element_bytes());
  const auto traffic = anchors ? AffineBoundaryTraffic(*anchors) : std::nullopt;
  if (!traffic) {
    overflow = true;
    return std::nullopt;
  }
  const rund::kernel::u128 corrections =
      static_cast<rund::kernel::u128>(traffic->left_affected) +
      traffic->right_affected;
  const rund::kernel::u128 differences = shape.output_count() - groups;
  rund::kernel::u128 reads = traffic->valid_samples;
  result.cost.shared_bytes = shared;
  if (!Accumulate(reads, corrections) ||
      !AccumulateProduct(reads, differences, 2u) ||
      !AccumulateBytes(result.cost.global_read_bytes, reads,
                       shape.element_bytes()) ||
      !AccumulateBytes(result.cost.global_write_bytes, shape.output_count(),
                       shape.element_bytes()) ||
      !Accumulate(result.cost.combine_ops, traffic->valid_samples - groups) ||
      !Accumulate(result.cost.combine_ops, corrections) ||
      !AccumulateProduct(result.cost.combine_ops, groups,
                         2u * (candidate.width() - 1u)) ||
      !Accumulate(result.cost.combine_ops, differences) ||
      !Accumulate(result.cost.combine_ops, shape.output_count()) ||
      !Accumulate(result.cost.inverse_ops, differences) ||
      !Accumulate(result.cost.scale_ops, corrections) ||
      !AppendStage(result, capabilities, RangeStageKind::TiledDifference, 0u,
                   shape.output_count(), groups, candidate.width(), overflow)) {
    overflow = true;
    return std::nullopt;
  }
  return result;
}
} // namespace rund::node::accel::detail::range_plan_detail
