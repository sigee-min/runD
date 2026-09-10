#pragma once

#include "../model/candidate.hpp"
#include "../model/shape.hpp"
#include "evaluation.hpp"
#include "traffic.hpp"

#include <optional>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildDirect(const RangeShape &shape, const RangeCaps &capabilities,
            const RangeCandidate candidate, bool &overflow) noexcept {
  CandidateEvaluation evaluation{candidate};
  const std::optional<BoundaryTraffic> traffic = AffineBoundaryTraffic(shape);
  if (!traffic.has_value()) {
    overflow = true;
    return std::nullopt;
  }
  const rund::kernel::u128 read_elements =
      shape.boundary() == RangeBoundary::Clamp
          ? static_cast<rund::kernel::u128>(shape.output_count()) *
                shape.window_size()
          : traffic->valid_samples;
  if (!AccumulateBytes(evaluation.cost.global_read_bytes, read_elements,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes, shape.output_count(),
                       shape.element_bytes()) ||
      !Add(evaluation.cost.combine_ops, read_elements - shape.output_count(),
           evaluation.cost.combine_ops)) {
    overflow = true;
    return std::nullopt;
  }

  const bool cpu = candidate.width() == 0u;
  const rund::kernel::u64 groups =
      cpu ? 1u : Groups(shape.output_count(), candidate.width());
  if (!AppendStage(evaluation, capabilities, RangeStageKind::Direct, 0u,
                   shape.output_count(), groups, candidate.width(), overflow)) {
    return std::nullopt;
  }
  if (cpu) {
    evaluation.cost.launched_lanes = shape.output_count();
  }
  return evaluation;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
