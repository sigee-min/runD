#pragma once

#include "../model/candidate.hpp"
#include "../model/shape.hpp"
#include "budget.hpp"
#include "evaluation.hpp"

#include <algorithm>
#include <optional>

namespace rund::node::accel::detail {
namespace range_plan_detail {

[[nodiscard]] constexpr std::optional<CandidateEvaluation>
BuildSharedHalo(const RangeShape &shape, const RangeCaps &capabilities,
                const RangeCandidate candidate, bool &overflow) noexcept {
  if (!shape.centered_clamp() ||
      candidate.radius_capacity() < shape.padding()) {
    return std::nullopt;
  }
  CandidateEvaluation evaluation{candidate};
  const rund::kernel::u64 groups =
      Groups(shape.output_count(), candidate.width());
  if (!FitsGroups(capabilities, groups)) {
    return std::nullopt;
  }
  const rund::kernel::u64 shared_elements =
      candidate.width() + 2u * candidate.radius_capacity();
  const rund::kernel::u64 shared_bytes =
      shared_elements * shape.element_bytes();
  if (!SharedBudgetFits(capabilities, shared_bytes)) {
    return std::nullopt;
  }
  evaluation.cost.shared_bytes = shared_bytes;

  rund::kernel::u128 read_elements = shape.input_count();
  if (groups > 1u) {
    rund::kernel::u128 group_term = 0u;
    rund::kernel::u128 halo = 0u;
    const rund::kernel::u64 tail =
        shape.input_count() - (groups - 1u) * candidate.width();
    if (!Multiply(groups, 2u, group_term) || group_term < 3u ||
        !Multiply(group_term - 3u, shape.padding(), halo) ||
        !Add(halo, std::min<rund::kernel::u64>(shape.padding(), tail), halo) ||
        !Add(read_elements, halo, read_elements)) {
      overflow = true;
      return std::nullopt;
    }
  }
  rund::kernel::u128 twice_radius = 0u;
  if (!AccumulateBytes(evaluation.cost.global_read_bytes, read_elements,
                       shape.element_bytes()) ||
      !AccumulateBytes(evaluation.cost.global_write_bytes, shape.output_count(),
                       shape.element_bytes()) ||
      !Multiply(shape.padding(), 2u, twice_radius) ||
      !Multiply(shape.output_count(), twice_radius,
                evaluation.cost.combine_ops) ||
      !AppendStage(evaluation, capabilities, RangeStageKind::SharedHalo, 0u,
                   shape.output_count(), groups, candidate.width(), overflow)) {
    overflow = true;
    return std::nullopt;
  }
  return evaluation;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
