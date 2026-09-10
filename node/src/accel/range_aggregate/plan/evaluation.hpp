#pragma once

#include "../model/candidate.hpp"
#include "../model/capability.hpp"
#include "arithmetic.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {
namespace range_plan_detail {

struct CandidateEvaluation final {
  explicit constexpr CandidateEvaluation(
      const RangeCandidate candidate_value) noexcept
      : candidate(candidate_value) {}

  RangeCandidate candidate;
  RangeCost cost{};
  std::array<RangeStagePlan, kRangeStageCap> stages{};
  std::size_t stage_count{};
  std::array<RangeTempReq, kRangeTempCap> temporaries{};
  std::size_t temporary_count{};
};

[[nodiscard]] constexpr bool
FitsGroups(const RangeCaps &capabilities,
           const rund::kernel::u64 groups) noexcept {
  return groups != 0u && (capabilities.cpu_only() ||
                          groups <= capabilities.maximum_group_count());
}

[[nodiscard]] constexpr bool
AppendStage(CandidateEvaluation &evaluation, const RangeCaps &capabilities,
            const RangeStageKind disposition, const std::uint8_t level,
            const rund::kernel::u64 element_count,
            const rund::kernel::u64 groups, const rund::kernel::u32 width,
            bool &overflow) noexcept {
  if (evaluation.stage_count == evaluation.stages.size() ||
      element_count > capabilities.maximum_storage_element_count() ||
      !FitsGroups(capabilities, groups)) {
    return false;
  }
  evaluation.stages[evaluation.stage_count++] = RangeStagePlan{
      .disposition = disposition,
      .level = level,
      .element_count = element_count,
      .groups = groups,
      .width = width,
  };
  if (evaluation.cost.dispatch_count ==
      std::numeric_limits<rund::kernel::u64>::max()) {
    overflow = true;
    return false;
  }
  ++evaluation.cost.dispatch_count;
  if (!rund::kernel::checked::add(evaluation.cost.workgroup_count, groups,
                                  evaluation.cost.workgroup_count) ||
      (width != 0u &&
       !AccumulateProduct(evaluation.cost.launched_lanes, groups, width))) {
    overflow = true;
    return false;
  }
  return true;
}

[[nodiscard]] constexpr bool
AppendTemporary(CandidateEvaluation &evaluation, const RangeTempRole role,
                const std::uint8_t ordinal, const rund::kernel::u64 bytes,
                const rund::kernel::u64 alignment,
                const std::uint8_t first_stage, const std::uint8_t last_stage,
                const RangeCaps &capabilities, bool &overflow) noexcept {
  if (evaluation.temporary_count == evaluation.temporaries.size()) {
    return false;
  }
  if (bytes > capabilities.maximum_storage_binding_bytes()) {
    return false;
  }
  rund::kernel::u64 scratch = 0u;
  if (!rund::kernel::checked::add(evaluation.cost.scratch_bytes, bytes,
                                  scratch)) {
    overflow = true;
    return false;
  }
  evaluation.cost.scratch_bytes = scratch;
  evaluation.temporaries[evaluation.temporary_count++] =
      RangeTempReq{.role = role,
                   .ordinal = ordinal,
                   .bytes = bytes,
                   .alignment = alignment,
                   .first_stage = first_stage,
                   .last_stage = last_stage};
  return true;
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
