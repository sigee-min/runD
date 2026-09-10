#pragma once

#include "../model/plan.hpp"
#include "arithmetic.hpp"
#include "block.hpp"
#include "direct.hpp"
#include "identity.hpp"
#include "order.hpp"
#include "prefix.hpp"
#include "shared.hpp"
#include "tiled.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr RangePlan
BuildRangePlan(const RangeShape &shape,
               const RangeCaps &capabilities) noexcept {
  using namespace range_plan_detail;
  if (!shape.valid()) {
    return RangePlan::rejected("compute_range_aggregate_shape_invalid");
  }
  if (!capabilities.valid()) {
    return RangePlan::rejected(
        capabilities.available()
            ? "compute_range_aggregate_capabilities_invalid"
            : "compute_range_aggregate_unavailable");
  }
  if (shape.input_count() > capabilities.maximum_storage_element_count() ||
      shape.output_count() > capabilities.maximum_storage_element_count()) {
    return RangePlan::rejected("compute_range_aggregate_candidate_unavailable");
  }

  std::array<std::optional<CandidateEvaluation>, kRangeCandidateCap>
      evaluations{};
  std::size_t evaluation_count = 0u;
  bool saw_overflow = false;
  const auto append = [&](std::optional<CandidateEvaluation> evaluation) {
    if (evaluation.has_value() && evaluation_count < evaluations.size()) {
      evaluations[evaluation_count++] = std::move(evaluation);
    }
  };

  if (capabilities.cpu_only()) {
    if (capabilities.supports(RangeSupport::Direct)) {
      append(BuildDirect(shape, capabilities, RangeCandidate::direct_cpu(),
                         saw_overflow));
    }
    if (capabilities.supports(RangeSupport::PrefixDifference) &&
        shape.traits().invertible()) {
      const std::optional<RangeCandidate> candidate =
          RangeCandidate::prefix_difference(kRangeCpuBlockWidth);
      if (candidate.has_value()) {
        append(BuildPrefixDifference(shape, capabilities, *candidate,
                                     saw_overflow));
      }
    }
    if (capabilities.supports(RangeSupport::BlockPrefixSuffix) &&
        shape.traits().idempotent() && shape.traits().ordered()) {
      const std::optional<RangeCandidate> candidate =
          RangeCandidate::block_prefix_suffix(kRangeCpuBlockWidth);
      if (candidate.has_value()) {
        append(BuildBlockPrefixSuffix(shape, capabilities, *candidate,
                                      saw_overflow));
      }
    }
  } else {
    for (const rund::kernel::u32 width : kRangeWidths) {
      if (!capabilities.supports_width(width)) {
        continue;
      }
      if (capabilities.supports(RangeSupport::Direct)) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::direct_gpu(width);
        if (candidate.has_value()) {
          append(BuildDirect(shape, capabilities, *candidate, saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::SharedHalo)) {
        const rund::kernel::u32 capacity =
            SharedRadiusCapacity(capabilities, width, shape.element_bytes());
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::shared_halo(width, capacity);
        if (candidate.has_value()) {
          append(
              BuildSharedHalo(shape, capabilities, *candidate, saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::TiledDifference)) {
        const auto candidate = RangeCandidate::tiled_difference(width);
        if (candidate) {
          append(BuildTiledDifference(shape, capabilities, *candidate,
                                      saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::PrefixDifference) &&
          shape.traits().invertible()) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::prefix_difference(width);
        if (candidate.has_value()) {
          append(BuildPrefixDifference(shape, capabilities, *candidate,
                                       saw_overflow));
        }
      }
      if (capabilities.supports(RangeSupport::BlockPrefixSuffix) &&
          shape.traits().idempotent() && shape.traits().ordered()) {
        const std::optional<RangeCandidate> candidate =
            RangeCandidate::block_prefix_suffix(width);
        if (candidate.has_value()) {
          append(BuildBlockPrefixSuffix(shape, capabilities, *candidate,
                                        saw_overflow));
        }
      }
    }
  }

  if (evaluation_count == 0u) {
    return RangePlan::rejected(
        saw_overflow ? "compute_range_aggregate_cost_overflow"
                     : "compute_range_aggregate_candidate_unavailable");
  }

  std::array<bool, kRangeCandidateCap> pareto{};
  std::size_t pareto_count = 0u;
  for (std::size_t index = 0u; index < evaluation_count; ++index) {
    bool dominated = false;
    for (std::size_t other = 0u; other < evaluation_count; ++other) {
      if (other != index &&
          Dominates(evaluations[other]->cost, evaluations[index]->cost)) {
        dominated = true;
        break;
      }
    }
    pareto[index] = !dominated;
    pareto_count += static_cast<std::size_t>(!dominated);
  }

  std::size_t selected = evaluation_count;
  for (std::size_t index = 0u; index < evaluation_count; ++index) {
    if (pareto[index] && (selected == evaluation_count ||
                          CompareLexicographic(*evaluations[index],
                                               *evaluations[selected]) < 0)) {
      selected = index;
    }
  }
  if (selected == evaluation_count || evaluation_count > 255u ||
      pareto_count > 255u) {
    return RangePlan::rejected("compute_range_aggregate_candidate_unavailable");
  }

  const CandidateEvaluation &choice = *evaluations[selected];
  const RangeIdentity source_identity =
      SourceIdentity(shape, capabilities, choice.candidate);
  const RangeIdentity execution_identity =
      ExecutionIdentity(shape, choice, source_identity);
  return RangePlan::selected(shape, choice.candidate,
                             capabilities.source_variant(), choice.cost,
                             choice.stage_count, choice.temporary_count,
                             static_cast<std::uint8_t>(evaluation_count),
                             static_cast<std::uint8_t>(pareto_count),
                             source_identity, execution_identity);
}

static_assert(std::is_nothrow_move_constructible_v<RangePlan>);
static_assert(std::is_nothrow_move_assignable_v<RangePlan>);

} // namespace rund::node::accel::detail
